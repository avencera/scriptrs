//! Low-precision formats for the encoder's linear layers, on the host
//!
//! The weight quantizers here and the `quant` kernels in
//! `crates/scriptrs-cuda-kernels/src/quant.rs` run the same arithmetic in the same
//! order, so a weight quantized at load, a weight read from a quantized file and an
//! activation quantized on the GPU land on the same bits for the same values. Every
//! format rounds to nearest even and saturates at its largest finite value.
//!
//! NVFP4 block scales use cuBLASLt's tiled layout (see [`ScaleLayout`]); rows of every
//! linear layer and operand are blocked along the input dimension, which is the
//! contiguous one in both the PyTorch `[out, in]` weight and the `[rows, in]`
//! activation. The launchers of the `quant` kernel area live here too

use cudakit::cudarc::driver::{
    CudaFunction, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg,
};
use cudakit::{Error, LoadedKernels, check_len};

use crate::cuda::kernels::{
    CudaRuntime, ELEMENT_THREADS, KernelModule, elements, len, len_mut, u32_of,
};

/// E4M3: 3 mantissa bits, bias 7
const E4M3: (u32, i32) = (3, 7);
/// Largest finite E4M3 value
pub(crate) const E4M3_MAX: f32 = 448.0;
/// E2M1: 1 mantissa bit, bias 1
const E2M1: (u32, i32) = (1, 1);
/// Largest E2M1 value
pub(crate) const E2M1_MAX: f32 = 6.0;
/// NVFP4 values per E4M3 scale
pub(crate) const NV_BLOCK: usize = 16;
/// INT8 weight values per scale
pub(crate) const INT8_GROUP: usize = 128;
/// What every linear layer's input count must be a multiple of: whole INT8 groups and
/// whole 4-block tiles of NVFP4 scales
pub(crate) const INPUT_MULTIPLE: usize = INT8_GROUP;

/// How a linear layer's weight and the activation it reads are quantized
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Scheme {
    /// FP8 E4M3 with one scale for the weight and one dynamic scale for the
    /// activation
    Fp8,
    /// NVFP4: E2M1 in blocks of 16 with E4M3 scales and a tensor scale, for both
    /// operands
    Nvfp4,
    /// INT8 weights with a scale per [`INT8_GROUP`] inputs, dequantized to FP16 for an
    /// FP16 product
    Int8,
}

impl Scheme {
    /// Quantizes a row-major `[out, inputs]` weight, with `inputs` a multiple of
    /// [`INPUT_MULTIPLE`]
    pub(crate) fn weight(self, weight: &[f32], out: usize, inputs: usize) -> QuantizedWeight {
        assert_eq!(weight.len(), out * inputs, "weight size");
        assert_eq!(inputs % INPUT_MULTIPLE, 0, "weight inputs");
        match self {
            Self::Fp8 => {
                let (values, scale) = fp8_tensor(weight);
                QuantizedWeight::Fp8 { values, scale }
            }
            Self::Nvfp4 => {
                let (values, scales, global) = nvfp4(weight, inputs);
                QuantizedWeight::Nvfp4 {
                    values,
                    scales,
                    global,
                }
            }
            Self::Int8 => {
                let (values, scales) = int8_groups(weight, INT8_GROUP);
                QuantizedWeight::Int8 { values, scales }
            }
        }
    }

    /// Whether the product reads FP16 activations and a dequantized FP16 weight
    pub(crate) fn weight_only(self) -> bool {
        self == Self::Int8
    }

    /// The name of the scheme in a quantized weight file's metadata
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Fp8 => "fp8",
            Self::Nvfp4 => "nvfp4",
            Self::Int8 => "int8",
        }
    }

    /// The scheme a quantized weight file's metadata names
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        [Self::Fp8, Self::Nvfp4, Self::Int8]
            .into_iter()
            .find(|scheme| scheme.name() == name)
    }
}

/// A quantized `[out, inputs]` weight as it is uploaded
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum QuantizedWeight {
    /// E4M3 bits and one FP32 scale
    Fp8 { values: Vec<u8>, scale: f32 },
    /// E2M1 pairs, tiled E4M3 block scales and the tensor scale
    Nvfp4 {
        values: Vec<u8>,
        scales: Vec<u8>,
        global: f32,
    },
    /// INT8 values and a scale per [`INT8_GROUP`] inputs
    Int8 { values: Vec<u8>, scales: Vec<f32> },
}

/// cuBLASLt's tiled layout of the `[rows, inputs / block]` block scales of a matrix
///
/// The scale matrix is padded to 128-row, 4-column tiles of 512 bytes, laid out
/// row-major over the tiles; inside a tile, row `r` and column `c` sit at
/// `(r % 32) * 16 + (r / 32) * 4 + c`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScaleLayout {
    tile_rows: usize,
    tile_columns: usize,
}

impl ScaleLayout {
    pub(crate) fn new(rows: usize, inputs: usize, block: usize) -> Self {
        Self {
            tile_rows: rows.div_ceil(128),
            tile_columns: (inputs / block).div_ceil(4),
        }
    }

    pub(crate) fn tile_columns(self) -> usize {
        self.tile_columns
    }

    /// Bytes of the padded scale matrix
    pub(crate) fn bytes(self) -> usize {
        self.tile_rows * self.tile_columns * 512
    }

    /// Byte offset of the scale of block `column` in `row`
    pub(crate) fn offset(self, row: usize, column: usize) -> usize {
        let tile = (row / 128) * self.tile_columns + column / 4;
        let r = row % 128;
        tile * 512 + (r % 32) * 16 + (r / 32) * 4 + column % 4
    }
}

/// `2^exponent` for `-126 <= exponent <= 127`
fn pow2(exponent: i32) -> f32 {
    f32::from_bits(((exponent + 127) as u32) << 23)
}

/// Encodes `a`, finite, non-negative and at most the format's largest value, in a
/// minifloat with `mbits` mantissa bits and exponent bias `bias`, subnormals included,
/// rounding to nearest even
fn encode(a: f32, (mbits, bias): (u32, i32)) -> u32 {
    let exponent = ((a.to_bits() >> 23) & 0xff) as i32 - 127;
    let emin = 1 - bias;
    let mut ulp = exponent.max(emin) - mbits as i32;
    // exact: a power-of-two scale, and the result is below 2^(mbits + 1)
    let mut q = (a * pow2(-ulp)).round_ties_even() as u32;
    let one = 1u32 << mbits;
    if q == 2 * one {
        q = one;
        ulp += 1;
    }
    if q < one {
        q
    } else {
        (((ulp + mbits as i32 + bias) as u32) << mbits) | (q - one)
    }
}

/// The value of an unsigned minifloat code from [`encode`]
fn decode(code: u32, (mbits, bias): (u32, i32)) -> f32 {
    let exponent = (code >> mbits) as i32;
    let mantissa = code & ((1 << mbits) - 1);
    if exponent == 0 {
        mantissa as f32 * pow2(1 - bias - mbits as i32)
    } else {
        ((1 << mbits) + mantissa) as f32 * pow2(exponent - bias - mbits as i32)
    }
}

/// Signed E4M3 bits of `value`, saturating at ±448
pub(crate) fn e4m3(value: f32) -> u8 {
    let sign = if value < 0.0 { 0x80 } else { 0 };
    (sign | encode(value.abs().min(E4M3_MAX), E4M3)) as u8
}

/// Signed E2M1 bits of `value` in the low nibble, saturating at ±6
fn e2m1(value: f32) -> u8 {
    let sign = if value < 0.0 { 0x8 } else { 0 };
    (sign | encode(value.abs().min(E2M1_MAX), E2M1)) as u8
}

fn amax(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()))
}

/// E4M3 bits of `values` with one scale, `amax / 448`
pub(crate) fn fp8_tensor(values: &[f32]) -> (Vec<u8>, f32) {
    let amax = amax(values);
    let scale = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
    (
        values.iter().map(|value| e4m3(value / scale)).collect(),
        scale,
    )
}

/// NVFP4 of rows of `width`: E2M1 pairs (even element in the low nibble), tiled E4M3
/// block scales relative to the tensor scale, and the tensor scale `amax / (448 * 6)`
pub(crate) fn nvfp4(values: &[f32], width: usize) -> (Vec<u8>, Vec<u8>, f32) {
    let rows = values.len() / width;
    let layout = ScaleLayout::new(rows, width, NV_BLOCK);
    let amax = amax(values);
    let global = if amax > 0.0 {
        amax / (E4M3_MAX * E2M1_MAX)
    } else {
        1.0
    };
    let mut scales = vec![0u8; layout.bytes()];
    let mut bits = Vec::with_capacity(values.len() / 2);
    for (b, block) in values.chunks(NV_BLOCK).enumerate() {
        let code = encode((self::amax(block) / E2M1_MAX / global).min(E4M3_MAX), E4M3);
        let block_scale = decode(code, E4M3) * global;
        let per_row = width / NV_BLOCK;
        scales[layout.offset(b / per_row, b % per_row)] = code as u8;
        for pair in block.chunks(2) {
            let (low, high) = if block_scale > 0.0 {
                (e2m1(pair[0] / block_scale), e2m1(pair[1] / block_scale))
            } else {
                (0, 0)
            };
            bits.push(low | (high << 4));
        }
    }
    (bits, scales, global)
}

/// Symmetric INT8 in groups of `group`, with a scale per group of `amax / 127`
fn int8_groups(values: &[f32], group: usize) -> (Vec<u8>, Vec<f32>) {
    let mut bits = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(values.len() / group);
    for block in values.chunks(group) {
        let amax = amax(block);
        let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        bits.extend(
            block
                .iter()
                .map(|value| (value / scale).round_ties_even().clamp(-127.0, 127.0) as i8 as u8),
        );
        scales.push(scale);
    }
    (bits, scales)
}

/// FP16 bits of `value`, rounded to nearest even, overflowing to infinity past ±65504
pub(crate) fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let a = value.abs();
    if a.is_nan() {
        return sign | 0x7e00;
    }
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    // FP16 is a minifloat with 10 mantissa bits and bias 15
    sign | encode(a, (10, 15)) as u16
}

/// The value of FP16 bits
pub(crate) fn f16_value(bits: u16) -> f32 {
    let magnitude = decode(u32::from(bits & 0x7fff), (10, 15));
    let magnitude = match bits & 0x7c00 {
        0x7c00 if bits & 0x3ff == 0 => f32::INFINITY,
        0x7c00 => f32::NAN,
        _ => magnitude,
    };
    if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// Per-block partial maxima the `quant_amax` kernel writes; the quantize kernels that
/// fold them run [`ELEMENT_THREADS`] threads per block, at least this many
pub(crate) const AMAX_PARTIALS: usize = 256;

const _: () = assert!(ELEMENT_THREADS as usize >= AMAX_PARTIALS);

/// The `quant` kernel area: activation quantizers and weight dequantizers
pub(crate) struct QuantKernels {
    amax: CudaFunction,
    fp8: CudaFunction,
    nvfp4: CudaFunction,
    dequant_i8: CudaFunction,
}

impl QuantKernels {
    pub(crate) fn load(runtime: &CudaRuntime) -> Result<Self, Error> {
        let module: LoadedKernels<KernelModule> = runtime.load_embedded(KernelModule::Quant)?;
        Ok(Self {
            amax: module.function("quant_amax")?,
            fp8: module.function("quant_fp8")?,
            nvfp4: module.function("quant_nvfp4")?,
            dequant_i8: module.function("quant_dequant_i8")?,
        })
    }

    /// Per-block maxima of `|x|` into the [`AMAX_PARTIALS`] `partials`
    pub(crate) fn amax(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, u16>,
        partials: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), Error> {
        check_len("amax partials", AMAX_PARTIALS, partials.len())?;
        let config = LaunchConfig {
            grid_dim: (AMAX_PARTIALS as u32, 1, 1),
            block_dim: (ELEMENT_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let lens = [len(x), len_mut(partials)];
        let mut launch = stream.launch_builder(&self.amax);
        launch.arg(x).arg(&lens[0]).arg(partials).arg(&lens[1]);
        // SAFETY: matches `quant_amax`: one block per partial, and the grid-stride
        // loop bounds-checks against `x`
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// E4M3 of `x` with one scale from `partials`, written to `scale`
    pub(crate) fn fp8(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, u16>,
        partials: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, u8>,
        scale: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), Error> {
        check_len("fp8 partials", AMAX_PARTIALS, partials.len())?;
        check_len("fp8 output", x.len(), out.len())?;
        check_len("fp8 scale", 1, scale.len())?;
        let config = elements(x.len())?;
        let lens = [len(x), len(partials), len_mut(out), len_mut(scale)];
        let mut launch = stream.launch_builder(&self.fp8);
        launch
            .arg(x)
            .arg(&lens[0])
            .arg(partials)
            .arg(&lens[1])
            .arg(out)
            .arg(&lens[2])
            .arg(scale)
            .arg(&lens[3]);
        // SAFETY: matches `quant_fp8`: one thread per element in full blocks of
        // `ELEMENT_THREADS`, the first of which writes `scale`
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// NVFP4 of rows of `width` with the tensor scale from `partials`, and `alpha[0] =
    /// alpha_base * tensor scale`
    #[allow(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn nvfp4(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, u16>,
        partials: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, u8>,
        scales: &mut CudaViewMut<'_, u8>,
        alpha: &mut CudaViewMut<'_, f32>,
        width: usize,
        alpha_base: f32,
    ) -> Result<(), Error> {
        let layout = block_operand("nvfp4", x.len(), width, NV_BLOCK)?;
        check_len("nvfp4 partials", AMAX_PARTIALS, partials.len())?;
        check_len("nvfp4 output", x.len() / 2, out.len())?;
        check_len("nvfp4 scales", layout.bytes(), scales.len())?;
        check_len("nvfp4 alpha", 1, alpha.len())?;
        let config = elements(x.len() / NV_BLOCK)?;
        let lens = [
            len(x),
            len(partials),
            len_mut(out),
            len_mut(scales),
            len_mut(alpha),
        ];
        let width = u32_of("nvfp4 width", width)?;
        let tiles = u32_of("nvfp4 tiles", layout.tile_columns())?;
        let mut launch = stream.launch_builder(&self.nvfp4);
        launch
            .arg(x)
            .arg(&lens[0])
            .arg(partials)
            .arg(&lens[1])
            .arg(out)
            .arg(&lens[2])
            .arg(scales)
            .arg(&lens[3])
            .arg(alpha)
            .arg(&lens[4])
            .arg(&width)
            .arg(&tiles)
            .arg(&alpha_base);
        // SAFETY: matches `quant_nvfp4`: one thread per block of 16 in full blocks of
        // `ELEMENT_THREADS`, bounds-checked against `x`, and every scale offset lies
        // inside `layout`
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// FP16 of an INT8 weight with a scale per [`INT8_GROUP`] elements
    pub(crate) fn dequant_i8(
        &self,
        stream: &CudaStream,
        q: &CudaView<'_, u8>,
        scales: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, u16>,
    ) -> Result<(), Error> {
        check_len("int8 weight", out.len(), q.len())?;
        check_len("int8 scales", out.len() / INT8_GROUP, scales.len())?;
        check_len("int8 groups", scales.len() * INT8_GROUP, out.len())?;
        let config = elements(out.len())?;
        let lens = [len(q), len(scales), len_mut(out)];
        let group = u32_of("int8 group", INT8_GROUP)?;
        let mut launch = stream.launch_builder(&self.dequant_i8);
        launch
            .arg(q)
            .arg(&lens[0])
            .arg(scales)
            .arg(&lens[1])
            .arg(out)
            .arg(&lens[2])
            .arg(&group);
        // SAFETY: matches `quant_dequant_i8`; the kernel bounds-checks against `out`
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}

/// The scale layout of a `[len / width, width]` operand blocked by `block`
fn block_operand(
    context: &'static str,
    len: usize,
    width: usize,
    block: usize,
) -> Result<ScaleLayout, Error> {
    if width == 0 || width % (4 * block) != 0 || len % width != 0 {
        return Err(Error::BufferLength {
            context,
            expected: width.max(1) * len.div_ceil(width.max(1)),
            actual: len,
        });
    }
    Ok(ScaleLayout::new(len / width, width, block))
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::{E2M1, E4M3, INT8_GROUP, NV_BLOCK, QuantizedWeight, ScaleLayout, decode};

    impl QuantizedWeight {
        /// Bytes the weight takes on the device and in a quantized weight file
        pub(crate) fn bytes(&self) -> usize {
            match self {
                Self::Fp8 { values, .. } => values.len() + 4,
                Self::Int8 { values, scales } => values.len() + 4 * scales.len(),
                Self::Nvfp4 { values, scales, .. } => values.len() + scales.len() + 4,
            }
        }

        /// The values the quantized weight stands for, row-major `[out, inputs]`
        pub(crate) fn dequantize(&self, out: usize, inputs: usize) -> Vec<f32> {
            let n = out * inputs;
            match self {
                Self::Fp8 { values, scale } => values
                    .iter()
                    .map(|bits| e4m3_value(*bits) * scale)
                    .collect(),
                Self::Nvfp4 {
                    values,
                    scales,
                    global,
                } => {
                    let layout = ScaleLayout::new(out, inputs, NV_BLOCK);
                    (0..n)
                        .map(|i| {
                            let code = scales[layout.offset(i / inputs, i % inputs / NV_BLOCK)];
                            let nibble = values[i / 2] >> (4 * (i % 2)) & 0xf;
                            e2m1_value(nibble) * decode(u32::from(code), E4M3) * global
                        })
                        .collect()
                }
                Self::Int8 { values, scales } => (0..n)
                    .map(|i| f32::from(values[i] as i8) * scales[i / INT8_GROUP])
                    .collect(),
            }
        }
    }

    /// The value of signed E4M3 `bits`
    pub(crate) fn e4m3_value(bits: u8) -> f32 {
        let magnitude = decode(u32::from(bits & 0x7f), E4M3);
        if bits & 0x80 == 0 {
            magnitude
        } else {
            -magnitude
        }
    }

    /// The value of a signed E2M1 nibble
    pub(crate) fn e2m1_value(nibble: u8) -> f32 {
        let magnitude = decode(u32::from(nibble & 0x7), E2M1);
        if nibble & 0x8 == 0 {
            magnitude
        } else {
            -magnitude
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        E4M3_MAX, QuantizedWeight, ScaleLayout, Scheme, e2m1, e4m3, f16_bits, f16_value,
        fp8_tensor, nvfp4,
    };
    use crate::cuda::quant::test_support::{e2m1_value, e4m3_value};

    /// Deterministic weights in about the range of the encoder's: mostly small, with
    /// a few outliers
    fn weights(len: usize) -> Vec<f32> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        (0..len)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                if i % 997 == 0 { unit * 4.0 } else { unit * 0.1 }
            })
            .collect()
    }

    #[test]
    fn e4m3_matches_the_format_table() {
        // every finite E4M3 code decodes and encodes back to itself
        for bits in (0u8..=0xff).filter(|bits| bits & 0x7f != 0x7f) {
            let value = e4m3_value(bits);
            let expected = if bits == 0x80 { 0 } else { bits };
            assert_eq!(e4m3(value), expected, "{bits:#04x} = {value}");
        }
        assert_eq!(e4m3_value(0x7e), E4M3_MAX);
        assert_eq!(e4m3_value(0x01), 2f32.powi(-9));
        assert_eq!(e4m3_value(0x38), 1.0);
        assert_eq!(e4m3(1e6), 0x7e, "saturates");
        // halfway between 1.0 and 1.125 rounds to the even 1.0; past it rounds up
        assert_eq!(e4m3(1.0625), 0x38);
        assert_eq!(e4m3(1.0626), 0x39);
        // halfway between the two smallest subnormals rounds to the even 2^-8
        assert_eq!(e4m3(1.5 * 2f32.powi(-9)), 0x02);
    }

    #[test]
    fn e2m1_matches_the_format_table() {
        let magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, magnitude) in magnitudes.iter().enumerate() {
            assert_eq!(e2m1_value(code as u8), *magnitude);
            assert_eq!(e2m1(*magnitude), code as u8);
            assert_eq!(e2m1(-magnitude), if code == 0 { 0 } else { code as u8 | 8 });
        }
        // 5 is halfway between 4 and 6 and rounds to 4, whose mantissa is even
        assert_eq!(e2m1(5.0), 6);
        assert_eq!(e2m1(2.5), 4);
        assert_eq!(e2m1(0.25), 0);
        assert_eq!(e2m1(0.26), 1);
        assert_eq!(e2m1(100.0), 7);
    }

    #[test]
    fn f16_round_trips_and_rounds_to_even() {
        for bits in (0u16..=0xffff).filter(|bits| bits & 0x7c00 != 0x7c00) {
            let value = f16_value(bits);
            let expected = if bits == 0x8000 { 0x8000 } else { bits };
            assert_eq!(f16_bits(value), expected, "{bits:#06x} = {value}");
        }
        assert_eq!(f16_bits(65504.0), 0x7bff);
        assert_eq!(f16_bits(65520.0), 0x7c00);
        assert_eq!(f16_bits(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f16_bits(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert_eq!(f16_value(0x7c00), f32::INFINITY);
    }

    #[test]
    fn scale_layout_places_each_block_once() {
        let (rows, inputs) = (256, 1024);
        let layout = ScaleLayout::new(rows, inputs, 16);
        assert_eq!(layout.bytes(), rows * inputs / 16);
        let mut seen = vec![false; layout.bytes()];
        for row in 0..rows {
            for column in 0..inputs / 16 {
                let at = layout.offset(row, column);
                assert!(!seen[at], "({row}, {column}) collides");
                seen[at] = true;
            }
        }
        // the four scales of a row's first tile are contiguous, and rows 32 apart
        // interleave within 16 bytes
        assert_eq!(layout.offset(0, 1), 1);
        assert_eq!(layout.offset(32, 0), 4);
        assert_eq!(layout.offset(1, 0), 16);
        assert_eq!(layout.offset(0, 4), 512);
        assert_eq!(layout.offset(128, 0), 512 * layout.tile_columns());
        // 70 rows pad to one 128-row tile
        assert_eq!(ScaleLayout::new(70, 1024, 16).bytes(), 128 * 64);
    }

    /// Each scheme's weight error stays within its format's rounding bound, so a
    /// misplaced scale or nibble shows up as an error of order the weight itself
    #[test]
    fn quantized_weights_stay_within_their_rounding_error() {
        let (out, inputs) = (256, 512);
        let weight = weights(out * inputs);
        let norm = weight.iter().map(|w| w * w).sum::<f32>().sqrt();
        let schemes = [
            (Scheme::Fp8, 0.08),
            (Scheme::Nvfp4, 0.15),
            (Scheme::Int8, 0.03),
        ];
        for (scheme, bound) in schemes {
            let quantized = scheme.weight(&weight, out, inputs);
            let restored = quantized.dequantize(out, inputs);
            let error = weight
                .iter()
                .zip(&restored)
                .map(|(w, r)| (w - r) * (w - r))
                .sum::<f32>()
                .sqrt()
                / norm;
            println!(
                "{scheme:?}: relative error {error:.4}, {} bytes",
                quantized.bytes()
            );
            assert!(error < bound, "{scheme:?} relative error {error}");
            assert!(
                error > 0.0,
                "{scheme:?} is lossless, so it did not quantize"
            );
            assert_eq!(Scheme::from_name(scheme.name()), Some(scheme));
        }
    }

    #[test]
    fn nvfp4_saturates_no_block() {
        let weight = weights(64 * 64);
        let (values, scales, global) = nvfp4(&weight, 64);
        let restored = QuantizedWeight::Nvfp4 {
            values,
            scales,
            global,
        }
        .dequantize(64, 64);
        for (block, restored) in weight.chunks(16).zip(restored.chunks(16)) {
            let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
            let restored_max = restored.iter().fold(0f32, |m, v| m.max(v.abs()));
            // the block maximum lands within E4M3's rounding of its scale
            assert!(
                (restored_max - amax).abs() <= amax * 0.07,
                "block amax {amax} restored as {restored_max}"
            );
        }
    }

    #[test]
    fn zero_tensors_quantize_to_zero() {
        let (values, scale) = fp8_tensor(&[0.0; 32]);
        assert!(values.iter().all(|bits| *bits == 0));
        assert_eq!(scale, 1.0);
    }
}
