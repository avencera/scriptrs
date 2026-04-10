use std::ffi::c_void;
use std::ptr::NonNull;
use std::slice;

use block2::RcBlock;
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2_core_ml::{MLMultiArray, MLMultiArrayDataType};
use objc2_foundation::{NSArray, NSNumber};

use crate::coreml::CachedInputShape;
use crate::error::TranscriptionError;

pub(super) fn ns_number_array(values: &[usize]) -> Retained<NSArray<NSNumber>> {
    let numbers: Vec<Retained<NSNumber>> = values
        .iter()
        .copied()
        .map(|value| NSNumber::new_isize(value as isize))
        .collect();
    NSArray::from_retained_slice(&numbers)
}

pub(super) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for index in (0..shape.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    strides
}

pub(super) fn multi_array_f32(
    values: &[f32],
    shape: &[usize],
    deallocator: &RcBlock<dyn Fn(NonNull<c_void>)>,
) -> Result<Retained<MLMultiArray>, TranscriptionError> {
    multi_array(
        values.as_ptr().cast::<c_void>() as *mut c_void,
        shape,
        MLMultiArrayDataType::Float32,
        deallocator,
    )
}

pub(super) fn multi_array_f32_cached(
    values: &[f32],
    cached: &CachedInputShape,
    deallocator: &RcBlock<dyn Fn(NonNull<c_void>)>,
) -> Result<Retained<MLMultiArray>, TranscriptionError> {
    multi_array_cached(
        values.as_ptr().cast::<c_void>() as *mut c_void,
        cached,
        MLMultiArrayDataType::Float32,
        deallocator,
    )
}

pub(super) fn multi_array_f32_strided_cached(
    ptr: *const f32,
    cached: &CachedInputShape,
    strides: &[usize],
    deallocator: &RcBlock<dyn Fn(NonNull<c_void>)>,
) -> Result<Retained<MLMultiArray>, TranscriptionError> {
    multi_array_cached_strides(
        ptr.cast::<c_void>() as *mut c_void,
        cached,
        strides,
        MLMultiArrayDataType::Float32,
        deallocator,
    )
}

pub(super) fn multi_array_i32(
    values: &[i32],
    shape: &[usize],
    deallocator: &RcBlock<dyn Fn(NonNull<c_void>)>,
) -> Result<Retained<MLMultiArray>, TranscriptionError> {
    multi_array(
        values.as_ptr().cast::<c_void>() as *mut c_void,
        shape,
        MLMultiArrayDataType::Int32,
        deallocator,
    )
}

pub(super) fn multi_array_i32_cached(
    values: &[i32],
    cached: &CachedInputShape,
    deallocator: &RcBlock<dyn Fn(NonNull<c_void>)>,
) -> Result<Retained<MLMultiArray>, TranscriptionError> {
    multi_array_cached(
        values.as_ptr().cast::<c_void>() as *mut c_void,
        cached,
        MLMultiArrayDataType::Int32,
        deallocator,
    )
}

fn multi_array(
    ptr: *mut c_void,
    shape: &[usize],
    data_type: MLMultiArrayDataType,
    deallocator: &RcBlock<dyn Fn(NonNull<c_void>)>,
) -> Result<Retained<MLMultiArray>, TranscriptionError> {
    let ptr = NonNull::new(ptr).ok_or_else(|| {
        TranscriptionError::CoreMl("input tensor had a null data pointer".to_owned())
    })?;
    let ns_shape = ns_number_array(shape);
    let ns_strides = ns_number_array(&contiguous_strides(shape));

    #[allow(deprecated)]
    // SAFETY: the pointer, shape, and contiguous strides all describe the same borrowed
    // SAFETY: tensor buffer, and CoreML only reads them during the prediction call
    unsafe {
        MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
            MLMultiArray::alloc(),
            ptr,
            &ns_shape,
            data_type,
            &ns_strides,
            Some(deallocator),
        )
    }
    .map_err(|error| TranscriptionError::CoreMl(format!("failed to create MLMultiArray: {error}")))
}

fn multi_array_cached(
    ptr: *mut c_void,
    cached: &CachedInputShape,
    data_type: MLMultiArrayDataType,
    deallocator: &RcBlock<dyn Fn(NonNull<c_void>)>,
) -> Result<Retained<MLMultiArray>, TranscriptionError> {
    let ptr = NonNull::new(ptr).ok_or_else(|| {
        TranscriptionError::CoreMl("input tensor had a null data pointer".to_owned())
    })?;

    #[allow(deprecated)]
    // SAFETY: the pointer, cached shape, and cached strides all describe the same borrowed
    // SAFETY: tensor buffer, and CoreML only reads them during the prediction call
    unsafe {
        MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
            MLMultiArray::alloc(),
            ptr,
            &cached.ns_shape,
            data_type,
            &cached.ns_strides,
            Some(deallocator),
        )
    }
    .map_err(|error| TranscriptionError::CoreMl(format!("failed to create MLMultiArray: {error}")))
}

fn multi_array_cached_strides(
    ptr: *mut c_void,
    cached: &CachedInputShape,
    strides: &[usize],
    data_type: MLMultiArrayDataType,
    deallocator: &RcBlock<dyn Fn(NonNull<c_void>)>,
) -> Result<Retained<MLMultiArray>, TranscriptionError> {
    let ptr = NonNull::new(ptr).ok_or_else(|| {
        TranscriptionError::CoreMl("input tensor had a null data pointer".to_owned())
    })?;
    let ns_strides = ns_number_array(strides);

    #[allow(deprecated)]
    // SAFETY: the pointer, cached shape, and supplied strides all describe the same borrowed
    // SAFETY: tensor buffer, and CoreML only reads them during the prediction call
    unsafe {
        MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
            MLMultiArray::alloc(),
            ptr,
            &cached.ns_shape,
            data_type,
            &ns_strides,
            Some(deallocator),
        )
    }
    .map_err(|error| TranscriptionError::CoreMl(format!("failed to create MLMultiArray: {error}")))
}

#[allow(deprecated)]
pub(super) fn extract_output(
    array: &MLMultiArray,
) -> Result<(Vec<f32>, Vec<usize>), TranscriptionError> {
    // SAFETY: CoreML guarantees these accessors describe the live MLMultiArray returned
    // SAFETY: by prediction, including the element count, dtype, and backing pointer
    let (count, ptr, dtype, shape, strides) = unsafe {
        (
            array.count() as usize,
            array.dataPointer(),
            array.dataType(),
            array.shape(),
            array.strides(),
        )
    };
    let shape: Vec<usize> = (0..shape.len())
        .map(|index| shape.objectAtIndex(index).as_isize() as usize)
        .collect();
    let strides: Vec<isize> = (0..strides.len())
        .map(|index| strides.objectAtIndex(index).as_isize())
        .collect();
    let data = match dtype {
        MLMultiArrayDataType::Float16 => {
            read_output(ptr.as_ptr() as *const u16, count, &shape, &strides)?
                .into_iter()
                .map(f16_to_f32)
                .collect()
        }
        MLMultiArrayDataType::Float32 => {
            read_output(ptr.as_ptr() as *const f32, count, &shape, &strides)?
        }
        MLMultiArrayDataType::Int32 => {
            read_output(ptr.as_ptr() as *const i32, count, &shape, &strides)?
                .iter()
                .copied()
                .map(|value| value as f32)
                .collect()
        }
        _ => {
            return Err(TranscriptionError::CoreMl(format!(
                "unsupported CoreML output dtype: {dtype:?}"
            )));
        }
    };
    Ok((data, shape))
}

fn read_output<T: Copy>(
    ptr: *const T,
    count: usize,
    shape: &[usize],
    strides: &[isize],
) -> Result<Vec<T>, TranscriptionError> {
    validate_shape_and_strides(shape, strides)?;

    if is_contiguous(shape, strides) {
        return Ok(read_contiguous(ptr, count));
    }

    read_strided(ptr, count, shape, strides)
}

fn validate_shape_and_strides(
    shape: &[usize],
    strides: &[isize],
) -> Result<(), TranscriptionError> {
    if shape.len() != strides.len() {
        return Err(TranscriptionError::CoreMl(format!(
            "shape/stride rank mismatch: shape={shape:?} strides={strides:?}"
        )));
    }
    Ok(())
}

fn is_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    contiguous_strides(shape)
        .into_iter()
        .map(|stride| stride as isize)
        .eq(strides.iter().copied())
}

fn read_contiguous<T: Copy>(ptr: *const T, count: usize) -> Vec<T> {
    if count == 0 {
        return Vec::new();
    }

    // SAFETY: contiguous outputs expose `count` initialized elements in row-major order
    unsafe { slice::from_raw_parts(ptr, count) }.to_vec()
}

fn read_strided<T: Copy>(
    ptr: *const T,
    count: usize,
    shape: &[usize],
    strides: &[isize],
) -> Result<Vec<T>, TranscriptionError> {
    let mut values = Vec::with_capacity(count);
    for linear_index in 0..count {
        let offset = linear_offset(linear_index, shape, strides)?;
        // SAFETY: `offset` is computed from the live MLMultiArray shape/strides and points
        // SAFETY: to the logical element for this row-major linear index
        values.push(unsafe { *ptr.offset(offset) });
    }

    Ok(values)
}

fn linear_offset(
    mut linear_index: usize,
    shape: &[usize],
    strides: &[isize],
) -> Result<isize, TranscriptionError> {
    let mut offset = 0isize;
    for dimension in (0..shape.len()).rev() {
        let size = shape[dimension];
        if size == 0 {
            return Err(TranscriptionError::CoreMl(
                "CoreML output had a zero-sized dimension".to_owned(),
            ));
        }
        let index = linear_index % size;
        linear_index /= size;
        offset += strides[dimension]
            .checked_mul(index as isize)
            .ok_or_else(|| {
                TranscriptionError::CoreMl("CoreML stride offset overflowed".to_owned())
            })?;
    }
    Ok(offset)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut exponent = exp as i32;
        let mut mantissa = mantissa;
        while mantissa & 0x400 == 0 {
            mantissa <<= 1;
            exponent -= 1;
        }
        mantissa &= 0x3ff;
        let f32_exp = ((127 - 15) + exponent + 1) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13));
    }

    if exp == 0x1f {
        return f32::from_bits((sign << 31) | (0xff_u32 << 23) | (mantissa << 13));
    }

    let f32_exp = (exp as i32) - 15 + 127;
    f32::from_bits((sign << 31) | ((f32_exp as u32) << 23) | (mantissa << 13))
}
