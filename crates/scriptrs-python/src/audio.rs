//! Decoding audio files to the 16 kHz mono samples scriptrs takes
//!
//! Ported from speakrs-sdk, which decodes the same formats for diarization

use std::path::Path;

use rubato::audioadapter_buffers::owned::InterleavedOwned;
use rubato::{Fft, FixedSync, Resampler};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::packet::Packet;

const SAMPLE_RATE_HZ: u32 = 16_000;

/// Decode an audio file, downmix it to mono, and resample it to 16 kHz
///
/// WAV, FLAC, MP3, Ogg Vorbis, and AAC in MP4/M4A are supported
pub(crate) fn decode_file_to_mono_16khz(path: &Path) -> Result<Vec<f32>, String> {
    let (interleaved, sample_rate_hz, channels) = decode_interleaved(path)?;
    let mono = match channels {
        1 => interleaved,
        _ => interleaved
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect(),
    };
    resample_mono_to_16khz(mono, sample_rate_hz)
}

fn resample_mono_to_16khz(samples: Vec<f32>, sample_rate_hz: u32) -> Result<Vec<f32>, String> {
    if sample_rate_hz == 0 {
        return Err(format!("invalid sample rate {sample_rate_hz}"));
    }

    if sample_rate_hz == SAMPLE_RATE_HZ {
        return Ok(samples);
    }

    let resample_err = |error: &dyn std::fmt::Display| format!("failed to resample audio: {error}");
    let input_len = samples.len();
    let input =
        InterleavedOwned::new_from(samples, 1, input_len).map_err(|err| resample_err(&err))?;
    // the FFT resampler low-passes before decimating, so content above 8 kHz
    // cannot fold back into the band the model hears
    let mut resampler = Fft::<f32>::new(
        sample_rate_hz as usize,
        SAMPLE_RATE_HZ as usize,
        1024,
        1,
        FixedSync::Input,
    )
    .map_err(|err| resample_err(&err))?;
    let output_capacity = resampler.process_all_needed_output_len(input_len);
    let mut output = InterleavedOwned::new(0.0, 1, output_capacity);
    let (_, output_len) = resampler
        .process_all_into_buffer(&input, &mut output, input_len, None)
        .map_err(|err| resample_err(&err))?;

    let mut samples = output.take_data();
    samples.truncate(output_len);
    Ok(samples)
}

/// Interleaved samples, sample rate, and channel count of a file's audio track
fn decode_interleaved(path: &Path) -> Result<(Vec<f32>, u32, usize), String> {
    let display = path.display();
    let src = std::fs::File::open(path)
        .map_err(|err| format!("failed to open audio file `{display}`: {err}"))?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    // the extension only speeds up probing; symphonia still checks the container
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|err| format!("failed to probe audio file `{display}`: {err}"))?;
    let missing_track = || format!("audio file `{display}` has no decodable audio track");
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(missing_track)?;
    let track_id = track.id;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(missing_track)?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|err| format!("unsupported audio codec in `{display}`: {err}"))?;

    let mut samples = Vec::new();
    let mut sample_rate_hz = None;
    let mut channels = None;
    while let Some(packet) = next_packet(&mut *format, path)? {
        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = decoded.spec();
                let current_channels = spec.channels().count();
                if current_channels == 0 {
                    return Err(format!(
                        "audio file `{display}` has a packet with no channels"
                    ));
                }

                sample_rate_hz.get_or_insert(spec.rate());
                channels.get_or_insert(current_channels);

                let start = samples.len();
                samples.resize(start + decoded.samples_interleaved(), 0.0);
                decoded.copy_to_slice_interleaved(&mut samples[start..]);
            }
            // a corrupt packet is skipped, as players do
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => return Err(format!("failed to decode audio file `{display}`: {err}")),
        }
    }

    if samples.is_empty() {
        return Err(format!("audio file `{display}` contains no samples"));
    }

    Ok((
        samples,
        sample_rate_hz.unwrap_or(SAMPLE_RATE_HZ),
        channels.unwrap_or(1),
    ))
}

fn next_packet(format: &mut dyn FormatReader, path: &Path) -> Result<Option<Packet>, String> {
    match format.next_packet() {
        Ok(packet) => Ok(packet),
        Err(SymphoniaError::IoError(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
            Ok(None)
        }
        Err(err) => Err(format!(
            "failed to read audio file `{}`: {err}",
            path.display()
        )),
    }
}
