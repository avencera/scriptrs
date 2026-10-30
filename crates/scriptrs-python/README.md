# scriptrs

scriptrs is fast speech-to-text with NVIDIA's Parakeet TDT 0.6B v2 and v3 models.
Inference runs in Rust on Apple Core ML or NVIDIA CUDA, with no PyTorch or ONNX
Runtime.

```sh
pip install scriptrs
```

| Platform | Backend |
| --- | --- |
| macOS arm64 | Core ML |
| Linux x86_64 (glibc 2.28+) | CUDA, on a Turing (RTX 20, T4) or newer NVIDIA GPU |

The package needs Python 3.11 or newer. One wheel covers every Python version.

## Use

```python
import scriptrs

pipeline = scriptrs.Pipeline.from_pretrained("parakeet/v2")  # downloads the model once
result = pipeline.transcribe_file("meeting.mp3")

print(result.text)
for chunk in result.chunks:
    print(f"{chunk.start:7.2f} {chunk.end:7.2f} {chunk.text}")
```

## Choosing a model

Models are named `"<family>/<version>"`. Both Parakeet models come with the same
package, and you pick one when you load the pipeline:

| Model | Languages | Pick it when |
| --- | --- | --- |
| `"parakeet/v2"` (default) | English only | all your audio is English; v2 is slightly more accurate on it |
| `"parakeet/v3"` | 25 European languages, detected automatically | the audio may not be English |

```python
english = scriptrs.Pipeline.from_pretrained("parakeet/v2")
multilingual = scriptrs.Pipeline.from_pretrained("parakeet/v3")
```

`scriptrs.Model.PARAKEET_V2` and `scriptrs.Model.PARAKEET_V3` work too, and so do
NVIDIA's model ids, such as `"nvidia/parakeet-tdt-0.6b-v3"`. Each model downloads
separately the first time you load it, so loading only the one you use downloads
only that one.

Parakeet v3 detects the language of each window by itself. To fix it, pass an
ISO 639-1 code:

```python
result = multilingual.transcribe_file("interview.wav", language="de")
print(scriptrs.Model.PARAKEET_V3.languages)  # ['bg', 'hr', 'cs', ...]
```

## Audio

`transcribe_file` reads WAV, FLAC, MP3, Ogg Vorbis, and AAC in MP4/M4A, and
converts it to 16 kHz mono. Audio already in memory has to be 16 kHz mono
float32, such as a NumPy array:

```python
result = pipeline.transcribe(samples, language="en")
```

Recordings of any length work. Audio longer than one model window is split into
overlapping windows and the transcripts are merged. For recordings that are
mostly silence or noise, load the pipeline with `vad=True` to split at detected
speech instead.

To transcribe many clips, such as utterances, use `transcribe_many`. It
prepares the next clips while the model runs and, on CUDA, batches the clips'
windows together, up to 32 in one pass:

```python
results = pipeline.transcribe_many([clip_a, clip_b, clip_c])
```

Each result has the full `text`, timed `chunks`, and every token in `tokens`
with its start, end, and confidence.

## Command line

```sh
scriptrs transcribe meeting.mp3
scriptrs transcribe interview.wav --model parakeet/v3 --language de --json
```

With one file, the transcript is printed alone. With several, each one follows a
header, as `head` prints them:

```
$ scriptrs transcribe a.wav b.wav
==> a.wav <==
First transcript.

==> b.wav <==
Second transcript.
```

`--json` prints one JSON object per line, with the file, text, timed chunks, and
tokens.

To install only the command line, in its own environment, use
[uv](https://docs.astral.sh/uv/):

```sh
uv tool install scriptrs
```

## CUDA

The Linux wheel needs an NVIDIA driver, but no CUDA toolkit: it installs cuBLAS
from the `nvidia-cublas-cu12` package. The default `precision="fp16"` is the
fastest. `"tf32"` keeps the weights in FP32, and `"fp32"` matches NVIDIA NeMo's
FP32 output token for token:

```python
pipeline = scriptrs.Pipeline.from_pretrained("parakeet/v3", precision="fp32")
```

## Models

`from_pretrained()` downloads the models from Hugging Face into `cache_dir=`
when given, else the Hugging Face hub cache that `HF_HUB_CACHE` or `HF_HOME`
sets, else `~/.cache/huggingface/hub`. On macOS a model is about
0.6 GB of Core ML models from
[`avencera/scriptrs-models`](https://huggingface.co/avencera/scriptrs-models).
On Linux it is about 2.5 GB of weights from
[`avencera/parakeet-tdt-0.6b-v2-cuda`](https://huggingface.co/avencera/parakeet-tdt-0.6b-v2-cuda)
or [`avencera/parakeet-tdt-0.6b-v3-cuda`](https://huggingface.co/avencera/parakeet-tdt-0.6b-v3-cuda).

To use models you manage yourself, load them with
`scriptrs.Pipeline.from_dir(path, "parakeet/v2")`, using the layout in the
[scriptrs README](https://github.com/avencera/scriptrs#local-model-layout).

## Errors

Invalid arguments, such as an unknown model name or samples that are not 1-D
float32, raise `ValueError` or `TypeError`. Everything else that fails raises
`scriptrs.ScriptrsError`. Its `code` is a stable value, such as
`unsupported_language`, `model_download`, `audio_decode`, or `cuda`, so you can
branch on it without parsing the message.
