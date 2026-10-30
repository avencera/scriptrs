"""NeMo cache-aware streaming references for the native streaming engine

Runs the loop of NeMo's `speech_to_text_cache_aware_streaming_infer.py` one utterance
at a time over LibriSpeech test-clean: the whole utterance's mel sliced into chunks by
`CacheAwareStreamingAudioBuffer`, `conformer_stream_step` per chunk with the caches and
partial hypotheses carried, greedy batched RNNT decoding as the script configures it.
Batch 1 keeps padding out of the reference. NeMo runs in FP32 with TF32 off, stricter
than the script's default `matmul_precision=high`

For each lookahead R (`att_context_size` [left, R]) it writes, under <out>/<model>/:

    r<R>.texts.json   {id: text}
    r<R>.hyps.json    {id: {"tokens": [...], "timestamps": [...]}}
    r<R>.timing.json  per-chunk `conformer_stream_step` wall time, GPU synced
    r<R>.dump.safetensors   with --dump N: the first N utterances' audio, whole mel,
                            and per chunk its mel input, drop, every layer's output and
                            the encoder output, plus the final tokens

Needs the GPU: submit it to the homebased gpu0 queue
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file

LS_MANIFEST = "/shared/datasets/librispeech-test-clean/manifest.jsonl"
MODELS = {
    "nemotron-en": "/shared/models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo",
    "nemotron-3.5": "/shared/models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="nemotron-en")
    parser.add_argument("--lookahead", type=int, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Only the first N manifest rows")
    parser.add_argument("--dump", type=int, default=0, help="Tensor dumps for the first N rows")
    parser.add_argument("--lang", default="en-US", help="Prompt language of a prompted model")
    return parser.parse_args()


def load_model(name: str, lang: str):
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
    from nemo.utils import logging as nemo_logging

    nemo_logging.setLevel(logging.ERROR)
    model = nemo_asr.models.ASRModel.restore_from(MODELS[name], map_location="cuda")
    # as the streaming script: the model's RNNT decoding with an unfused joint
    model.change_decoding_strategy(RNNTDecodingConfig(fused_batch_size=-1))
    if hasattr(model, "set_inference_prompt"):
        model.set_inference_prompt(lang)
        model.decoding.set_strip_lang_tags(False)
    model.eval()
    return model


def to_list(value) -> list[int]:
    if torch.is_tensor(value):
        return [int(v) for v in value.tolist()]
    return [int(v) for v in value]


class LayerRecorder:
    """Every encoder layer's output for the chunk in flight, `[T, d_model]`"""

    def __init__(self, model):
        self.outputs: dict[str, torch.Tensor] = {}
        self.active = False
        self.handles = [
            layer.register_forward_hook(self.hook(index))
            for index, layer in enumerate(model.encoder.layers)
        ]

    def hook(self, index: int):
        def record(_module, _inputs, output):
            if self.active:
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                self.outputs[f"layers.{index}.out"] = tensor[0].detach().float().cpu()

        return record

    def take(self) -> dict[str, torch.Tensor]:
        outputs, self.outputs = self.outputs, {}
        return outputs


def stream_one(model, path: str, recorder: LayerRecorder | None):
    """Streams one file; returns its final hypothesis, per-chunk times and the dump"""
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    buffer = CacheAwareStreamingAudioBuffer(
        model=model, online_normalization=False, pad_and_drop_preencoded=False
    )
    buffer.append_audio_file(path, stream_id=-1)
    dump = {}
    if recorder is not None:
        dump["mel"] = buffer.buffer[0].detach().float().cpu().contiguous()
        dump["mel_len"] = torch.tensor([int(buffer.streams_length[0])], dtype=torch.int64)
        recorder.active = True

    cache_channel, cache_time, cache_len = model.encoder.get_initial_cache_state(batch_size=1)
    hypotheses = None
    times = []
    for step, (chunk, lengths) in enumerate(iter(buffer)):
        drop = 0 if step == 0 else model.encoder.streaming_cfg.drop_extra_pre_encoded
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            (
                _predictions,
                _texts,
                cache_channel,
                cache_time,
                cache_len,
                hypotheses,
            ) = model.conformer_stream_step(
                processed_signal=chunk,
                processed_signal_length=lengths,
                cache_last_channel=cache_channel,
                cache_last_time=cache_time,
                cache_last_channel_len=cache_len,
                keep_all_outputs=buffer.is_buffer_empty(),
                previous_hypotheses=hypotheses,
                drop_extra_pre_encoded=drop,
                return_transcription=True,
            )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        if recorder is not None:
            prefix = f"chunk{step}"
            for key, tensor in recorder.take().items():
                dump[f"{prefix}.{key}"] = tensor.contiguous()
            dump[f"{prefix}.mel"] = chunk[0].detach().float().cpu().contiguous()
            dump[f"{prefix}.mel_len"] = lengths.detach().to(torch.int64).cpu().reshape(1)
            dump[f"{prefix}.drop"] = torch.tensor([drop], dtype=torch.int64)
            dump[f"{prefix}.tokens"] = torch.tensor(to_list(hypotheses[0].y_sequence), dtype=torch.int64)

    if recorder is not None:
        recorder.active = False
        dump["chunks"] = torch.tensor([len(times)], dtype=torch.int64)
    return hypotheses[0], times, dump


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_grad_enabled(False)

    rows = [json.loads(line) for line in open(LS_MANIFEST)]
    if args.limit is not None:
        rows = rows[: args.limit]
    model = load_model(args.model, args.lang)
    left = model.encoder.att_context_size_all[0][0]
    out = args.out / args.model
    out.mkdir(parents=True, exist_ok=True)

    for lookahead in args.lookahead:
        model.encoder.set_default_att_context_size([left, lookahead])
        cfg = model.encoder.streaming_cfg
        print(f"[r{lookahead}] att_context [{left}, {lookahead}] chunk {cfg.chunk_size} shift {cfg.shift_size}")
        recorder = LayerRecorder(model) if args.dump else None
        texts, hyps, dumps = {}, {}, {}
        chunk_times, audio_seconds = [], 0.0
        for index, row in enumerate(rows):
            dumping = index < args.dump
            hypothesis, times, dump = stream_one(model, row["audio_filepath"], recorder if dumping else None)
            texts[row["id"]] = hypothesis.text
            hyps[row["id"]] = {
                "tokens": to_list(hypothesis.y_sequence),
                "timestamps": to_list(hypothesis.timestamp),
            }
            if dumping:
                audio, rate = sf.read(row["audio_filepath"], dtype="float32")
                assert rate == 16000, f"{row['id']} is {rate} Hz"
                dump["audio"] = torch.from_numpy(np.ascontiguousarray(audio))
                dump["tokens"] = torch.tensor(hyps[row["id"]]["tokens"], dtype=torch.int64)
                dump["timestamps"] = torch.tensor(hyps[row["id"]]["timestamps"], dtype=torch.int64)
                dumps.update({f"utt{index}.{key}": value for key, value in dump.items()})
            else:
                # hooks off, so the timed utterances run as the script would
                chunk_times.extend(times)
                audio_seconds += row["duration"]
            if index % 200 == 0:
                print(f"[r{lookahead}] {index}/{len(rows)} {row['id']}: {hypothesis.text}", flush=True)

        prefix = out / f"r{lookahead}"
        Path(f"{prefix}.texts.json").write_text(json.dumps(texts, indent=1, ensure_ascii=False))
        Path(f"{prefix}.hyps.json").write_text(json.dumps(hyps))
        if chunk_times:
            total = float(np.sum(chunk_times))
            timing = {
                "chunks": len(chunk_times),
                "mean_ms": 1e3 * float(np.mean(chunk_times)),
                "p50_ms": 1e3 * float(np.percentile(chunk_times, 50)),
                "p99_ms": 1e3 * float(np.percentile(chunk_times, 99)),
                "audio_seconds": audio_seconds,
                "rtfx": audio_seconds / total,
                "gpu": torch.cuda.get_device_name(),
                "nemo": __import__("nemo").__version__,
                "torch": torch.__version__,
            }
            Path(f"{prefix}.timing.json").write_text(json.dumps(timing, indent=1))
            print(f"[r{lookahead}] {timing}")
        if dumps:
            save_file(dumps, f"{prefix}.dump.safetensors", metadata={"model": args.model})
        if recorder is not None:
            for handle in recorder.handles:
                handle.remove()


if __name__ == "__main__":
    main()
