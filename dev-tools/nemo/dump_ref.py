"""Dump NeMo reference activations for Parakeet TDT 0.6B v2/v3 and Silero VAD v5

Writes <out>/b1.safetensors and <out>/b8.safetensors (plus .json metadata): per-stage
encoder activations (fp32, TF32 off), a step-by-step greedy TDT decode trace per
utterance, and a bf16-autocast encoder output for tolerance scale. With --vad it also
writes <out>/vad.safetensors from the Silero v5 TorchScript model.

Ported from ~/code/open-asr-kernels/parakeet-profile/scripts/dump_ref.py, which wrote
the v3 set in /shared/datasets/parakeet-ref; the key layout is unchanged
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file

LS_MANIFEST = "/shared/datasets/librispeech-test-clean/manifest.jsonl"
NEMO_PATHS = {
    "v2": "/shared/models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo",
    "v3": "/shared/models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo",
}


def load_model(version, device="cuda"):
    import logging

    import nemo.collections.asr as nemo_asr
    from nemo.utils import logging as nemo_logging

    nemo_logging.setLevel(logging.ERROR)
    model = nemo_asr.models.ASRModel.restore_from(NEMO_PATHS[version], map_location=device)
    model.eval()
    # dither adds random noise to the mel input; off so runs and dumps are deterministic
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    return model


def load_manifest(n=None):
    rows = [json.loads(line) for line in open(LS_MANIFEST)]
    return rows[:n] if n else rows

# layers whose every submodule input/output is captured; all other layers only record their output
FULL_LAYERS = (0, 23)


def pick_sets(rows):
    # b1: first utterance between 8 and 12 s; b8: 8 utterances with spread lengths so padding paths run
    b1 = [next(r for r in rows if 8.0 <= r["duration"] <= 12.0)]
    by_len = sorted(rows, key=lambda r: (r["duration"], r["id"]))
    n = len(by_len)
    b8 = [by_len[int(n * q)] for q in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.99)]
    return {"b1": b1, "b8": b8}


def first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for v in x:
            t = first_tensor(v)
            if t is not None:
                return t
    return None


class Recorder:
    def __init__(self, model):
        self.store = {}
        self.handles = []
        enc = model.encoder
        targets = {"encoder.pre_encode": enc.pre_encode, "encoder.pos_enc": enc.pos_enc}
        for name, mod in enc.pre_encode.named_modules():
            if name:
                targets[f"encoder.pre_encode.{name}"] = mod
        for i, layer in enumerate(enc.layers):
            targets[f"encoder.layers.{i}"] = layer
            if i in FULL_LAYERS:
                for name, mod in layer.named_modules():
                    if name:
                        targets[f"encoder.layers.{i}.{name}"] = mod
        for name, mod in targets.items():
            self.handles.append(mod.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(mod, inputs, output):
            i = first_tensor(inputs)
            o = first_tensor(output)
            if i is not None and f"{name}.in" not in self.store:
                self.store[f"{name}.in"] = i.detach().float().cpu().contiguous()
            if o is not None and f"{name}.out" not in self.store:
                self.store[f"{name}.out"] = o.detach().float().cpu().contiguous()

        return fn

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def trace_tdt(model, enc_bt, length):
    """Reference greedy TDT decode for one utterance, mirroring NeMo GreedyTDTInfer._greedy_decode"""
    dec, joint = model.decoder, model.joint
    durations = list(model.cfg.decoding.durations)
    max_symbols = model.cfg.decoding.greedy.max_symbols
    blank = dec.blank_idx
    enc_proj = joint.project_encoder(enc_bt[:length].unsqueeze(0))  # (1, T, 640)
    state, last = None, None
    g = None
    rec = {k: [] for k in ("t", "label_in", "token", "dur", "pred_out", "logits", "h", "c")}
    tokens, stamps, durs = [], [], []
    t = 0
    while t < length:
        symbols = 0
        need_loop = True
        while need_loop and symbols < max_symbols:
            label = torch.tensor([[blank if last is None else last]], device=enc_bt.device)
            # SOS is the blank index; with blank_as_pad its embedding is zero
            g_new, state_new = dec.predict(None if last is None else label, state, add_sos=False, batch_size=1)
            pred_proj = joint.project_prednet(g_new)  # (1, 1, 640)
            logits = joint.joint_after_projection(enc_proj[:, t : t + 1], pred_proj)[0, 0, 0]
            tok = int(logits[: -len(durations)].argmax())
            dk = int(logits[-len(durations) :].argmax())
            skip = durations[dk]
            rec["t"].append(t)
            rec["label_in"].append(blank if last is None else last)
            rec["token"].append(tok)
            rec["dur"].append(skip)
            rec["pred_out"].append(g_new[0, 0].float().cpu())
            rec["logits"].append(logits.float().cpu())
            if tok != blank:
                tokens.append(tok)
                stamps.append(t)
                durs.append(skip)
                state, last = state_new, tok
            symbols += 1
            t += skip
            need_loop = skip == 0
        if symbols == max_symbols:
            t += 1
    out = {
        "trace.t": torch.tensor(rec["t"], dtype=torch.int32),
        "trace.label_in": torch.tensor(rec["label_in"], dtype=torch.int32),
        "trace.token": torch.tensor(rec["token"], dtype=torch.int32),
        "trace.dur": torch.tensor(rec["dur"], dtype=torch.int32),
        "trace.pred_out": torch.stack(rec["pred_out"]),
        "trace.logits": torch.stack(rec["logits"]),
        "trace.enc_proj": enc_proj[0].float().cpu().contiguous(),
        "hyp.tokens": torch.tensor(tokens, dtype=torch.int32),
        "hyp.timestamps": torch.tensor(stamps, dtype=torch.int32),
        "hyp.durations": torch.tensor(durs, dtype=torch.int32),
    }
    return out


@torch.no_grad()
def run_set(model, rows, device):
    audios = [sf.read(r["audio_filepath"], dtype="float32")[0] for r in rows]
    lens = torch.tensor([len(a) for a in audios], dtype=torch.int64)
    sig = torch.zeros(len(audios), int(lens.max()))
    for i, a in enumerate(audios):
        sig[i, : len(a)] = torch.from_numpy(a)
    sig, lens_d = sig.to(device), lens.to(device)

    rec = Recorder(model)
    mel, mel_len = model.preprocessor(input_signal=sig, length=lens_d)
    enc, enc_len = model.encoder(audio_signal=mel, length=mel_len)
    rec.close()

    t = {"audio": sig.cpu(), "audio_len": lens, "mel": mel.float().cpu(), "mel_len": mel_len.cpu(),
         "encoder.out": enc.float().cpu(), "encoder.out_len": enc_len.cpu()}
    t.update(rec.store)

    # NeMo's own batched label-looping decode (the path that gets profiled)
    hyps = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=True)
    hyps = hyps[0] if isinstance(hyps, tuple) else hyps
    meta = {"utterances": [], "nemo_decoding": str(type(model.decoding.decoding).__name__)}
    for b, (r, h) in enumerate(zip(rows, hyps)):
        tr = trace_tdt(model, enc[b].transpose(0, 1), int(enc_len[b]))
        nemo_tokens = [int(x) for x in h.y_sequence]
        ref_tokens = tr["hyp.tokens"].tolist()
        for k, v in tr.items():
            t[f"utt{b}.{k}"] = v
        t[f"utt{b}.nemo_batched.tokens"] = torch.tensor(nemo_tokens, dtype=torch.int32)
        meta["utterances"].append({
            "index": b, "id": r["id"], "duration_s": r["duration"], "ref_text": r["text"],
            "nemo_text": h.text, "trace_text": model.tokenizer.ids_to_text(ref_tokens),
            "trace_matches_nemo_batched": ref_tokens == nemo_tokens,
            "trace_steps": len(tr["trace.t"]), "tokens": len(ref_tokens),
        })

    # bf16 autocast encoder output, to show the scale of deviation a bf16/fp16 kernel should expect
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mel16, mel16_len = model.preprocessor(input_signal=sig, length=lens_d)
        enc16, _ = model.encoder(audio_signal=mel16, length=mel16_len)
    t["encoder.out.bf16_autocast"] = enc16.float().cpu()
    diff = (enc16.float() - enc).abs()
    meta["bf16_vs_fp32_encoder_out"] = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                                        "ref_rms": float(enc.pow(2).mean().sqrt())}
    return t, meta


@torch.no_grad()
def dump_vad(out, audio_path):
    """Silero v5 probabilities for every 512-sample window of one file, plus window 0's stages

    Mirrors scriptrs `process_windows`: zero state per recording, 64 context samples from
    the previous chunk, and a short final chunk padded with its last sample
    """
    import importlib.resources

    jit = torch.jit.load(
        str(importlib.resources.files("silero_vad").joinpath("data/silero_vad.jit")), map_location="cpu"
    )
    model = jit._model
    audio = torch.from_numpy(sf.read(audio_path, dtype="float32")[0])
    window, context_len = 512, 64
    state = torch.zeros(2, 1, 128)
    context = torch.zeros(context_len)
    probs, inputs = [], []
    for start in range(0, len(audio), window):
        chunk = audio[start : start + window]
        if len(chunk) < window:
            chunk = torch.cat([chunk, chunk[-1:].expand(window - len(chunk))])
        x = torch.cat([context, chunk]).unsqueeze(0)
        if start == 0:
            stages = {"window0.in": x.clone()}
            spec = model.stft(x)
            stages["window0.stft"] = spec
            enc = model.encoder(spec)
            stages["window0.encoder"] = enc
        out_prob, state = model(x, state)
        probs.append(float(out_prob))
        inputs.append(x[0])
        context = chunk[-context_len:]
    tensors = {
        "audio": audio,
        "probs": torch.tensor(probs, dtype=torch.float32),
        "windows": torch.stack(inputs),
        **{k: v.float().contiguous() for k, v in stages.items()},
    }
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "vad.safetensors"))
    meta = {"audio": audio_path, "windows": len(probs), "keys": {k: list(v.shape) for k, v in tensors.items()}}
    (out / "vad.json").write_text(json.dumps(meta, indent=1))
    print("vad", meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=sorted(NEMO_PATHS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vad-audio", help="also dump Silero v5 references for this 16 kHz wav")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.vad_audio:
        dump_vad(out, args.vad_audio)
    model = load_model(args.version, device)

    rows = load_manifest()
    for name, sel in pick_sets(rows).items():
        tensors, meta = run_set(model, sel, device)
        tensors = {k: v.contiguous() for k, v in tensors.items()}
        save_file(tensors, str(out / f"{name}.safetensors"))
        meta["keys"] = {k: [list(v.shape), str(v.dtype)] for k, v in tensors.items()}
        (out / f"{name}.json").write_text(json.dumps(meta, indent=1))
        print(name, len(tensors), "tensors", json.dumps({k: v for k, v in meta.items() if k != "keys"})[:2000])


if __name__ == "__main__":
    main()
