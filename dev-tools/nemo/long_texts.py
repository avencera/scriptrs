"""NeMo transcripts of whole long recordings, the profile's long-form recipe

Each wav runs as one batch-1 input through Parakeet with local attention
[context, context], FP32, dither off. Writes `{stem: text}` for `long_wer.py`.
The subsampling convolution splits only when its input would pass PyTorch's
indexing limit, as in the profile. A recording that runs out of GPU memory is
listed under `"_skipped"` instead of transcribed
"""
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict

V3 = "/home/praveen/.cache/huggingface/hub/models--nvidia--parakeet-tdt-0.6b-v3/snapshots/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.nemo"


def transcribe(model, audio):
    signal = torch.from_numpy(audio).cuda()[None]
    length = torch.tensor([len(audio)], device="cuda")
    mel, mel_len = model.preprocessor(input_signal=signal, length=length)
    enc, enc_len = model.encoder(audio_signal=mel, length=mel_len)
    hyps = model.decoding.rnnt_decoder_predictions_tensor(
        encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=False
    )
    hyps = hyps[0] if isinstance(hyps, tuple) else hyps
    return hyps[0].text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=V3)
    ap.add_argument("--local-context", type=int, default=256)
    args = ap.parse_args()

    model = nemo_asr.models.ASRModel.restore_from(args.model, map_location="cuda").eval()
    model.preprocessor.featurizer.dither = 0.0
    model.change_attention_model("rel_pos_local_attn", [args.local_context, args.local_context])
    model.change_subsampling_conv_chunking_factor(1)
    # NeMo's CUDA-graph greedy decoder hits an illegal memory access on the second
    # hour-long input; the eager label-looping decoder gives the same hypotheses
    cfg = model.cfg.decoding
    with open_dict(cfg):
        cfg.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(cfg)
    # the swaps rebuild modules in train mode
    model.eval()

    texts = {}
    skipped = []
    with torch.inference_mode():
        for path in sorted(args.wavs):
            audio, sr = sf.read(path, dtype="float32")
            assert sr == 16000, path
            try:
                texts[Path(path).stem] = transcribe(model, audio)
                print(Path(path).stem, len(audio) / sr, "s", flush=True)
            except torch.OutOfMemoryError:
                skipped.append(Path(path).stem)
                print(Path(path).stem, len(audio) / sr, "s: out of GPU memory", flush=True)
            torch.cuda.empty_cache()
    if skipped:
        texts["_skipped"] = skipped
    Path(args.out).write_text(json.dumps(texts, indent=1))


if __name__ == "__main__":
    main()
