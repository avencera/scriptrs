"""Time NeMo's encoder alone on the reference mel batches, eager and as a CUDA graph

The native CUDA encoder is timed on the same inputs by `just gpu-parity`
(`gpu_encoder_graph_replays_the_eager_pass`), so the two compare stage for stage:
mel in, encoder output out, no featurizer or decode. Run it through the gpu0 queue
"""
import argparse
import time

import torch
from safetensors.torch import load_file

from dump_ref import load_model

REPLAYS = 100


def timed(fn, iters):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v3")
    ap.add_argument("--refs", default="/shared/datasets/parakeet-ref")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.version)

    for name in ("b1", "b8"):
        ref = load_file(f"{args.refs}/{name}.safetensors")
        mel, mel_len = ref["mel"].cuda(), ref["mel_len"].cuda()
        audio_s = float(ref["audio_len"].sum()) / 16000
        for precision in ("fp32", "bf16-autocast"):
            autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision != "fp32")

            def run():
                with autocast:
                    return model.encoder(audio_signal=mel, length=mel_len)

            for _ in range(3):
                run()
            eager_ms = timed(run, 20)
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph):
                run()
            replay_ms = timed(graph.replay, REPLAYS)
            print(f"nemo {args.version} {name}: batch {mel.shape[0]}, {audio_s:.1f} s audio, {precision}: "
                  f"eager {eager_ms:.3f} ms, graph replay {replay_ms:.3f} ms")


if __name__ == "__main__":
    main()
