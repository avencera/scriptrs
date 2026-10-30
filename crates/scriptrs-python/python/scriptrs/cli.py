"""The ``scriptrs`` command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import Model, Pipeline, ScriptrsError, TranscriptionResult, __version__


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the exit status."""
    parser = argparse.ArgumentParser(
        prog="scriptrs", description="Speech-to-text with NVIDIA Parakeet TDT."
    )
    parser.add_argument("--version", action="version", version=f"scriptrs {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe = commands.add_parser(
        "transcribe",
        help="transcribe audio files",
        description="Transcribe audio files and print each transcript to stdout. With several files, each transcript follows a ==> FILE <== header, as head does.",
    )
    transcribe.add_argument("audio", type=Path, nargs="+", help="audio files (WAV, FLAC, MP3, Ogg, M4A)")
    transcribe.add_argument(
        "--model",
        default=Model.PARAKEET_V2.value,
        help="parakeet/v2 (English, the default) or parakeet/v3 (25 European languages)",
    )
    transcribe.add_argument("--language", help="ISO 639-1 code such as en or de (default: detect)")
    transcribe.add_argument(
        "--vad",
        action="store_true",
        help="split long recordings at detected speech; helps when most of the audio is not speech",
    )
    transcribe.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "tf32", "fp32"],
        help="CUDA matrix multiply precision (default: fp16)",
    )
    transcribe.add_argument("--model-dir", type=Path, help="load the model from this directory instead of downloading it")
    transcribe.add_argument("--cache-dir", type=Path, help="model download cache (default: the Hugging Face hub cache)")
    transcribe.add_argument(
        "--json",
        action="store_true",
        help="print one JSON object per file, with timed chunks and tokens",
    )
    args = parser.parse_args(argv)

    try:
        if args.model_dir is not None:
            pipeline = Pipeline.from_dir(
                args.model_dir, args.model, vad=args.vad, precision=args.precision
            )
        else:
            pipeline = Pipeline.from_pretrained(
                args.model, vad=args.vad, precision=args.precision, cache_dir=args.cache_dir
            )
        # a single transcript stays bare so it pipes cleanly; several get headers
        # the way head and tail label several files
        headers = not args.json and len(args.audio) > 1
        for index, path in enumerate(args.audio):
            result = pipeline.transcribe_file(path, language=args.language)
            if args.json:
                print(json.dumps(_as_json(path, result), ensure_ascii=False))
                continue

            if headers:
                if index:
                    print()
                print(f"==> {path} <==")
            print(result.text, flush=True)
    except (ScriptrsError, ValueError) as error:
        print(f"error: {getattr(error, 'message', error)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130

    return 0


def _as_json(path: Path, result: TranscriptionResult) -> dict[str, object]:
    return {
        "file": str(path),
        "text": result.text,
        "duration_seconds": result.duration_seconds,
        "chunks": [
            {"start": chunk.start, "end": chunk.end, "text": chunk.text} for chunk in result.chunks
        ],
        "tokens": [
            {
                "text": token.text,
                "start": token.start,
                "end": token.end,
                "confidence": token.confidence,
            }
            for token in result.tokens
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
