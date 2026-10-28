"""Earnings-21 WER of `{stem: text}` transcript files against the .nlp references

Per file and pooled over files (total errors over total reference words), each
side through Whisper's English normalizer, the reference built from the .nlp
token and punctuation fields as the NeMo profile's `long_wer.py` does. Every file
scores only the stems it shares with the first file, so the pooled numbers
compare the same recordings
"""
import argparse
import json
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

REFS = "/shared/datasets/earnings21/nlp_references"


def reference(path):
    lines = Path(path).read_text().splitlines()[1:]
    return " ".join(line.split("|")[0] + line.split("|")[4] for line in lines if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="+")
    ap.add_argument("--refs", default=REFS)
    ap.add_argument("--per-file", action="store_true")
    args = ap.parse_args()

    norm = EnglishTextNormalizer()
    runs = [json.loads(Path(p).read_text()) for p in args.texts]
    stems = sorted(set(runs[0]).intersection(*runs[1:]))
    refs = {stem: norm(reference(Path(args.refs) / f"{stem}.nlp")) for stem in stems}
    for path, run in zip(args.texts, runs):
        out = jiwer.process_words([refs[s] for s in stems], [norm(run[s]) for s in stems])
        print(f"{path}: WER {100 * out.wer:.2f}% over {len(stems)} files")
        if args.per_file:
            for stem in stems:
                print(f"  {stem} {100 * jiwer.wer(refs[stem], norm(run[stem])):.2f}%")


if __name__ == "__main__":
    main()
