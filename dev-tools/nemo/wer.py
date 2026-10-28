"""LibriSpeech test-clean WER of a `{id: text}` transcript file, as the NeMo profile scores it

Whisper's English normalizer on both sides, utterances with an empty normalized
reference dropped, corpus WER with jiwer. With `--against`, also counts utterances
whose raw text differs from another transcript file, such as NeMo's
"""
import argparse
import json

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

MANIFEST = "/shared/datasets/librispeech-test-clean/manifest.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts")
    ap.add_argument("--against", help="another {id: text} file to diff utterance by utterance")
    ap.add_argument("--manifest", default=MANIFEST)
    args = ap.parse_args()

    refs = {row["id"]: row["text"] for row in map(json.loads, open(args.manifest))}
    hyps = json.load(open(args.texts))
    missing = sorted(set(refs) - set(hyps))
    if missing:
        raise SystemExit(f"{len(missing)} utterances missing, first {missing[0]}")

    norm = EnglishTextNormalizer()
    pairs = [(norm(refs[i]), norm(hyps[i])) for i in sorted(refs)]
    pairs = [(r, h) for r, h in pairs if r.strip()]
    print(f"{args.texts}: WER {100 * jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]):.3f}% over {len(pairs)} utterances")

    if args.against:
        other = json.load(open(args.against))
        differ = [i for i in sorted(refs) if hyps[i] != other.get(i)]
        print(f"{len(differ)} of {len(refs)} transcripts differ from {args.against}")
        for i in differ[:20]:
            print(f"  {i}\n    ours:  {hyps[i]}\n    other: {other.get(i)}")


if __name__ == "__main__":
    main()
