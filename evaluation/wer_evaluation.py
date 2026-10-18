"""
Word Error Rate evaluation.

Usage:
  python evaluation/wer_evaluation.py evaluation/manifest.csv --conditions baseline noise_10db speed_1.25

For every manifest row and condition it streams the audio through the SAME local pipeline the app uses,
normalizes reference and hypothesis identically, and reports substitutions, deletions, insertions
and WER (jiwer). Results are written to evaluation/results/wer_<timestamp>.json.
Nothing is fabricated: if the manifest or a model is missing the script says so and exits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jiwer

from common import ROOT, apply_condition, load_manifest, load_model, normalize, read_wav_16k_mono, stream_audio
from services.speech_recognition import ModelNotInstalled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--conditions", nargs="+", default=["baseline"])
    args = ap.parse_args()
    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}. See evaluation/manifest_template.csv.")
        return 1
    rows = load_manifest(args.manifest)
    models, results = {}, []
    for row in rows:
        lang = row["language"]
        if lang not in models:
            try:
                models[lang] = load_model(lang)
            except ModelNotInstalled as exc:
                print(f"Model for '{lang}' is not installed: {exc.config.download_hint}")
                return 1
        samples = read_wav_16k_mono(row["audio_path"])
        for condition in args.conditions:
            res = stream_audio(lang, models[lang], apply_condition(samples, condition))
            ref, hyp = normalize(row["reference"]), normalize(res.hypothesis)
            out = jiwer.process_words(ref, hyp)
            results.append({"audio": str(row["audio_path"].name), "language": lang, "condition": condition,
                            "reference": ref, "hypothesis": hyp, "substitutions": out.substitutions,
                            "deletions": out.deletions, "insertions": out.insertions, "hits": out.hits,
                            "wer": out.wer, "ref_words": len(ref.split())})
            print(f"{row['audio_path'].name:30s} {lang} {condition:12s} WER={out.wer:.3f} "
                  f"S={out.substitutions} D={out.deletions} I={out.insertions}")
    summary = {}
    for (lang, cond) in {(r["language"], r["condition"]) for r in results}:
        subset = [r for r in results if r["language"] == lang and r["condition"] == cond]
        n = sum(r["ref_words"] for r in subset)
        errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in subset)
        summary[f"{lang}/{cond}"] = {"files": len(subset), "ref_words": n, "wer": errors / n if n else None,
                                     "S": sum(r["substitutions"] for r in subset),
                                     "D": sum(r["deletions"] for r in subset),
                                     "I": sum(r["insertions"] for r in subset)}
    path = ROOT / "evaluation" / "results" / f"wer_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"kind": "wer", "summary": summary, "per_file": results}, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print("\nCorpus-level WER (total errors / total reference words):")
    for k, v in sorted(summary.items()):
        print(f"  {k:20s} WER={v['wer']:.3f}  S={v['S']} D={v['D']} I={v['I']}  ({v['files']} files)")
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
