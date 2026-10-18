"""
Transcription latency evaluation (offline, reproducible).

Usage:
  python evaluation/latency_evaluation.py evaluation/manifest.csv --realtime

Method (documented so the report can cite it exactly):
  1. Each WAV is converted to 16 kHz mono PCM16 and fed to the recognizer in 100 ms chunks,
     the same call path as the live app (services.speech_recognition back-ends).
  2. per-chunk inference time = wall time of backend.feed(chunk), via time.perf_counter().
  3. time-to-first-partial   = audio time elapsed when the first non-empty partial appears.
  4. final delay             = wall time of backend.flush() after the last chunk.
  5. real-time factor (RTF)  = total inference time / audio duration (< 1 means faster than real time).
With --realtime the chunks are paced at 1x speed, like a live microphone.
Measured on YOUR machine; numbers depend on CPU and model. Browser capture and network
transport are not included; the live app's own measurement (shown after each session) uses
the definition in services/speech_recognition.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from common import ROOT, apply_condition, load_manifest, load_model, read_wav_16k_mono, stream_audio
from services.speech_recognition import ModelNotInstalled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--conditions", nargs="+", default=["baseline"])
    ap.add_argument("--realtime", action="store_true", help="pace chunks at 1x speed")
    args = ap.parse_args()
    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}. See evaluation/manifest_template.csv.")
        return 1
    models, results = {}, []
    for row in load_manifest(args.manifest):
        lang = row["language"]
        if lang not in models:
            try:
                models[lang] = load_model(lang)
            except ModelNotInstalled as exc:
                print(f"Model for '{lang}' is not installed: {exc.config.download_hint}")
                return 1
        samples = read_wav_16k_mono(row["audio_path"])
        for condition in args.conditions:
            res = stream_audio(lang, models[lang], apply_condition(samples, condition), realtime=args.realtime)
            total = sum(res.chunk_times_s) + res.final_delay_s
            results.append({"audio": row["audio_path"].name, "language": lang, "condition": condition,
                            "audio_s": res.audio_s, "first_partial_s": res.first_partial_s,
                            "median_chunk_ms": float(np.median(res.chunk_times_s) * 1000),
                            "p95_chunk_ms": float(np.percentile(res.chunk_times_s, 95) * 1000),
                            "final_delay_ms": res.final_delay_s * 1000, "rtf": total / res.audio_s})
            r = results[-1]
            fp = f"{r['first_partial_s']:.2f}s" if r["first_partial_s"] is not None else "none"
            print(f"{r['audio']:28s} {lang} {condition:12s} first_partial={fp:7s} "
                  f"median_chunk={r['median_chunk_ms']:.1f}ms p95={r['p95_chunk_ms']:.1f}ms "
                  f"final={r['final_delay_ms']:.0f}ms RTF={r['rtf']:.2f}")
    path = ROOT / "evaluation" / "results" / f"latency_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"kind": "latency", "realtime_pacing": args.realtime, "per_file": results}, indent=2),
                    encoding="utf-8")
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
