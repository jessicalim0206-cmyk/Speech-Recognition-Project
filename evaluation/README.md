# Evaluation workflow (reproducible)

1. Install models (see ../models/README.md).
2. Record clips: `python evaluation/record_test_clip.py evaluation/clips/en_01.wav --seconds 15`
   Read a fixed script aloud. Use at least several clips per language (e.g. 5 EN + 5 ID, different speakers if possible).
3. Copy `manifest_template.csv` to `manifest.csv`, one row per clip, with the EXACT reference text you read.
4. WER, baseline + two experimental conditions:
   `python evaluation/wer_evaluation.py evaluation/manifest.csv --conditions baseline noise_10db speed_1.25`
5. Latency: `python evaluation/latency_evaluation.py evaluation/manifest.csv --realtime`
6. Results are saved in `evaluation/results/*.json` and appear on the Developer tests page.

Conditions: `noise_<dB>db` adds seeded white noise at that SNR; `speed_<factor>` resamples (pitch changes too, a limitation).
Reference and hypothesis are lowercased and stripped of punctuation before scoring (jiwer). Report this normalization.
For the error analysis, read the per-file `reference` vs `hypothesis` in the JSON and categorize S/D/I errors.
