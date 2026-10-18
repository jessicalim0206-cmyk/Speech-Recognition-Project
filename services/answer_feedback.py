"""
Transparent, rule-based answer feedback. No machine learning and no external services.

What it checks (and ONLY this):
  * relevance : how many of the question's curated keywords appear in the transcript
  * structure : which STAR parts (Situation, Task, Action, Result) have signal phrases
  * length    : word count against a practical 40-200 word range
It does NOT evaluate pronunciation, confidence, emotion or suitability for a job.
Because it matches phrases, it can miss good answers worded differently. Treat it as a
checklist, not a grade.
"""
from __future__ import annotations

from dataclasses import dataclass

STAR_CUES = {
    "en": {
        "situation": ["when i was", "at my", "during", "last semester", "in my", "internship", "class", "we were", "there was", "once", "project"],
        "task": ["my role", "i was responsible", "i had to", "my task", "goal", "needed to", "assigned", "objective", "responsible for", "challenge"],
        "action": ["i decided", "i created", "i built", "i organized", "i led", "i implemented", "i started", "i talked", "i worked", "i developed", "i analyzed", "i planned", "so i", "i proposed", "i asked", "i learned"],
        "result": ["as a result", "the result", "outcome", "in the end", "finally", "improved", "increased", "reduced", "achieved", "succeeded", "completed", "delivered", "%", "percent", "feedback", "i learned"],
    },
    "id": {
        "situation": ["ketika", "saat", "waktu itu", "di kampus", "semester", "proyek", "magang", "kelompok", "tim", "pada suatu", "situasi", "dulu"],
        "task": ["tugas saya", "tanggung jawab", "peran saya", "saya harus", "saya perlu", "tujuan", "target", "ditugaskan", "tantangan"],
        "action": ["saya memutuskan", "saya membuat", "saya membangun", "saya mengatur", "saya memimpin", "saya menerapkan", "saya mulai", "saya berdiskusi", "saya mengerjakan", "saya mengembangkan", "saya menganalisis", "saya merencanakan", "akhirnya saya", "saya mengusulkan", "saya bertanya", "saya belajar"],
        "result": ["hasilnya", "akibatnya", "pada akhirnya", "akhirnya", "meningkat", "berhasil", "menurun", "selesai", "pelajaran", "belajar", "persen", "%", "umpan balik", "dampak"],
    },
}

TEXT = {
    "en": {
        "star_names": {"situation": "Situation", "task": "Task", "action": "Action", "result": "Result"},
        "rel_strong": "Your answer uses vocabulary closely tied to the question.",
        "rel_partial": "Your answer touches the question but could name its key ideas more directly.",
        "rel_low": "Few of the question's key ideas appear in your answer. Check that you answered what was asked.",
        "short": "Your answer is short ({n} words). Aim for roughly 40 to 200 words, about 30 to 90 seconds.",
        "long": "Your answer is long ({n} words). Consider trimming to the points that matter most.",
        "ok_len": "Length looks comfortable ({n} words).",
        "found": "Found signals for: {parts}.",
        "missing": "No clear signal for: {parts}.",
        "tip": {
            "situation": "Open with one sentence of context: where and when this happened.",
            "task": "State your own responsibility or goal in the situation.",
            "action": "Describe what you personally did, using \"I\" and concrete verbs.",
            "result": "Close with the outcome. A number or a lesson learned makes it memorable.",
        },
        "tip_relevance": "Reuse the question's key words in your first sentence so the answer starts on topic.",
        "tip_length": "Practice again and time yourself against the 30 to 90 second range.",
        "tip_good": "Good foundation. Practice once more and try to say it more smoothly.",
        "caveat": "This is a rule-based phrase check, not an assessment of delivery or ability.",
    },
    "id": {
        "star_names": {"situation": "Situasi", "task": "Tugas", "action": "Tindakan", "result": "Hasil"},
        "rel_strong": "Jawaban Anda memakai kosakata yang dekat dengan pertanyaan.",
        "rel_partial": "Jawaban Anda menyentuh pertanyaan, tetapi bisa menyebut gagasan intinya lebih langsung.",
        "rel_low": "Sedikit gagasan inti pertanyaan muncul di jawaban Anda. Pastikan Anda menjawab yang ditanyakan.",
        "short": "Jawaban Anda singkat ({n} kata). Usahakan sekitar 40 sampai 200 kata, kira-kira 30 sampai 90 detik.",
        "long": "Jawaban Anda panjang ({n} kata). Pertimbangkan memangkasnya ke poin terpenting.",
        "ok_len": "Panjang jawaban sudah nyaman ({n} kata).",
        "found": "Ditemukan tanda untuk: {parts}.",
        "missing": "Belum ada tanda jelas untuk: {parts}.",
        "tip": {
            "situation": "Buka dengan satu kalimat konteks: di mana dan kapan kejadiannya.",
            "task": "Sebutkan tanggung jawab atau tujuan Anda dalam situasi itu.",
            "action": "Jelaskan apa yang Anda lakukan sendiri, dengan kata \"saya\" dan kata kerja konkret.",
            "result": "Tutup dengan hasilnya. Angka atau pelajaran yang didapat membuatnya berkesan.",
        },
        "tip_relevance": "Gunakan kata kunci pertanyaan di kalimat pertama agar jawaban langsung sesuai topik.",
        "tip_length": "Latihan lagi dan catat waktu Anda untuk rentang 30 sampai 90 detik.",
        "tip_good": "Fondasinya sudah baik. Latihan sekali lagi agar penyampaian lebih lancar.",
        "caveat": "Ini pengecekan frasa berbasis aturan, bukan penilaian cara bicara atau kemampuan.",
    },
}


@dataclass
class Feedback:
    word_count: int
    relevance_label: str          # strong | partial | low
    relevance_text: str
    matched_keywords: list[str]
    star_found: list[str]
    star_missing: list[str]
    structure_text: str
    length_text: str
    suggestions: list[str]
    caveat: str


def count_words(text: str) -> int:
    return len(text.split())


def evaluate_answer(transcript: str, keywords: list[str], language: str) -> Feedback:
    t, lower = TEXT[language], transcript.lower()
    words = count_words(transcript)

    matched = [k for k in keywords if k in lower]
    ratio = len(matched) / len(keywords) if keywords else 0
    label = "strong" if ratio >= 0.4 else "partial" if ratio >= 0.2 else "low"

    found = [p for p, cues in STAR_CUES[language].items() if any(c in lower for c in cues)]
    missing = [p for p in STAR_CUES[language] if p not in found]
    names = t["star_names"]
    structure = t["found"].format(parts=", ".join(names[p] for p in found)) if found else ""
    if missing:
        structure = (structure + " " + t["missing"].format(parts=", ".join(names[p] for p in missing))).strip()
    elif not structure:
        structure = ""

    if words < 40:
        length_text, length_ok = t["short"].format(n=words), False
    elif words > 200:
        length_text, length_ok = t["long"].format(n=words), False
    else:
        length_text, length_ok = t["ok_len"].format(n=words), True

    suggestions = [t["tip"][p] for p in missing]
    if label == "low":
        suggestions.append(t["tip_relevance"])
    if not length_ok:
        suggestions.append(t["tip_length"])
    if not suggestions:
        suggestions.append(t["tip_good"])

    return Feedback(words, label, t[f"rel_{label}"], matched, found, missing, structure, length_text,
                    suggestions, t["caveat"])
