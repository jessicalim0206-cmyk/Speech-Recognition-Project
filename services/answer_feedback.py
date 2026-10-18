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



def detect_question_type(question: str, language: str) -> str:
    """Classify common interview questions using simple phrase matching."""
    q = question.lower().strip()

    if language == "en":
        if any(x in q for x in [
            "tell me about yourself", "introduce yourself"
        ]):
            return "introduction"
        if any(x in q for x in [
            "why do you want", "why are you interested",
            "what do you hope to learn", "where do you see yourself"
        ]):
            return "motivation"
        if any(x in q for x in [
            "greatest strength", "biggest weakness",
            "your strengths", "your weaknesses"
        ]):
            return "strengths"
        if any(x in q for x in [
            "tell me about a time", "describe a time",
            "difficult problem", "when you are stuck",
            "how did you approach"
        ]):
            return "behavioral"
        if "balance your studies" in q:
            return "planning"

    else:
        if any(x in q for x in [
            "ceritakan tentang diri", "perkenalkan diri"
        ]):
            return "introduction"
        if any(x in q for x in [
            "mengapa", "apa yang ingin anda pelajari",
            "di mana anda melihat diri"
        ]):
            return "motivation"
        if any(x in q for x in [
            "kelebihan terbesar", "kelemahan terbesar"
        ]):
            return "strengths"
        if any(x in q for x in [
            "ceritakan saat", "masalah sulit",
            "bagaimana pendekatan"
        ]):
            return "behavioral"
        if "menyeimbangkan kuliah" in q:
            return "planning"

    return "general"


QUESTION_TIPS = {
    "en": {
        "introduction": (
            "Briefly introduce your background, relevant skills or experience, "
            "and what you hope to contribute."
        ),
        "motivation": (
            "Connect your motivation to this role, explain what you hope to learn, "
            "and describe how you could contribute."
        ),
        "strengths": (
            "Name one specific strength and support it with a concrete example "
            "of how you used it."
        ),
        "behavioral": (
            "Use STAR: explain the situation, your responsibility, the actions "
            "you took, and the result or lesson learned."
        ),
        "planning": (
            "Explain how you would plan your schedule, prioritize tasks, "
            "and handle deadlines."
        ),
        "general": (
            "Answer the specific question directly, then support your main point "
            "with a relevant detail or example."
        ),
    },
    "id": {
        "introduction": (
            "Perkenalkan latar belakang, kemampuan atau pengalaman yang relevan, "
            "serta kontribusi yang ingin Anda berikan."
        ),
        "motivation": (
            "Hubungkan motivasi Anda dengan posisi ini, jelaskan hal yang ingin "
            "dipelajari, dan kontribusi yang bisa diberikan."
        ),
        "strengths": (
            "Sebutkan satu kelebihan yang spesifik dan berikan contoh nyata "
            "saat Anda menggunakannya."
        ),
        "behavioral": (
            "Gunakan STAR: jelaskan situasi, tanggung jawab, tindakan yang Anda "
            "lakukan, serta hasil atau pelajaran yang diperoleh."
        ),
        "planning": (
            "Jelaskan cara mengatur jadwal, menentukan prioritas, "
            "dan menyelesaikan tugas sesuai tenggat."
        ),
        "general": (
            "Jawab pertanyaan secara langsung, lalu dukung poin utama "
            "dengan detail atau contoh yang relevan."
        ),
    },
}



def get_content_suggestions(
    transcript: str,
    language: str,
    question_type: str,
) -> list[str]:
    """Give transparent, phrase-based suggestions based on answer content."""
    text = transcript.lower()
    suggestions = []

    if language == "en":
        if question_type == "introduction":
            has_background = any(
                phrase in text
                for phrase in [
                    "my name is",
                    "i study",
                    "i am studying",
                    "i major in",
                    "i majored in",
                    "i am majoring",
                    "my major",
                    "computer science",
                    "i graduated",
                ]
            )
            has_skill = any(
                phrase in text
                for phrase in [
                    "skill", "programming", "coding", "python",
                    "javascript", "communication", "teamwork",
                ]
            )
            has_experience = any(
                phrase in text
                for phrase in [
                    "project", "internship", "experience",
                    "i developed", "i created", "i built",
                    "i worked on",
                ]
            )
            has_goal = any(
                phrase in text
                for phrase in [
                    "i hope", "i want to", "my goal",
                    "i aim to", "in the future", "i would like",
                ]
            )

            if not has_background:
                suggestions.append(
                    "Add one brief detail about your education or background."
                )
            if not has_skill and not has_experience:
                suggestions.append(
                    "Mention one relevant skill or project to make your introduction more specific."
                )
            elif not has_experience:
                suggestions.append(
                    "Consider adding a brief example of a project or experience where you used your skills."
                )
            if not has_goal:
                suggestions.append(
                    "Consider ending with what you hope to learn or contribute."
                )

            if "i am majored" in text:
                suggestions.append(
                    'Grammar tip: say "I am majoring in Computer Science" or "I major in Computer Science," not "I am majored in Computer Science."'
                )

            if (
                "that's why" in text and "that is the reason" in text
            ) or (
                "that is why" in text and "the reason" in text
            ):
                suggestions.append(
                    "Avoid repeating the same reason. Choose one clear expression to explain your motivation."
                )

        elif question_type == "strengths":
            if not any(
                phrase in text
                for phrase in ["because", "for example", "such as", "when i"]
            ):
                suggestions.append(
                    "Explain why this is a strength and give a specific example."
                )

        elif question_type == "motivation":
            if not any(
                phrase in text
                for phrase in ["because", "so that", "learn", "contribute", "goal"]
            ):
                suggestions.append(
                    "Explain why this opportunity interests you and what you hope to gain or contribute."
                )

        elif question_type == "planning":
            if not any(
                phrase in text
                for phrase in ["prioritize", "schedule", "plan", "deadline", "calendar"]
            ):
                suggestions.append(
                    "Give a concrete example of how you would plan your time and prioritize deadlines."
                )

    else:
        if question_type == "introduction":
            has_background = any(
                phrase in text
                for phrase in [
                    "saya kuliah", "saya sedang kuliah", "jurusan saya",
                    "saya mengambil jurusan", "saya lulusan",
                ]
            )
            has_skill = any(
                phrase in text
                for phrase in [
                    "kemampuan", "pemrograman", "coding",
                    "komunikasi", "kerja sama",
                ]
            )
            has_experience = any(
                phrase in text
                for phrase in [
                    "proyek", "magang", "pengalaman",
                    "saya membuat", "saya mengembangkan",
                ]
            )
            has_goal = any(
                phrase in text
                for phrase in [
                    "saya ingin", "tujuan saya", "berharap",
                    "ke depannya", "saya berharap",
                ]
            )

            if not has_background:
                suggestions.append(
                    "Tambahkan satu informasi singkat tentang pendidikan atau latar belakang Anda."
                )
            if not has_skill and not has_experience:
                suggestions.append(
                    "Sebutkan satu kemampuan atau proyek yang relevan agar perkenalan lebih spesifik."
                )
            elif not has_experience:
                suggestions.append(
                    "Tambahkan contoh singkat proyek atau pengalaman yang menunjukkan kemampuan Anda."
                )
            if not has_goal:
                suggestions.append(
                    "Akhiri dengan hal yang ingin Anda pelajari atau kontribusikan."
                )

        elif question_type == "strengths":
            if not any(
                phrase in text
                for phrase in ["karena", "contohnya", "misalnya", "ketika saya"]
            ):
                suggestions.append(
                    "Jelaskan alasan kelebihan tersebut penting dan berikan contoh nyata."
                )

        elif question_type == "motivation":
            if not any(
                phrase in text
                for phrase in ["karena", "belajar", "berkontribusi", "tujuan"]
            ):
                suggestions.append(
                    "Jelaskan alasan Anda tertarik serta hal yang ingin dipelajari atau diberikan."
                )

        elif question_type == "planning":
            if not any(
                phrase in text
                for phrase in ["prioritas", "jadwal", "merencanakan", "tenggat", "deadline"]
            ):
                suggestions.append(
                    "Berikan contoh cara mengatur waktu dan menentukan prioritas tugas."
                )

    return suggestions


def evaluate_answer(
    transcript: str,
    keywords: list[str],
    language: str,
    question: str = "",
) -> Feedback:
    t = TEXT[language]
    lower = transcript.lower()
    words = count_words(transcript)
    question_type = detect_question_type(question, language)

    if question_type == "introduction":
        matched = []

        if any(x in lower for x in [
            "my name is", "i am", "i'm"
        ]):
            matched.append("personal introduction")

        if any(x in lower for x in [
            "computer science", "my major", "i major in",
            "i majored in", "i am majoring in"
        ]):
            matched.append("education or major")

        if any(x in lower for x in [
            "coding", "programming", "i enjoy", "i love"
        ]):
            matched.append("interests")

        if any(x in lower for x in [
            "project", "internship", "experience",
            "i developed", "i built", "i created"
        ]):
            matched.append("experience")

        if any(x in lower for x in [
            "i hope", "my goal", "in the future",
            "i want to contribute"
        ]):
            matched.append("goals")

        core_parts = {"personal introduction", "education or major"}
        ratio = len(set(matched) & core_parts) / len(core_parts)
        label = "strong" if ratio == 1 else "partial" if ratio > 0 else "low"
    else:
        matched = [
            keyword for keyword in keywords
            if keyword.lower() in lower
        ]
        ratio = len(matched) / len(keywords) if keywords else 0
        label = (
            "strong" if ratio >= 0.4
            else "partial" if ratio >= 0.2
            else "low"
        )

    label = (
        "strong" if ratio >= 0.4
        else "partial" if ratio >= 0.2
        else "low"
    )

    # Use STAR only for experience-based questions.
    if question_type == "behavioral":
        found = [
            part for part, cues in STAR_CUES[language].items()
            if any(cue in lower for cue in cues)
        ]
        missing = [
            part for part in STAR_CUES[language]
            if part not in found
        ]

        names = t["star_names"]
        structure_parts = []

        if found:
            structure_parts.append(
                t["found"].format(
                    parts=", ".join(names[p] for p in found)
                )
            )
        if missing:
            structure_parts.append(
                t["missing"].format(
                    parts=", ".join(names[p] for p in missing)
                )
            )

        structure = " ".join(structure_parts)
        suggestions = [t["tip"][part] for part in missing]
    else:
        found = []
        missing = []
        structure = QUESTION_TIPS[language][question_type]
        suggestions = []

    # Add question-specific content feedback.
    suggestions.extend(
        get_content_suggestions(transcript, language, question_type)
    )

    # Keep word count as guidance, not the only measure of quality.
    if words < 40:
        length_text = t["short"].format(n=words)
        if language == "en":
            suggestions.append(
                "Your answer is brief. Add useful details rather than repeating the same point just to increase the word count."
            )
        else:
            suggestions.append(
                "Jawaban masih singkat. Tambahkan detail yang berguna, bukan mengulang poin yang sama hanya untuk menambah jumlah kata."
            )
    elif words > 200:
        length_text = t["long"].format(n=words)
        suggestions.append(t["tip_length"])
    else:
        length_text = t["ok_len"].format(n=words)

    if label == "low":
        suggestions.append(t["tip_relevance"])

    if not suggestions:
        suggestions.append(t["tip_good"])

    suggestions = list(dict.fromkeys(suggestions))

    return Feedback(
        words,
        label,
        t[f"rel_{label}"],
        matched,
        found,
        missing,
        structure,
        length_text,
        suggestions,
        t["caveat"],
    )
