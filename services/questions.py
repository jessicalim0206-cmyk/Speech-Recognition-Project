"""Manually curated interview questions. `keywords` drive the transparent relevance check."""

CATEGORIES = {
    "general": {"en": "General interview", "id": "Wawancara umum"},
    "internship": {"en": "Internship", "id": "Magang"},
    "strengths": {"en": "Strengths and weaknesses", "id": "Kelebihan dan kekurangan"},
    "behavioral": {"en": "Behavioral interview", "id": "Wawancara perilaku"},
    "problem": {"en": "Problem-solving", "id": "Pemecahan masalah"},
}

# Each entry: {"en": (question, keywords), "id": (question, keywords)}
QUESTIONS = {
    "general": [
        {"en": ("Tell me about yourself.", ["study", "student", "university", "experience", "skill", "interest", "project", "goal"]),
         "id": ("Ceritakan tentang diri Anda.", ["kuliah", "mahasiswa", "universitas", "pengalaman", "kemampuan", "minat", "proyek", "tujuan"])},
        {"en": ("Why do you want to work at our company?", ["company", "team", "value", "mission", "learn", "contribute", "role"]),
         "id": ("Mengapa Anda ingin bekerja di perusahaan kami?", ["perusahaan", "tim", "nilai", "misi", "belajar", "berkontribusi", "posisi"])},
        {"en": ("Where do you see yourself in five years?", ["years", "grow", "career", "goal", "learn", "lead", "skill"]),
         "id": ("Di mana Anda melihat diri Anda lima tahun lagi?", ["tahun", "berkembang", "karier", "tujuan", "belajar", "memimpin", "kemampuan"])},
    ],
    "internship": [
        {"en": ("Why are you interested in this internship?", ["internship", "learn", "experience", "field", "skill", "career", "opportunity"]),
         "id": ("Mengapa Anda tertarik pada program magang ini?", ["magang", "belajar", "pengalaman", "bidang", "kemampuan", "karier", "kesempatan"])},
        {"en": ("What do you hope to learn from this internship?", ["learn", "skill", "mentor", "experience", "industry", "grow", "practice"]),
         "id": ("Apa yang ingin Anda pelajari dari magang ini?", ["belajar", "kemampuan", "mentor", "pengalaman", "industri", "berkembang", "praktik"])},
        {"en": ("How would you balance your studies with an internship?", ["time", "schedule", "priority", "plan", "deadline", "class", "manage"]),
         "id": ("Bagaimana Anda menyeimbangkan kuliah dan magang?", ["waktu", "jadwal", "prioritas", "rencana", "tenggat", "kuliah", "mengatur"])},
    ],
    "strengths": [
        {"en": ("What is your greatest strength?", ["strength", "good at", "example", "team", "skill", "help", "result"]),
         "id": ("Apa kelebihan terbesar Anda?", ["kelebihan", "kekuatan", "pandai", "contoh", "tim", "kemampuan", "membantu", "hasil"])},
        {"en": ("What is your biggest weakness, and how are you working on it?", ["weakness", "improve", "working on", "learn", "practice", "feedback", "better"]),
         "id": ("Apa kelemahan terbesar Anda dan bagaimana Anda memperbaikinya?", ["kelemahan", "memperbaiki", "belajar", "latihan", "masukan", "lebih baik", "berusaha"])},
    ],
    "behavioral": [
        {"en": ("Tell me about a time you worked in a team and faced a disagreement.", ["team", "group", "disagree", "conflict", "listen", "solution", "agree", "member"]),
         "id": ("Ceritakan saat Anda bekerja dalam tim dan menghadapi perbedaan pendapat.", ["tim", "kelompok", "berbeda pendapat", "konflik", "mendengar", "solusi", "sepakat", "anggota"])},
        {"en": ("Describe a time you missed a deadline or made a mistake.", ["deadline", "mistake", "late", "learn", "fix", "responsib", "next time", "improve"]),
         "id": ("Ceritakan saat Anda terlambat memenuhi tenggat atau melakukan kesalahan.", ["tenggat", "kesalahan", "terlambat", "belajar", "memperbaiki", "tanggung jawab", "lain kali", "memperbaiki"])},
        {"en": ("Tell me about a time you took the lead on something.", ["lead", "leader", "organize", "team", "decision", "responsib", "motivate", "result"]),
         "id": ("Ceritakan saat Anda memimpin sesuatu.", ["memimpin", "pemimpin", "mengatur", "tim", "keputusan", "tanggung jawab", "memotivasi", "hasil"])},
    ],
    "problem": [
        {"en": ("Describe a difficult problem you solved. How did you approach it?", ["problem", "approach", "analy", "solution", "step", "test", "tried", "result"]),
         "id": ("Ceritakan masalah sulit yang pernah Anda selesaikan. Bagaimana pendekatan Anda?", ["masalah", "pendekatan", "menganalisis", "solusi", "langkah", "menguji", "mencoba", "hasil"])},
        {"en": ("What do you do when you are stuck on a task and no one can help?", ["stuck", "research", "search", "documentation", "break", "try", "ask", "learn"]),
         "id": ("Apa yang Anda lakukan saat buntu mengerjakan tugas dan tidak ada yang bisa membantu?", ["buntu", "mencari", "dokumentasi", "memecah", "mencoba", "bertanya", "belajar", "riset"])},
    ],
}


def questions_for(category: str, language: str) -> list[tuple[str, list[str]]]:
    return [item[language] for item in QUESTIONS[category]]


def category_label(category: str, language: str) -> str:
    return CATEGORIES[category][language]
