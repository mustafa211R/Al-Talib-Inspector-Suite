"""يولّد مصنف XLSX تجريبياً بنفس بنية Google Sheet (Model_A_sheet ... + Student_Profiles) لتجربة الأداة بلا بيانات حقيقية.
تحذير: بيانات عشوائية مصطنعة للاختبار التقني فقط، لا تصلح لأي استنتاج تربوي."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai_talib.config import CATALOG, MODEL_IDS, SHEET_COLS  # noqa: E402

HEAD = ["التاريخ", "الاسم", "الصف", "المدرسة", "الزمن الكلي (ث)", "متوسط زمن السؤال (ث)", "رقم النمط", "النمط الغالب",
        "التوزيع %", "تفاصيل الإجابات", "المشاكل (كتبها الطالب)", "النتائج والتحليل", "الموثوقية", "تنبيهات", "تحليل الزمن",
        "سريع/متأنٍّ", "مؤشر تجميل الصورة", "التوقع"]


def make(path, n_students=40, seed=3):
    rng = np.random.default_rng(seed)
    names = [f"طالب تجريبي {i + 1}" for i in range(n_students)]
    trait = rng.normal(size=n_students)                      # سمة كامنة تربط النماذج ببعضها
    sheets = {}
    for mid in MODEL_IDS:
        types = CATALOG[mid]["types"]
        rows = []
        for si, nm in enumerate(names):
            if rng.random() > (0.9 if mid in "ABCD" else 0.45):
                continue
            logits = rng.normal(size=4) + np.array([trait[si], 0, -trait[si] * .5, -trait[si]])
            pct = np.exp(logits) / np.exp(logits).sum()
            counts = rng.multinomial(7, pct)
            dom = int(counts.argmax())
            dist = " | ".join(f"{t}: {round(100 * c / 7)}%" for t, c in zip(types, counts))
            imp = rng.random() < .08
            avg = round(float(rng.uniform(.6, 1.3) if imp else rng.uniform(4, 16)), 1)
            det = "\n".join(f"س{q + 1} [base/{'اندفاع' if imp else 'متزن'}]: خيار ← {types[t]} ({avg}ث)"
                            for q, t in enumerate(rng.choice(4, 7, p=pct)))
            prob = rng.choice(["لا يوجد", "أعاني من الامتحانات", "ssssss", "المعلم يشرح بسرعة", ""])
            rows.append([f"2026-11-{rng.integers(1, 14):02d} 10:00", nm, "الثاني متوسط", "ذي النفس الزكية", round(avg * 9),
                         avg, dom + 1, types[dom], dist, det, prob, "", rng.choice(["عالية", "متوسطة"]), "",
                         f"اندفاع {7 if imp else 0} | متزن {0 if imp else 7} | بطيء 0 | متزن", "—",
                         int(rng.integers(0, 70)), ""])
        sheets[f"Model_{mid}_sheet"] = pd.DataFrame(rows, columns=HEAD)
    prof = pd.DataFrame([[f"{n}|الثاني متوسط|ذي النفس الزكية", n, "الثاني متوسط", "ذي النفس الزكية", 5, "ABCDE",
                          "", "", "", "11/13/2026"] for n in names[:5]],
                        columns=["المفتاح", "الاسم", "الصف", "المدرسة", "عدد النماذج", "النماذج المنجزة", "نقاط القوة",
                                 "تحتاج انتباه", "الأسباب الجذرية المحتملة", "آخر تحديث"])
    with pd.ExcelWriter(path) as w:
        for k, v in sheets.items():
            v.to_excel(w, sheet_name=k, index=False)
        prof.to_excel(w, sheet_name="Student_Profiles", index=False)
    return path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "demo_data.xlsx"
    print("written", make(out))
