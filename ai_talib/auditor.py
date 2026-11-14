"""طبقة التدقيق والتمحيص: تحميل النماذج، وأربع بوابات (تكرار، اندفاع زمني، نص عشوائي، موازنة الفئات)."""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import IMPULSE_SEC, K, MODELS_DIR, MODEL_IDS, SHEET_COLS, ar_norm, grade_key, school_key, student_key

_REL = [("عالي", 3), ("متوسط", 2), ("منخفض", 1), ("ضعيف", 1)]
_LAT = re.compile(r"اندفاع\s*(\d+)\s*\|\s*متزن\s*(\d+)\s*\|\s*بطيء\s*(\d+)")


def rel_rank(s) -> int:
    s = ar_norm(s)
    for k, v in _REL:
        if ar_norm(k) in s:
            return v
    return 2


def parse_dist(s):
    """'اسم: 40% | اسم2: 20% ...' -> [0.4, 0.2, ...] (أو None)."""
    p = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", str(s))]
    if len(p) != K or sum(p) <= 0:
        return None
    t = sum(p)
    return [x / t for x in p]


def is_gibberish(t) -> bool:
    t = str(t or "").strip()
    if not t or t.lower() == "nan":
        return False
    letters = re.findall(r"[\u0600-\u06FFa-zA-Z]", t)
    if len(letters) < 2 or re.search(r"(.)\1{3,}", t):
        return True
    flat = t.replace(" ", "")
    if len(flat) >= 5 and len(set(flat)) <= 3:
        return True
    if len(t.split()) == 1 and len(t) >= 12 and len(set(t)) / len(t) < 0.35:
        return True
    if re.fullmatch(r"[a-zA-Z]{5,}", t) and not re.search(r"[aeiouAEIOU]", t):
        return True
    return False


def load_models_long() -> pd.DataFrame:
    parts = []
    for mid in MODEL_IDS:
        f = MODELS_DIR / f"{mid}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, dtype=object, encoding="utf-8-sig")
        if df.empty:
            continue
        df = df.iloc[:, :len(SHEET_COLS)]
        df.columns = SHEET_COLS[:df.shape[1]]
        for c in SHEET_COLS[df.shape[1]:]:
            df[c] = np.nan
        df["model"] = mid
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=SHEET_COLS + ["model", "skey"])
    d = pd.concat(parts, ignore_index=True)
    d = d[d["name"].astype(str).str.strip().replace("nan", "") != ""].copy()
    for c in ("total_sec", "avg_sec", "type_idx", "faking"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["ts"] = pd.to_datetime(d["ts"], errors="coerce")
    d["skey"] = [student_key(n, g, s) for n, g, s in zip(d["name"], d["grade"], d["school"])]
    return d.reset_index(drop=True)


@dataclass
class AuditResult:
    clean: pd.DataFrame = None        # الصفوف بعد إزالة التكرار (مع أعلام)
    rejected: pd.DataFrame = None     # المكررات المستبعدة
    class_weights: dict = field(default_factory=dict)   # model -> [w1..w4]
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        s = self.stats
        return (f"إجمالي الصفوف {s['total']} | مكررة مستبعدة {s['duplicates']} | اندفاع زمني {s['impulsive']} | "
                f"نص عشوائي {s['gibberish']} | صالحة للتدريب {s['trainable']} | طلاب فريدون {s['students']}")


def audit(df: pd.DataFrame) -> AuditResult:
    if df.empty:
        return AuditResult(clean=df.assign(trainable=False, impulsive=False, gibberish=False, trust=0.0, dist_p=None),
                           rejected=df.copy(), stats=dict(total=0, duplicates=0, impulsive=0, gibberish=0,
                                                          trainable=0, students=0))
    d = df.copy()
    d["dist_p"] = d["dist"].map(parse_dist)
    lat = d["latency"].astype(str).str.extract(_LAT).astype(float)
    imp_ratio = lat[0] / lat.sum(axis=1).replace(0, np.nan)
    # بوابة 2: متوسط زمن السؤال < 1.5 ث، أو غالبية الإجابات موسومة «اندفاع»
    d["impulsive"] = ((d["avg_sec"] > 0) & (d["avg_sec"] < IMPULSE_SEC)) | (imp_ratio >= 0.6)
    d["gibberish"] = d["problems"].map(is_gibberish)             # بوابة 3
    d["rel_rank"] = d["reliability"].map(rel_rank)
    # بوابة 1: تكرار (طالب × نموذج) -> نبقي الأعلى موثوقية، غير المندفع، الأحدث
    d = d.sort_values(["rel_rank", "impulsive", "ts"], ascending=[False, True, False], na_position="last")
    dup = d.duplicated(["skey", "model"], keep="first")
    rejected = d[dup].assign(reason="duplicate")
    c = d[~dup].copy()
    c["trainable"] = (~c["impulsive"]) & c["dist_p"].notna()
    fk = (c["faking"].fillna(0) / 100).clip(0, 1)
    c["trust"] = (1 - 0.5 * c["impulsive"] - 0.2 * c["gibberish"] - 0.3 * fk).clip(0, 1)
    # بوابة 4: موازنة الفئات لكل نموذج: n / (k * count) مقصوصة بين 0.5 و5
    cw = {}
    for mid, g in c[c["trainable"]].groupby("model"):
        counts = np.array([(g["type_idx"] == i + 1).sum() for i in range(K)], float)
        w = np.where(counts > 0, len(g) / (K * np.maximum(counts, 1)), 1.0)
        cw[mid] = np.clip(w, 0.5, 5).tolist()
    stats = dict(total=len(df), duplicates=int(dup.sum()), impulsive=int(c["impulsive"].sum()),
                 gibberish=int(c["gibberish"].sum()), trainable=int(c["trainable"].sum()),
                 students=int(c["skey"].nunique()))
    return AuditResult(clean=c.sort_values(["skey", "model"]).reset_index(drop=True),
                       rejected=rejected.reset_index(drop=True), class_weights=cw, stats=stats)
