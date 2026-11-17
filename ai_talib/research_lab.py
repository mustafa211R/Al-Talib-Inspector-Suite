"""مختبر البحث الأكاديمي: إحصاءات وصفية/استدلالية، تحليل الأسئلة، ارتباط عبر النماذج، تعمية الهوية، منشئ النماذج."""
import hashlib
import json
import re
from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats

from .config import CATALOG, DB_DIR, K, MODEL_IDS, TYPE_VALUE
from .neural_trainer import value_score

_LINE = re.compile(r"^س(\d+)\s*\[([^\]]*)\]:\s*(.*?)(?:\s*←\s*(.*?))?\s*\((\d+(?:\.\d+)?)ث\)\s*$")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, c - h), min(1, c + h)


def type_distribution(clean, mid):
    g = clean[(clean["model"] == mid) & clean["trainable"]]
    n = len(g)
    rows = []
    for i, name in enumerate(CATALOG[mid]["types"]):
        k = int((g["type_idx"] == i + 1).sum())
        lo, hi = wilson(k, n)
        rows.append(dict(type=name, count=k, pct=100 * k / n if n else 0, lo=100 * lo, hi=100 * hi))
    return pd.DataFrame(rows), n


def time_summary(clean, mids):
    out = []
    for m in mids:
        a = clean[clean["model"] == m]["avg_sec"].dropna()
        out.append(dict(model=m, n=len(a), mean=a.mean() if len(a) else np.nan, sd=a.std(ddof=1) if len(a) > 1 else np.nan,
                        q1=a.quantile(.25) if len(a) else np.nan, med=a.median() if len(a) else np.nan,
                        q3=a.quantile(.75) if len(a) else np.nan))
    return pd.DataFrame(out)


def score_matrix(clean):
    """طلاب × نماذج: درجة قيمة النمط ∈ [-1,1] (من توزيع النسب × جدول القيم)."""
    g = clean[clean["trainable"]]
    s = {(r.skey, r.model): value_score(r.dist_p, r.model) for r in g.itertuples()}
    if not s:
        return pd.DataFrame()
    return pd.Series(s).unstack().reindex(columns=[m for m in MODEL_IDS if m in {k[1] for k in s}])


def correlations(clean, mids, method="spearman", min_n=5):
    M = score_matrix(clean)
    cols = [m for m in mids if m in M.columns]
    r = pd.DataFrame(np.nan, index=cols, columns=cols)
    p = r.copy()
    n = r.copy()
    for a in cols:
        for b in cols:
            if a == b:
                r.loc[a, b], p.loc[a, b], n.loc[a, b] = 1.0, 0.0, M[a].notna().sum()
                continue
            ok = M[[a, b]].dropna()
            n.loc[a, b] = len(ok)
            if len(ok) >= min_n and ok[a].std() > 0 and ok[b].std() > 0:
                f = stats.spearmanr if method == "spearman" else stats.pearsonr
                rr, pp = f(ok[a], ok[b])
                r.loc[a, b], p.loc[a, b] = rr, pp
    return r, p, n


def item_analysis(clean, mid):
    """مستوى السؤال: نسبة اختيار كل بديل، متوسط الزمن، ارتباط السؤال بالمجموع (مصحَّح)، ومعامل ألفا."""
    g = clean[(clean["model"] == mid) & clean["trainable"]]
    types = CATALOG[mid]["types"]
    val = TYPE_VALUE[mid]
    rec = []
    for r in g.itertuples():
        for line in str(r.details).splitlines():
            m = _LINE.match(line.strip())
            if m and m.group(1).isdigit() and m.group(2).split("/")[0] == "base":
                ti = types.index(m.group(4).strip()) if m.group(4) and m.group(4).strip() in types else -1
                rec.append((r.skey, int(m.group(1)), ti, float(m.group(5))))
    if not rec:
        return pd.DataFrame(), np.nan, 0
    d = pd.DataFrame(rec, columns=["skey", "q", "t", "sec"])
    d["score"] = d["t"].map(lambda t: val[t] if t >= 0 else np.nan)
    sc = d.pivot_table(index="skey", columns="q", values="score", aggfunc="first").dropna()
    rows = []
    for q, gq in d.groupby("q"):
        row = dict(q=q, question=CATALOG[mid]["questions"][q - 1]["q"] if q - 1 < len(CATALOG[mid]["questions"]) else "",
                   n=len(gq), mean_sec=gq["sec"].mean())
        for i, t in enumerate(types):
            row[t] = 100 * (gq["t"] == i).mean()
        if q in sc.columns and len(sc) >= 5:
            rest = sc.drop(columns=q).sum(axis=1)
            row["item_total_r"] = sc[q].corr(rest) if sc[q].std() > 0 and rest.std() > 0 else np.nan
        rows.append(row)
    alpha = np.nan
    if sc.shape[1] > 1 and len(sc) >= 5:
        k = sc.shape[1]
        tv = sc.sum(axis=1).var(ddof=1)
        alpha = k / (k - 1) * (1 - sc.var(ddof=1).sum() / tv) if tv > 0 else np.nan
    return pd.DataFrame(rows), alpha, len(sc)


def anonymize(clean: pd.DataFrame) -> pd.DataFrame:
    """يستبدل الأسماء والمدارس برموز ID_001.. ثابتة عبر الجلسات (ملح محلي في db/)."""
    salt_f = DB_DIR / ".salt"
    if not salt_f.exists():
        salt_f.write_text(hashlib.sha256(np.random.default_rng().bytes(16)).hexdigest())
    salt = salt_f.read_text()
    h = lambda x: hashlib.sha256((salt + str(x)).encode()).hexdigest()
    keys = sorted(clean["skey"].unique(), key=h)
    sid = {k: f"ID_{i + 1:03d}" for i, k in enumerate(keys)}
    sch = {k: f"SCH_{i + 1:02d}" for i, k in enumerate(sorted(clean["school"].astype(str).unique(), key=h))}
    d = clean.drop(columns=["name", "details", "problems", "result", "skey"], errors="ignore").copy()
    d.insert(0, "student_id", clean["skey"].map(sid).values)
    d["school"] = clean["school"].astype(str).map(sch).values
    return d.drop(columns=["dist_p"], errors="ignore")


def read_result(a, b, r, p, n, method="spearman"):
    """يصوغ جملة «قراءة النتائج» بلغة أكاديمية. إرشادية: لا تُغني عن مراجعة الباحث."""
    if np.isnan(r):
        return f"لا تكفي البيانات (n={int(n)}) لتقدير العلاقة بين {a} و{b}."
    mag = abs(r)
    strength = "ضعيفة" if mag < .3 else "متوسطة" if mag < .5 else "قوية" if mag < .7 else "قوية جداً"
    direction = "طردية" if r > 0 else "عكسية"
    sig = ("ذات دلالة إحصائية عند مستوى 0.01" if p < .01 else "ذات دلالة إحصائية عند مستوى 0.05" if p < .05
           else "غير دالة إحصائياً")
    sym = "ρ" if method == "spearman" else "r"
    return (f"أظهرت نتائج تحليل الارتباط ({'سبيرمان' if method == 'spearman' else 'بيرسون'}) علاقة {direction} {strength} "
            f"{sig} بين «{CATALOG[a]['title']}» و«{CATALOG[b]['title']}» ({sym} = {r:.2f}, p = {p:.3f}, n = {int(n)}). "
            f"تنبيه: العينة قد تكون صغيرة ولا تسمح بالتعميم، والارتباط لا يعني السببية.")


def apa_table(df: pd.DataFrame) -> str:
    return df.round(2).to_html(index=False, border=0, classes="apa", na_rep="—")


def build_model_js(model_id, title, icon, types, questions):
    """يولّد تعريف نموذج جديد بصيغة M(...) المستخدمة في questions-a.js. types: [(اسم، وصف، توصية)×4]،
    questions: [(نص، [4 بدائل بترتيب الأنماط])]."""
    if len(types) != 4 or any(len(q[1]) != 4 for q in questions):
        raise ValueError("يجب أن يكون لكل نموذج 4 أنماط ولكل سؤال 4 بدائل بترتيب الأنماط.")
    return f'M({json.dumps(model_id)}, {json.dumps(title, ensure_ascii=False)}, {json.dumps(icon, ensure_ascii=False)}, ' \
           f'{json.dumps([list(t) for t in types], ensure_ascii=False, indent=2)}, ' \
           f'{json.dumps([[q, list(o)] for q, o in questions], ensure_ascii=False, indent=2)});\n'
