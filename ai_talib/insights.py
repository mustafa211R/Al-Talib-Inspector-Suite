"""الاستقراء والاستدلال: جدول موحّد للطلاب، شيوع المشكلات والأسباب، ارتباطات، تجميع أنماط، مقارنة المدارس، وقرارات مقترحة.
كل الأرقام وصفية/استدلالية على عيّنتك فقط؛ العلاقات ارتباطية لا سببية، والعيّنات الصغيرة تُنبَّه صراحةً."""
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import CATALOG, RISK_WORDS, THEMES, TYPE_VALUE, ar_norm
from .research_lab import wilson

ROOT_RULES = [
    ([("D", 1), ("O", 1)], "ملل مع نظرة إلى المعلم كملقِّن", "الكسل ناتج غالباً عن الملل مع نظرة إلى المعلم كملقِّن، لا عن ضعف في الفهم."),
    ([("D", 1), ("V", 0)], "رفض التلقين مع ملل", "رفضه للتلقين مع ملله يوحي بأن عقله يطلب تحدّياً لا تكراراً."),
    ([("S", 2), ("D", 3)], "تشتت الشاشات والانتباه", "تشتت الشاشات يغذّي كسل التشتت؛ المشكلة بيئة انتباه أكثر منها قدرة."),
    ([("X", 2), ("O", 2)], "خوف من العقاب وسلطة المعلم", "خوف من العقاب مع نظرة إلى المعلم كسلطة محاسِبة قد يدفع إلى الإخفاء والتردد في السؤال."),
    ([("Z", 2), ("N", 2)], "ضغط البيت وكتمان الغضب", "ضغط البيت مع كتمان الغضب قد يظهر لاحقاً كانطواء أو انفجار."),
    ([("Z", 3), ("D", 2)], "ضعف الدعم الأسري مع إحباط دراسي", "ضعف الدعم الأسري مع إحباط دراسي؛ يحتاج سنداً بالغاً في المدرسة."),
    ([("A", 3), ("W", 2)], "تجنّب التفكير وهشاشة الثقة", "تجنّب التفكير مع الخوف من الصعب يشير إلى هشاشة ثقة أكثر منها قصوراً معرفياً."),
]
_LAT = re.compile(r"اندفاع\s*(\d+)\s*\|\s*متزن\s*(\d+)\s*\|\s*بطيء\s*(\d+)")


def items(s):
    return [x.strip() for x in str(s).split("|") if x.strip() and x.strip().lower() != "nan"]


def analyze_text(text):
    norm = ar_norm(text)
    themes = sorted({t for kws, t in THEMES for k in kws if k in norm})
    risk = [w for w in RISK_WORDS if w in norm]
    return themes, risk


def lat_counts(s):
    m = _LAT.search(str(s))
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


# ------------------------------------------------------------------ الجدول الموحّد
def student_table(store) -> pd.DataFrame:
    c, pr = store.clean, store.profiles
    prof = pr.set_index("skey") if len(pr) else pd.DataFrame()
    recs = []
    for sk, me in store.students().set_index("skey").iterrows():
        g = c[c["skey"] == sk] if len(c) else c
        att, stg, causes, cdesc = [], [], [], {}
        if len(g):
            have = set()
            for r in g.itertuples():
                if not r.dist_p or pd.isna(r.type_idx):
                    continue
                ti = int(r.type_idx) - 1
                have.add((r.model, ti))
                label = f"{CATALOG[r.model]['title']}: {r.type_name}"
                v = TYPE_VALUE[r.model][ti]
                (att if v == -1 else stg if v == 1 else []).append(label)
            for cond, title, desc in ROOT_RULES:
                if all(x in have for x in cond):
                    causes.append(title); cdesc[title] = desc
        elif sk in prof.index:
            att, stg = items(prof.loc[sk, "attention"]), items(prof.loc[sk, "strengths"])
        if sk in prof.index:
            for it in items(prof.loc[sk, "causes"]):
                title = it.split(":")[0].strip()
                causes.append(title); cdesc.setdefault(title, it.split(":", 1)[-1].strip())
        probs = " ".join(dict.fromkeys(str(x).strip() for x, gb in zip(g["problems"], g["gibberish"])
                                       if str(x).strip() not in ("", "nan") and not gb)) if len(g) else ""
        themes, risk = analyze_text(probs)
        lat = np.sum([lat_counts(x) for x in g["latency"]], axis=0) if len(g) else np.zeros(3)
        recs.append(dict(skey=sk, name=me["name"], grade=me["grade"], school=me["school"],
                         n_models=len(g) if len(g) else me["n_models"], problems=probs, wrote=bool(probs),
                         themes=themes, safety=risk, attention=list(dict.fromkeys(att)), strengths=list(dict.fromkeys(stg)),
                         causes=list(dict.fromkeys(causes)), cause_desc=cdesc,
                         faking=float(g["faking"].mean()) if len(g) else np.nan,
                         trust=float(g["trust"].mean()) if len(g) else np.nan,
                         impulsive=float(lat[0] / lat.sum()) if lat.sum() else np.nan))
    return pd.DataFrame(recs, columns=["skey", "name", "grade", "school", "n_models", "problems", "wrote", "themes", "safety",
                                       "attention", "strengths", "causes", "cause_desc", "faking", "trust", "impulsive"])


# ------------------------------------------------------------------ الشيوع
def prevalence(t: pd.DataFrame, col: str, top=None) -> pd.DataFrame:
    n = len(t)
    rows = {}
    for sk, lst in zip(t["skey"], t[col]):
        for x in lst:
            rows.setdefault(x, set()).add(sk)
    out = []
    for k, s in rows.items():
        lo, hi = wilson(len(s), n)
        out.append(dict(item=k, count=len(s), pct=100 * len(s) / n if n else 0, lo=100 * lo, hi=100 * hi, skeys=sorted(s)))
    d = pd.DataFrame(out, columns=["item", "count", "pct", "lo", "hi", "skeys"])
    d = d.sort_values(["count", "item"], ascending=[False, True]).reset_index(drop=True)
    return d.head(top) if top else d


def root_causes(t: pd.DataFrame) -> pd.DataFrame:
    d = prevalence(t, "causes")
    desc = {}
    for dd in t["cause_desc"]:
        for k, v in dd.items():
            desc.setdefault(k, v)
    d["description"] = d["item"].map(desc)
    return d


# ------------------------------------------------------------------ ارتباطات
def _bh(p):
    p = np.asarray(p, float); n = len(p)
    if not n:
        return p
    o = np.argsort(p); q = np.empty(n)
    run = 1.0
    for rank, i in zip(range(n, 0, -1), o[::-1]):
        run = min(run, p[i] * n / rank); q[i] = run
    return q


def feature_matrix(t: pd.DataFrame):
    cols = {}
    for kind, col in (("نقطة انتباه", "attention"), ("مشكلة", "themes"), ("سبب جذري", "causes")):
        for sk, lst in zip(t["skey"], t[col]):
            for x in lst:
                cols.setdefault(f"{kind}: {x}", set()).add(sk)
    X = pd.DataFrame({k: t["skey"].isin(v) for k, v in cols.items()}) if cols else pd.DataFrame(index=range(len(t)))
    X.index = t["skey"].values
    return X.astype(int)


def associations(t: pd.DataFrame, min_support=3, top=60) -> pd.DataFrame:
    X = feature_matrix(t)
    n = len(X)
    X = X.loc[:, X.sum() >= min_support]
    if X.shape[1] > top:
        X = X[X.sum().sort_values(ascending=False).index[:top]]
    out = []
    cols = list(X.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            n11 = int((X[a] & X[b]).sum())
            if n11 < min_support:
                continue
            na, nb = int(X[a].sum()), int(X[b].sum())
            tab = [[n11, na - n11], [nb - n11, n - na - nb + n11]]
            _, p = stats.fisher_exact(tab, alternative="greater")
            lift = n11 * n / (na * nb)
            out.append(dict(a=a, b=b, n_both=n11, n_a=na, n_b=nb, conf_a=100 * n11 / na, conf_b=100 * n11 / nb, lift=lift, p=p))
    d = pd.DataFrame(out, columns=["a", "b", "n_both", "n_a", "n_b", "conf_a", "conf_b", "lift", "p"])
    if len(d):
        d["q"] = _bh(d["p"])
        d = d[d["lift"] > 1].sort_values(["p", "lift"], ascending=[True, False])
    return d.reset_index(drop=True)


# ------------------------------------------------------------------ تجميع الأنماط
def clusters(t: pd.DataFrame, k_max=6):
    X = feature_matrix(t)
    X = X.loc[:, X.sum() >= 2]
    n = len(X)
    if n < 12 or X.shape[1] < 3:
        return None
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best = None
    for k in range(2, min(k_max, n // 4) + 1):
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(X.values)
        if len(set(lab)) < 2:
            continue
        sc = silhouette_score(X.values, lab)
        if best is None or sc > best[0]:
            best = (sc, k, lab)
    if best is None:
        return None
    sc, k, lab = best
    base = X.mean()
    groups = []
    for g in range(k):
        m = lab == g
        prev = X[m].mean()
        lift = (prev / base.replace(0, np.nan)).fillna(0)
        feats = [(f, 100 * prev[f], lift[f]) for f in prev.index if prev[f] >= .4 and lift[f] >= 1.3]
        feats.sort(key=lambda x: (-x[2], -x[1]))
        att = [f for f in feats if f[0].startswith("نقطة انتباه")][:2]
        title = "، ".join(f[0].split(": ", 1)[1] for f in att) or (feats[0][0].split(": ", 1)[1] if feats else "مجموعة عامة")
        groups.append(dict(cluster=g + 1, size=int(m.sum()), title=title, features=feats[:6],
                           skeys=list(X.index[m]), names=list(t.set_index("skey").loc[X.index[m], "name"])))
    return dict(k=k, silhouette=float(sc), groups=sorted(groups, key=lambda x: -x["size"]))


# ------------------------------------------------------------------ مقارنة المدارس والصفوف
def group_comparison(t: pd.DataFrame, top_items=6, min_group=5) -> pd.DataFrame:
    prev = prevalence(t, "attention", top_items)
    out = []
    for gcol, gname in (("school", "المدرسة"), ("grade", "الصف")):
        for gv, sub in t.groupby(gcol):
            if len(sub) < min_group or len(sub) == len(t):
                continue
            rest = t[~t["skey"].isin(sub["skey"])]
            for it in prev["item"]:
                a = int(sub["attention"].map(lambda l: it in l).sum()); b = int(rest["attention"].map(lambda l: it in l).sum())
                _, p = stats.fisher_exact([[a, len(sub) - a], [b, len(rest) - b]])
                out.append(dict(kind=gname, group=gv, n=len(sub), item=it, rate=100 * a / len(sub),
                                rest=100 * b / max(len(rest), 1), p=p))
    d = pd.DataFrame(out, columns=["kind", "group", "n", "item", "rate", "rest", "p"])
    return d.sort_values("p").reset_index(drop=True) if len(d) else d


# ------------------------------------------------------------------ القرارات
def conf_label(n):
    return ("عالية" if n >= 100 else "متوسطة" if n >= 40 else "منخفضة (عيّنة صغيرة)")


def plan_for(label):
    for mid, m in CATALOG.items():
        for i, tn in enumerate(m["types"]):
            if f"{m['title']}: {tn}" == label:
                return m["plans"][i]
    return None


@dataclass
class Insight:
    n: int
    table: pd.DataFrame
    att: pd.DataFrame
    problems: pd.DataFrame
    causes: pd.DataFrame
    assoc: pd.DataFrame
    clusters: dict
    groups: pd.DataFrame
    decisions: list
    quality: dict


def build(store) -> Insight:
    t = student_table(store)
    n = len(t)
    att = prevalence(t, "attention"); prob = prevalence(t, "themes"); rc = root_causes(t)
    assoc = associations(t) if n >= 8 else pd.DataFrame()
    cl = clusters(t) if n >= 12 else None
    gc = group_comparison(t) if n >= 10 else pd.DataFrame()
    q = dict(wrote=int(t["wrote"].sum()), safety=int((t["safety"].map(len) > 0).sum()),
             low_trust=int((t["trust"] < 0.6).sum()), high_faking=int((t["faking"] >= 40).sum()),
             impulsive=int((t["impulsive"] >= 0.3).sum()), with_rows=int(t["trust"].notna().sum()))
    dec = []
    cf = conf_label(n)
    if q["safety"]:
        dec.append(dict(score=1000, kind="أمان", title=f"{q['safety']} طالباً ذكروا كلمات تتعلق بالأمان/الأذى",
                        evidence="رصد كلمات خطر في نصوصهم الحرة", action="مقابلة فردية خلال 24 ساعة من المرشد، قبل أي إجراء جماعي.", conf="—"))
    for r in att.head(4).itertuples():
        pl = plan_for(r.item)
        dec.append(dict(score=r.pct * 1.0, kind="نقطة انتباه", title=f"{r.item}",
                        evidence=f"{r.count} من {n} طالباً ({r.pct:.0f}%؛ فترة ثقة 95%: {r.lo:.0f}–{r.hi:.0f}%)",
                        action=pl or "خصّص تدخلاً جماعياً قصيراً ثم أعد القياس بعد فصل دراسي.", conf=cf))
    for r in rc.head(3).itertuples():
        dec.append(dict(score=r.pct * 1.2, kind="سبب جذري", title=r.item,
                        evidence=f"{r.count} من {n} ({r.pct:.0f}%؛ {r.lo:.0f}–{r.hi:.0f}%)",
                        action="عالج السبب لا العَرَض: " + (r.description or "راجع تفاصيل الحالات المرتبطة بهذا السبب."), conf=cf))
    for r in prob.head(2).itertuples():
        dec.append(dict(score=r.pct * 0.9, kind="مشكلة مكتوبة", title=f"موضوع شائع: {r.item}",
                        evidence=f"{r.count} طالباً ({r.pct:.0f}%) كتبوه بأنفسهم", action="ضعه في أجندة اجتماع المعلمين والإرشاد.", conf=cf))
    if len(assoc):
        for r in assoc[assoc["p"] < .05].head(2).itertuples():
            dec.append(dict(score=r.conf_a * .6, kind="ارتباط", title=f"{r.a}  ↔  {r.b}",
                            evidence=f"يتزامنان عند {r.n_both} طالباً؛ lift={r.lift:.1f}؛ p={r.p:.3f}؛ q={r.q:.2f}",
                            action="افحص الثاني تلقائياً عند ظهور الأول. العلاقة ارتباطية لا سببية.", conf=cf))
    if len(gc):
        for r in gc[(gc["p"] < .1) & (gc["rate"] > gc["rest"])].head(2).itertuples():
            dec.append(dict(score=r.rate * .5, kind=f"مقارنة {r.kind}", title=f"{r.group}: {r.item}",
                            evidence=f"{r.rate:.0f}% مقابل {r.rest:.0f}% في البقية (n={r.n}؛ p={r.p:.3f})",
                            action="ابحث عن عامل مشترك (معلم، جدول، مجموعة مؤثّرة) قبل التعميم.", conf=cf))
    if q["with_rows"] and q["low_trust"] / max(q["with_rows"], 1) > .25:
        dec.append(dict(score=30, kind="جودة البيانات", title="نسبة موثوقية منخفضة مرتفعة",
                        evidence=f"{q['low_trust']} من {q['with_rows']} طالباً بموثوقية < 60%", action="أعد إجراء القياس بإشراف هادئ قبل بناء قرارات.", conf="—"))
    dec.sort(key=lambda x: -x["score"])
    return Insight(n, t, att, prob, rc, assoc, cl, gc, dec, q)


def decisions_html(I: Insight) -> str:
    import html as H
    e = lambda x: H.escape(str(x))
    out = ['<html dir="rtl"><body style="font-family:Tahoma">', "<h2>تقرير الاستقراء والاستدلال</h2>",
           f"<p>العيّنة: <b>{I.n}</b> طالباً · الثقة العامة: <b>{conf_label(I.n)}</b></p>"]
    if I.n < 30:
        out.append("<p style='color:#a45b00'><b>تنبيه:</b> العيّنة أقل من 30 طالباً؛ النسب أولية وفترات الثقة عريضة. لا تتخذ قراراً مصيرياً بناءً عليها وحدها.</p>")
    out.append("<h3>القرارات المقترحة (مرتبة بالأولوية)</h3><ol>")
    for d in I.decisions:
        out.append(f"<li><b>[{e(d['kind'])}] {e(d['title'])}</b><br>الدليل: {e(d['evidence'])}<br>الإجراء: {e(d['action'])}<br>"
                   f"<small>درجة الثقة: {e(d['conf'])}</small></li>")
    out.append("</ol>" if I.decisions else "<p>لا قرارات بعد: البيانات غير كافية.</p>")
    if I.clusters:
        out.append(f"<h3>أنماط الطلاب المتجمعة</h3><p>{I.clusters['k']} مجموعات (silhouette = {I.clusters['silhouette']:.2f}"
                   + (" — تجمعات ضعيفة، استخدمها للاستكشاف فقط" if I.clusters['silhouette'] < .25 else "") + ")</p><ul>")
        for g in I.clusters["groups"]:
            out.append(f"<li><b>{e(g['title'])}</b> — {g['size']} طالباً</li>")
        out.append("</ul>")
    out.append("<hr><p><small>الأرقام وصفية على هذه العيّنة فقط. الارتباط لا يعني السببية، والتقرير للإرشاد لا للعقاب ولا يغني عن المختص.</small></p></body></html>")
    return "".join(out)
