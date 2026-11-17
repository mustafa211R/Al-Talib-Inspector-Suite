"""المعالج والمشخّص: بحث الطالب، تقرير من ستة أقسام، توائم سلوكية. التفسير هنا قائم على قواعد، لا نموذج لغوي."""
import html
import re

import numpy as np
import pandas as pd

from .config import CATALOG, K, MODEL_IDS, RISK_WORDS, THEMES, TYPE_VALUE, ar_norm
from .neural_trainer import value_score
from .insights import lat_counts
from .research_lab import _LINE

DISCLAIMER = ("هذا التقرير مؤشرات إرشادية وتنبؤية لدعم الطالب وتطوير تعلّمه، وليس حكماً قضائياً ولا تشخيصاً طبياً، "
              "ولا يغني عن المختص البشري. يُستخدم للإرشاد لا للعقاب.")
_DENY = re.compile(r"^(لا يوجد|لايوجد|لا شي|لا شيء|لا اعاني|ما اعاني|ما عندي|لا مشاكل|لا مشكله|لا عندي|ماكو|لا شيئ)")

# قواعد الجذر: مجموعات (نموذج، فهرس نمط 0-based) يجب أن يكون الطالب غالباً عليها
ROOT_RULES = [
    ([("D", 1), ("O", 1)], "الكسل ناتج غالباً عن الملل مع نظرة إلى المعلم كملقِّن، لا عن ضعف في الفهم."),
    ([("D", 1), ("V", 0)], "رفضه للتلقين مع ملله يوحي بأن عقله يطلب تحدّياً لا تكراراً."),
    ([("S", 2), ("D", 3)], "تشتت الشاشات يغذّي كسل التشتت؛ المشكلة بيئة انتباه أكثر منها قدرة."),
    ([("X", 2), ("O", 2)], "خوف من العقاب مع نظرة إلى المعلم كسلطة محاسِبة قد يدفع إلى الإخفاء والتردد في السؤال."),
    ([("Z", 2), ("N", 2)], "ضغط البيت مع كتمان الغضب قد يظهر لاحقاً كانطواء أو انفجار."),
    ([("Z", 3), ("D", 2)], "ضعف الدعم الأسري مع إحباط دراسي؛ يحتاج سنداً بالغاً في المدرسة."),
    ([("A", 3), ("W", 2)], "تجنّب التفكير مع الخوف من الصعب يشير إلى هشاشة ثقة أكثر منها قصوراً معرفياً."),
]


def parse_details(mid, details):
    """أسئلة النموذج: رقم، نوع، سرعة، نص السؤال، إجابة الطالب، النمط، الزمن."""
    out, qs = [], CATALOG[mid]["questions"]
    for line in str(details).splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        n, kind = int(m.group(1)), m.group(2)
        k, _, sp = kind.partition("/")
        out.append(dict(n=n, kind=k, speed=sp or "—", q=qs[n - 1]["q"] if k == "base" and n <= len(qs) else "",
                        answer=m.group(3), type=(m.group(4) or "").strip(), sec=float(m.group(5))))
    return out


def faking_level(v):
    return "منخفض" if v < 20 else "معتدل" if v < 40 else "مرتفع" if v < 60 else "مرتفع جداً"


def _behavior(rows):
    per = []
    imp_t = tot_t = 0
    for r in rows.itertuples():
        i, b, sl = lat_counts(r.latency)
        imp_t += i; tot_t += i + b + sl
        pen = [("اندفاع زمني", 50 if r.impulsive else 0), ("نص عشوائي", 20 if r.gibberish else 0),
               ("تجميل الصورة", round(0.3 * (r.faking if pd.notna(r.faking) else 0), 1))]
        per.append(dict(id=r.model, title=CATALOG[r.model]["title"], faking=float(r.faking) if pd.notna(r.faking) else 0.0,
                        reliability=r.reliability, trust=float(r.trust) * 100, avg=r.avg_sec, impulsive=bool(r.impulsive),
                        imp_n=i, bal_n=b, slow_n=sl, dual=str(r.dual), alerts=str(r.alerts) if str(r.alerts) != "nan" else "",
                        penalties=[x for x in pen if x[1]]))
    fk = [x["faking"] for x in per]
    return dict(per=per, faking_mean=float(np.mean(fk)) if fk else 0.0, faking_max=max(fk) if fk else 0.0,
                faking_hi=sum(1 for x in fk if x >= 40), imp_ratio=imp_t / tot_t if tot_t else 0.0,
                imp_models=sum(x["impulsive"] for x in per), trust_mean=float(np.mean([x["trust"] for x in per])) if per else None,
                slow_better=sum("slowBetter" in x["dual"] for x in per))


def search(store, text="", school="", grade="", issue=""):
    s = store.students()
    for col, q in (("name", text), ("school", school), ("grade", grade)):
        q = ar_norm(q)
        if q:
            s = s[s[col].map(ar_norm).str.contains(q, regex=False)]
    q = ar_norm(issue)
    if q:
        keys = set(store.clean[store.clean["type_name"].map(ar_norm).str.contains(q, regex=False)]["skey"])
        pr = store.profiles
        keys |= set(pr[(pr["attention"].map(ar_norm) + " " + pr["strengths"].map(ar_norm)).str.contains(q, regex=False)]["skey"])
        s = s[s["skey"].isin(keys)]
    return s.reset_index(drop=True)


def profile_items(strings):
    return [x.strip() for x in str(strings).split("|") if x.strip() and x.strip().lower() != "nan"]


def build_profile(store, skey: str) -> dict:
    rows = store.clean[store.clean["skey"] == skey].sort_values("model")
    prof = store.profiles[store.profiles["skey"] == skey]
    pr = prof.iloc[0] if len(prof) else None
    st = store.students()
    me = st[st["skey"] == skey].iloc[0]
    P = {"name": me["name"], "grade": me["grade"], "school": me["school"], "skey": skey}
    # 1) بطاقة الهوية
    models = []
    for r in rows.itertuples():
        models.append(dict(id=r.model, title=CATALOG[r.model]["title"], type=r.type_name, dist=r.dist_p,
                           avg=r.avg_sec, rel=r.reliability, faking=r.faking, impulsive=bool(r.impulsive),
                           score=value_score(r.dist_p, r.model) if r.dist_p else 0.0))
    P["models"] = models
    P["n_done"] = len(models) or (int(float(pr["n_models"])) if pr is not None and pd.notna(pr["n_models"]) else 0)
    P["avg_time"] = float(rows["avg_sec"].mean()) if len(rows) else None
    P["mean_trust"] = float(rows["trust"].mean()) if len(rows) else None
    # 2) نقاط القوة / 3) ما يحتاج انتباهاً (من القيم 1/-1 لكل نمط غالب)
    strengths = [f"{m['title']}: {m['type']}" for m in models if TYPE_VALUE[m["id"]][int(rows[rows.model == m["id"]]["type_idx"].iloc[0]) - 1] == 1]
    attention = [f"{m['title']}: {m['type']}" for m in models if TYPE_VALUE[m["id"]][int(rows[rows.model == m["id"]]["type_idx"].iloc[0]) - 1] == -1]
    if pr is not None:
        strengths = strengths or profile_items(pr["strengths"])
        attention = attention or profile_items(pr["attention"])
    P["strengths"], P["attention"] = strengths, attention
    # 3) جذور
    have = {(m["id"], int(rows[rows.model == m["id"]]["type_idx"].iloc[0]) - 1) for m in models}
    causes = [t for cond, t in ROOT_RULES if all(c in have for c in cond)]
    if pr is not None:
        causes += profile_items(pr["causes"])
    P["causes"] = causes
    # 4) الصدق والتناقض
    problems = " ".join(str(x) for x in rows["problems"].dropna() if str(x).strip() and str(x) != "nan").strip()
    norm = ar_norm(problems)
    fk = float(rows["faking"].mean()) if len(rows) else 0.0
    slow_better = int(rows["dual"].astype(str).str.contains("slowBetter").sum())
    deny = bool(norm) and bool(_DENY.search(norm)) and len(norm) < 40
    themes = sorted({t for kws, t in THEMES for k in kws if k in norm})
    risk_hits = [w for w in RISK_WORDS if w in norm]
    notes = []
    if deny and (fk >= 40 or slow_better >= 2 or len(attention) >= 3):
        notes.append("كتب أنه لا يعاني من مشكلة بينما تشير المؤشرات إلى تجميل الصورة أو تعدّد نقاط الانتباه؛ "
                     "قد يكون ذلك إنكاراً ظاهرياً أو كبتاً دفاعياً — يُفضَّل حوار فردي آمن غير اتهامي.")
    elif not norm:
        notes.append("لم يكتب الطالب شيئاً في حقل المشاكل.")
    if rows["gibberish"].any():
        notes.append("بعض نصوص المشاكل عشوائية فاستُبعدت من التحليل النصي.")
    if fk >= 55:
        notes.append(f"مؤشر تجميل الصورة مرتفع ({fk:.0f}%): إجاباته المتأنية أفضل من السريعة، وهذا إرشادي لا حكم.")
    P["honesty"] = dict(notes=notes, themes=themes, problems=problems, faking=fk)
    P["safety"] = risk_hits
    # 5) التنبؤ
    risk, src = None, "تقدير بالقواعد"
    f = store.features
    if f is not None and skey in f.keys:
        i = f.keys.index(skey)
        risk, src = float(f.risk[i]), "مؤشر قاعدي (تسمية ضعيفة)"
        if store.trainer.model is not None or store.trainer.load():
            try:
                _, risk = store.trainer.predict_student(f, i)
                src = f"الشبكة العصبية ({store.trainer.backend})"
            except Exception:
                pass
    P["risk"], P["risk_src"] = risk, src
    P["forecast"] = [str(x) for x in rows["forecast"].dropna() if str(x) != "nan"][-2:]
    # 6) خطة التدخل
    plans = []
    for m in models:
        ti = int(rows[rows.model == m["id"]]["type_idx"].iloc[0]) - 1
        if TYPE_VALUE[m["id"]][ti] == -1:
            plans.append(CATALOG[m["id"]]["plans"][ti])
    P["plan"] = dict(
        teacher=plans[:3] or ["لا توجد نقاط انتباه واضحة؛ أبقِ التحدي مناسباً لقدراته وامنحه مسؤوليات صفية."],
        counselor=(["ابدأ بحوار فردي هادئ يبدأ بنقاط قوته."] +
                   (["راجع النص الذي كتبه عن مشاكله وتحقق من أي كلمة تتعلق بالأمان فوراً."] if risk_hits else []) +
                   (["اربط الحوار بمحور: " + "، ".join(themes) + "."] if themes else [])),
        parent=["ابنكم يملك مواطن قوة حقيقية؛ أثنوا على جهده لا على درجته فقط.",
                "خصّصوا وقتاً قصيراً يومياً للاستماع إليه دون تقييم.",
                "تواصلوا مع المرشد إن لاحظتم تغيراً في نومه أو مزاجه."])
    P["twins"] = twins(store, skey)
    P["behavior"] = _behavior(rows) if len(rows) else None
    detail = {}
    for r in rows.itertuples():
        ti = int(r.type_idx) - 1 if pd.notna(r.type_idx) else None
        cat = CATALOG[r.model]
        detail[r.model] = dict(
            id=r.model, title=cat["title"], types=cat["types"], ti=ti, type=r.type_name, dist=r.dist_p, avg=r.avg_sec,
            total=r.total_sec, reliability=r.reliability, faking=float(r.faking) if pd.notna(r.faking) else 0.0,
            impulsive=bool(r.impulsive), lat=lat_counts(r.latency), dual=str(r.dual), alerts="" if str(r.alerts) == "nan" else str(r.alerts),
            result="" if str(r.result) == "nan" else str(r.result), forecast="" if str(r.forecast) == "nan" else str(r.forecast),
            problems="" if str(r.problems) == "nan" else str(r.problems), gibberish=bool(r.gibberish), trust=float(r.trust) * 100,
            level=TYPE_VALUE[r.model][ti] if ti is not None else 0, plan=cat["plans"][ti] if ti is not None else "",
            questions=parse_details(r.model, r.details), ts=r.ts)
    P["detail"] = detail
    return P


def twins(store, skey, top=3):
    f = store.features
    if f is None or skey not in f.keys:
        return []
    i = f.keys.index(skey)
    out = []
    for j, k in enumerate(f.keys):
        if j == i:
            continue
        common = np.where((f.obs[i] > 0) & (f.obs[j] > 0))[0]
        if len(common) < 2:
            continue
        sims = []
        for m in common:
            a, b = f.dist[i, m * K:(m + 1) * K], f.dist[j, m * K:(m + 1) * K]
            sims.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        out.append((float(np.mean(sims)), len(common), k))
    out.sort(reverse=True)
    st = store.students().set_index("skey")
    return [dict(name=st.loc[k, "name"], sim=s, common=c) for s, c, k in out[:top]]


def to_html(P: dict, anonymous=False) -> str:
    e = lambda s: html.escape(str(s))
    name = "طالب (مُعمّى)" if anonymous else P["name"]
    li = lambda xs: "<ul>" + "".join(f"<li>{e(x)}</li>" for x in xs) + "</ul>" if xs else "<p>—</p>"
    h = [f'<html dir="rtl"><body style="font-family:Tahoma;font-size:12pt">',
         f'<h2>ملف الحالة: {e(name)}</h2>',
         f'<p>{"" if anonymous else e(P["grade"]) + " — " + e(P["school"]) + " | "}النماذج المنجزة: {P["n_done"]}'
         + (f' | متوسط زمن السؤال: {P["avg_time"]:.1f} ث' if P["avg_time"] else "")
         + (f' | الموثوقية الإجمالية: {P["mean_trust"] * 100:.0f}%' if P["mean_trust"] is not None else "") + "</p>"]
    if P["safety"]:
        h.append('<p style="background:#fde2e2;padding:6px"><b>⚠ تنبيه أمان:</b> ورد في نص الطالب ما يستدعي اهتماماً فورياً '
                 'من المرشد (' + e("، ".join(P["safety"])) + '). تواصل مع الطالب اليوم وبشكل مباشر وهادئ.</p>')
    h.append("<h3>1) البطاقة المعرفية والسلوكية</h3><table border='1' cellpadding='4' cellspacing='0'><tr><th>النموذج</th><th>النمط الغالب</th><th>زمن/سؤال</th><th>موثوقية</th></tr>")
    for m in P["models"]:
        flag = " ⚡اندفاع" if m["impulsive"] else ""
        h.append(f"<tr><td>{e(m['id'])} — {e(m['title'])}</td><td>{e(m['type'])}{flag}</td><td>{m['avg']:.1f}</td><td>{e(m['rel'])}</td></tr>")
    h.append("</table><h3>2) خريطة نقاط القوة</h3>" + li(P["strengths"]))
    h.append("<h3>3) ما يحتاج انتباهاً والجذور المحتملة</h3>" + li(P["attention"]) + "<b>الجذور المحتملة:</b>" + li(P["causes"]))
    hn = P["honesty"]
    h.append("<h3>4) الصدق والتناقض</h3>" + li(hn["notes"]) +
             (f"<p>ما كتبه الطالب: «{e(hn['problems'])}»</p>" if hn["problems"] and not anonymous else "") +
             (f"<p>مواضيع مرصودة: {e('، '.join(hn['themes']))}</p>" if hn["themes"] else ""))
    b = P.get("behavior")
    if b:
        h.append(f"<h3>الموثوقية والاندفاع وتجميل الصورة</h3><p>تجميل الصورة: متوسط {b['faking_mean']:.0f}% ({faking_level(b['faking_mean'])})، "
                 f"الأعلى {b['faking_max']:.0f}% | الاندفاع: {b['imp_ratio'] * 100:.0f}% من الإجابات، {b['imp_models']} نماذج | "
                 f"الموثوقية: {b['trust_mean']:.0f}%</p><p><small>تجميل الصورة مؤشر ميل لإظهار الذات بشكل أفضل وليس كاشف كذب.</small></p>")
    h.append("<h3>5) التنبؤ المستقبلي</h3>")
    h.append(f"<p>مؤشر الخطر: <b>{P['risk'] * 100:.0f}%</b> ({e(P['risk_src'])}). هذا الرقم مؤشر أولي وليس نتيجة مُعايَرة بحالات تسرّب فعلية.</p>"
             if P["risk"] is not None else "<p>لا تتوفر ميزات كافية.</p>")
    h.append(li(P["forecast"]))
    h.append("<h3>6) خطة التدخل</h3><b>رسالة للمعلم:</b>" + li(P["plan"]["teacher"]) +
             "<b>رسالة للمرشد:</b>" + li(P["plan"]["counselor"]) + "<b>رسالة لولي الأمر:</b>" + li(P["plan"]["parent"]))
    if P["twins"] and not anonymous:
        h.append("<h3>توائم سلوكية</h3>" + li([f"{t['name']} — تشابه {t['sim'] * 100:.0f}% على {t['common']} نماذج مشتركة" for t in P["twins"]]))
    h.append(f"<hr><p style='color:#666'><small>{e(DISCLAIMER)}</small></p></body></html>")
    return "".join(h)
