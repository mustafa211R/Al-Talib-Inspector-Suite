"""اختبار شامل للنواة (بدون واجهة): استيعاب -> تدقيق -> ميزات -> تدريب -> ملف طالب -> مختبر بحثي."""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ai_talib.config as C

tmp = Path(tempfile.mkdtemp())
C.DB_DIR, C.MODELS_DIR, C.BACKUP_DIR = tmp / "db", tmp / "db/models", tmp / "db/backups"
C.PROFILES_CSV, C.ANALYSIS_CSV, C.MODEL_FILE = C.DB_DIR / "profiles_master.csv", C.DB_DIR / "analysis.csv", tmp / "m.keras"
import ai_talib.cloud_fetcher as CF, ai_talib.auditor as A, ai_talib.store as S, ai_talib.neural_trainer as NT, ai_talib.research_lab as R
import ai_talib.clinical_profiler as CP
for mod in (CF, A, S, NT, R):
    for n in ("DB_DIR", "MODELS_DIR", "BACKUP_DIR", "PROFILES_CSV", "ANALYSIS_CSV", "MODEL_FILE"):
        if hasattr(mod, n):
            setattr(mod, n, getattr(C, n))
from tools.make_demo_xlsx import make

x = make(tmp / "demo.xlsx")
print("ingest:", CF.ingest_workbook(x, log=lambda *_: None))
st = S.Store()
st.trainer = NT.Trainer(C.MODEL_FILE)
st.refresh()
print(st.audit.summary())
print("features", st.features.X.shape if hasattr(st.features, "X") else st.features.x().shape, "backend", st.trainer.backend)
losses = []
st.trainer.train(st.features, epochs=30, on_epoch=lambda e, l: losses.append(l), log=print)
print("loss first/last", round(losses[0], 4), round(losses[-1], 4))
st.trainer.train(st.features, epochs=30, incremental=True, log=print)      # تعلّم تزايدي
print("CV:", st.trainer.cross_validate(st.features, folds=4, epochs=20, log=lambda *_: None))
s = CP.search(st, text="تجريبي 3")
print("search hits", len(s), list(s["name"])[:3])
P = CP.build_profile(st, s.iloc[0]["skey"])
html = CP.to_html(P)
print("profile html chars", len(html), "risk", P["risk"], P["risk_src"], "twins", len(P["twins"]))
d, n = R.type_distribution(st.clean, "A")
print(d.round(1).to_string(), n)
r, p, nn = R.correlations(st.clean, list(C.LENSES["علم النفس التربوي والعيادي"]))
print("corr shape", r.shape)
print(R.read_result("D", "S", r.loc["D", "S"], p.loc["D", "S"], nn.loc["D", "S"]) if "D" in r and "S" in r else "")
ia, alpha, n_alpha = R.item_analysis(st.clean, "A")
print("items", len(ia), "alpha", alpha, n_alpha)
an = R.anonymize(st.clean)
print("anon cols", list(an.columns)[:6], an["student_id"].iloc[0])
print(R.build_model_js("Q2", "اختبار", "🧪", [("أ", "و", "ت")] * 4, [("س", ["1", "2", "3", "4"])])[:60])
import ai_talib.insights as INS
I = st.insight()
print("insight n", I.n, "decisions", len(I.decisions), "assoc", len(I.assoc), "att", len(I.att), "causes", len(I.causes))
assert len(INS.decisions_html(I)) > 300
Pd = CP.build_profile(st, s.iloc[0]["skey"])
assert Pd["behavior"] and Pd["detail"], "تفاصيل السلوك والأنماط"
# حالة فارغة
for f in C.MODELS_DIR.glob("*.csv"): f.unlink()
if C.PROFILES_CSV.exists(): C.PROFILES_CSV.unlink()
e = S.Store().refresh().insight()
print("empty insight n", e.n, len(e.decisions))
shutil.rmtree(tmp)
print("ALL OK")
