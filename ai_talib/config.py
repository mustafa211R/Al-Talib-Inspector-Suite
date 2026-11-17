"""الإعدادات العامة، كتالوج النماذج A–Z، وأدوات تطبيع النص العربي."""
import json
import re
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DB_DIR = APP_DIR / "db"
MODELS_DIR = DB_DIR / "models"
BACKUP_DIR = DB_DIR / "backups"
MODEL_FILE = APP_DIR / "talib_model.keras"          # ملف أوزان الشبكة (keras.model)
PROFILES_CSV = DB_DIR / "profiles_master.csv"
ANALYSIS_CSV = DB_DIR / "analysis.csv"

_cat = json.loads((Path(__file__).parent / "data" / "catalog.json").read_text(encoding="utf-8"))
CATALOG = _cat["models"]          # id -> {title, icon, types[4], plans[4], questions[]}
TYPE_VALUE = _cat["value"]        # id -> [1|0|-1 ×4]  (1 قوة، 0 محايد، -1 يحتاج انتباه) — من engine.js
MODEL_IDS = list(CATALOG)         # A..Z
K = 4                             # عدد أنماط كل نموذج

# أعمدة ورقة النموذج كما في Code.gs (بالترتيب نفسه)
SHEET_COLS = ["ts", "name", "grade", "school", "total_sec", "avg_sec", "type_idx", "type_name",
              "dist", "details", "problems", "result", "reliability", "alerts", "latency",
              "dual", "faking", "forecast"]
PROFILE_COLS = ["key", "name", "grade", "school", "n_models", "done_list", "strengths",
                "attention", "causes", "updated"]

IMPULSE_SEC = 1.5                 # عتبة الاندفاع الزمني (بوابة 2)

# عدسات التخصص (رموز النماذج حسب Code.gs — ملاحظة: في الورقة رُمز الغضب J بالخطأ، والصحيح N)
LENSES = {
    "علم النفس التربوي والعيادي": list("DXNSBM"),
    "علم الاجتماع التربوي": list("ZOUT"),
    "المناهج وطرائق التدريس": list("ACFGJKVR"),
    "كل النماذج": MODEL_IDS,
}

# مواضيع النصوص الحرة وكلمات الخطر (مطابقة لـ engine.js)
THEMES = [
    (["امتحان", "اختبار", "درجة", "درجات", "رسوب"], "قلق الامتحانات والدرجات"),
    (["ابي", "امي", "اهلي", "البيت", "المنزل", "اخي"], "علاقة البيت والأسرة"),
    (["معلم", "مدرس", "استاذ", "مدرسه"], "العلاقة مع المعلم أو المدرسة"),
    (["ملل", "مملة", "ممل"], "الملل"),
    (["هاتف", "موبايل", "العاب", "يوتيوب", "تيك"], "الهاتف والألعاب"),
    (["نوم", "انام", "تعب", "تعبان", "صداع", "جوع"], "التعب أو النوم"),
    (["صديق", "اصدقاء", "وحيد", "وحده", "تنمر", "يضحكون"], "الأصدقاء والعلاقات"),
    (["افهم", "صعب", "صعبه", "فهم", "حفظ"], "صعوبة الفهم أو الحفظ"),
    (["خوف", "اخاف", "قلق", "اتوتر"], "الخوف والتوتر"),
]
RISK_WORDS = ["ضرب", "يضربني", "عنف", "تنمر", "يتنمر", "تحرش", "جوع", "اكره نفسي", "اموت",
              "انتحار", "اوذي نفسي", "اخاف من ابي", "اخاف من امي"]

_DIAC = re.compile(r"[\u064B-\u0652\u0670\u0640]")
_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})


def ar_norm(s) -> str:
    s = "" if s is None else str(s)
    if s.lower() == "nan":
        return ""
    s = _DIAC.sub("", s).translate(_MAP)
    return re.sub(r"\s+", " ", s).strip()


_ORD = {"اول": 1, "ثاني": 2, "ثالث": 3, "رابع": 4, "خامس": 5, "سادس": 6}


def grade_key(g) -> str:
    t = ar_norm(g)
    for tok in re.split(r"[\s\-_/،,]+", t):
        tok = tok[2:] if tok.startswith("ال") else tok
        if tok in _ORD:
            return str(_ORD[tok])
    return t


_SCHOOL_STOP = {"متوسطه", "ثانويه", "ابتدائيه", "ابتدايه", "مدرسه", "ذي", "ذو", "ذى", "اعداديه"}


def school_key(s) -> str:
    toks = [(w[2:] if w.startswith("ال") and len(w) > 3 else w) for w in ar_norm(s).split()]
    return " ".join(w for w in toks if w not in _SCHOOL_STOP)


def student_key(name, grade, school) -> str:
    return f"{ar_norm(name)}|{grade_key(grade)}|{school_key(school)}"


def ensure_dirs():
    for d in (DB_DIR, MODELS_DIR, BACKUP_DIR):
        d.mkdir(parents=True, exist_ok=True)
