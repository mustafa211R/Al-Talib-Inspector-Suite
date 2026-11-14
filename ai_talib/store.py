"""مخزن البيانات المشترك: يحمّل db/، يدقّق، ويبني الميزات."""
import pandas as pd

from .auditor import audit, load_models_long
from .config import PROFILES_CSV, PROFILE_COLS, student_key
from .neural_trainer import Features, Trainer, build_features


class Store:
    def __init__(self):
        self.raw = self.audit = self.profiles = self.features = None
        self.trainer = Trainer()

    def refresh(self):
        self.raw = load_models_long()
        self.audit = audit(self.raw)
        if PROFILES_CSV.exists():
            p = pd.read_csv(PROFILES_CSV, dtype=object, encoding="utf-8-sig").iloc[:, :len(PROFILE_COLS)]
            p.columns = PROFILE_COLS[:p.shape[1]]
            p["skey"] = [student_key(n, g, s) for n, g, s in zip(p["name"], p["grade"], p["school"])]
            # دمج صيغ كتابة الاسم/الصف/المدرسة المختلفة للطالب نفسه
            self.profiles = p.sort_values("updated", ascending=False).drop_duplicates("skey")
        else:
            self.profiles = pd.DataFrame(columns=PROFILE_COLS + ["skey"])
        c = self.audit.clean
        self.features = build_features(c, self.audit.class_weights) if len(c) else None
        return self

    @property
    def clean(self):
        return self.audit.clean

    def students(self) -> pd.DataFrame:
        c = self.clean
        base = (c.sort_values("ts").groupby("skey")
                .agg(name=("name", "last"), grade=("grade", "last"), school=("school", "last"),
                     n_models=("model", "nunique"), last=("ts", "max")).reset_index()) if len(c) else \
            pd.DataFrame(columns=["skey", "name", "grade", "school", "n_models", "last"])
        pr = self.profiles
        if len(pr):                     # طلاب لديهم ملف شامل فقط (بلا صفوف نماذج)
            extra = pr[~pr["skey"].isin(base["skey"])][["skey", "name", "grade", "school", "n_models"]].copy()
            extra["n_models"] = pd.to_numeric(extra["n_models"], errors="coerce")
            extra["last"] = pd.NaT
            base = pd.concat([base, extra], ignore_index=True)
        return base.sort_values("name").reset_index(drop=True)
