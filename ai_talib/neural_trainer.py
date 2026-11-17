"""المعلم والمدرب: متجهات سلوكية، شبكة متعددة المهام (Keras)، وتعلّم تزايدي. بديل scikit-learn إن غاب Keras."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import K, MODEL_FILE, MODEL_IDS, TYPE_VALUE

N = len(MODEL_IDS)
D = N * K                   # 104: توزيع الأنماط
FEAT_DIM = D + N + 5        # + قناع النماذج المنجزة + 5 مؤشرات عامة

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import keras
    from keras import layers
    HAS_KERAS = True
except Exception:       # TensorFlow/Keras غير مثبت
    keras = layers = None
    HAS_KERAS = False


# ---------------------------------------------------------------- هندسة الميزات
class Features:
    """keys, dist(n,104), obs(n,26 قناع), X(n,FEAT_DIM), risk(n)، w(n,104 أوزان الفئات)."""
    def __init__(self, keys, dist, obs, glob, risk, w):
        self.keys, self.dist, self.obs, self.glob, self.risk, self.w = keys, dist, obs, glob, risk, w

    def x(self, dist=None, obs=None):
        dist = self.dist if dist is None else dist
        obs = self.obs if obs is None else obs
        return np.hstack([dist * np.repeat(obs, K, axis=1), obs, self.glob]).astype("float32")


def value_score(p, mid):
    return float(np.dot(p, TYPE_VALUE[mid]))             # ∈ [-1, 1]


def build_features(clean: pd.DataFrame, class_weights: dict) -> Features:
    keys = sorted(clean["skey"].unique())
    idx = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    dist, obs, w = np.zeros((n, D), "float32"), np.zeros((n, N), "float32"), np.ones((n, D), "float32")
    for r in clean.itertuples():
        if not r.trainable:          # الصفوف المندفعة لا تعدّل أوزان التشخيص المعرفي
            continue
        i, m = idx[r.skey], MODEL_IDS.index(r.model)
        dist[i, m * K:(m + 1) * K] = r.dist_p
        obs[i, m] = 1
        cw = class_weights.get(r.model)
        if cw and 1 <= (r.type_idx or 0) <= K:
            w[i, m * K:(m + 1) * K] = cw[int(r.type_idx) - 1]
    glob, risk = np.zeros((n, 5), "float32"), np.zeros(n, "float32")
    for k, g in clean.groupby("skey"):
        i = idx[k]
        a = g["avg_sec"].dropna()
        mean_t = a.mean() if len(a) else 0
        cv = a.std(ddof=0) / mean_t if len(a) > 1 and mean_t > 0 else 0
        fk = g["faking"].fillna(0).mean() / 100
        imp = g["impulsive"].mean()
        glob[i] = [np.log1p(mean_t) / np.log(30), min(cv, 2) / 2, fk, imp, len(g) / N]
        att = [(1 - value_score(r.dist_p, r.model)) / 2 for r in g.itertuples() if r.trainable]
        nm = len(att)
        r0 = 0.6 * (np.mean(att) if att else 0.35) + 0.25 * fk + 0.15 * imp
        risk[i] = (nm * r0 + 2 * 0.35) / (nm + 2)       # انكماش نحو المتوسط عند قلة النماذج
    return Features(keys, dist, obs, glob, risk, w)


def masked_copies(f: Features, aug=8, p_hide=0.3, seed=0):
    """تعزيز ذاتي الإشراف: نُخفي نماذج منجزة من المدخل ونطلب من الشبكة استرجاعها."""
    rng = np.random.default_rng(seed)
    X, Y, R = [], [], []
    for _ in range(aug):
        hide = (rng.random(f.obs.shape) < p_hide) & (f.obs > 0)
        obs_in = f.obs * (~hide)
        X.append(f.x(obs=obs_in))
        Y.append(np.hstack([f.dist, np.repeat(f.obs, K, axis=1) * f.w]))
        R.append(f.risk)
    return np.vstack(X), np.vstack(Y).astype("float32"), np.concatenate(R).astype("float32")


# ---------------------------------------------------------------- الشبكة
if HAS_KERAS:
    @keras.saving.register_keras_serializable(package="talib")
    def masked_mse(y_true, y_pred):
        import keras.ops as ops
        d, m = y_true[:, :D], y_true[:, D:]
        return ops.sum(m * ops.square(d - y_pred), axis=-1) / (ops.sum(m, axis=-1) + 1e-6)


def _build_keras(lr):
    inp = keras.Input((FEAT_DIM,))
    h = layers.Dense(128, activation="relu")(inp)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(64, activation="relu")(h)
    recon = layers.Dense(D, activation="sigmoid", name="recon")(h)
    risk = layers.Dense(1, activation="sigmoid", name="risk")(h)
    m = keras.Model(inp, [recon, risk])
    m.compile(optimizer=keras.optimizers.Adam(lr), loss={"recon": masked_mse, "risk": "mse"},
              loss_weights={"recon": 1.0, "risk": 0.5})
    return m


def per_model_probs(vec104):
    v = np.asarray(vec104, float).reshape(N, K) + 1e-6
    return v / v.sum(axis=1, keepdims=True)


class Trainer:
    def __init__(self, path: Path = MODEL_FILE):
        self.path = Path(path)
        self.backend = "keras" if HAS_KERAS else "sklearn"
        self.model = None
        self.col_mean = np.full(D, 1.0 / K, "float32")
        self.history = []

    # ---- تحميل/حفظ
    def exists(self):
        return (self.path if HAS_KERAS else self.path.with_suffix(".joblib")).exists()

    def load(self):
        if not self.exists():
            return False
        if HAS_KERAS:
            self.model = keras.models.load_model(self.path, custom_objects={"masked_mse": masked_mse}, compile=False)
            self.model.compile(optimizer=keras.optimizers.Adam(1e-4), loss={"recon": masked_mse, "risk": "mse"},
                               loss_weights={"recon": 1.0, "risk": 0.5})
        else:
            import joblib
            blob = joblib.load(self.path.with_suffix(".joblib"))
            self.model, self.col_mean = blob["m"], blob["cm"]
        return True

    def save(self):
        if HAS_KERAS:
            self.model.save(self.path)
        else:
            import joblib
            joblib.dump({"m": self.model, "cm": self.col_mean}, self.path.with_suffix(".joblib"))

    # ---- تدريب
    def train(self, f: Features, epochs=60, incremental=True, on_epoch=None, log=print):
        if len(f.keys) < 3:
            raise ValueError("عدد الطلاب أقل من 3 — لا يكفي للتدريب.")
        X, Y, R = masked_copies(f)
        loaded = incremental and self.load()
        log(f"{'تحميل الأوزان السابقة وتدريب تزايدي (معدل تعلّم منخفض)' if loaded else 'تدريب جديد من الصفر'}"
            f" — الواجهة: {self.backend} — طلاب: {len(f.keys)}")
        if HAS_KERAS:
            if not loaded:
                self.model = _build_keras(1e-3)
            cb = keras.callbacks.LambdaCallback(on_epoch_end=lambda e, lg: (
                self.history.append(float(lg["loss"])), on_epoch and on_epoch(e, float(lg["loss"]))))
            self.model.fit(X, {"recon": Y, "risk": R}, epochs=epochs if not loaded else max(10, epochs // 3),
                           batch_size=32, verbose=0, callbacks=[cb])
        else:
            from sklearn.neural_network import MLPRegressor
            # بديل: الأهداف غير المرصودة تُملأ بمتوسط العمود (لا يوجد تقنيع للخسارة هنا)
            obs_cols = Y[:, D:] > 0
            sums = np.where(obs_cols, Y[:, :D], 0).sum(0)
            cnt = obs_cols.sum(0)
            self.col_mean = np.where(cnt > 0, sums / np.maximum(cnt, 1), 1.0 / K).astype("float32")
            T = np.where(obs_cols, Y[:, :D], self.col_mean)
            T = np.hstack([T, R[:, None]])
            if not loaded or self.model is None:
                self.model = MLPRegressor(hidden_layer_sizes=(128, 64), learning_rate_init=1e-3, random_state=0)
            else:
                self.model.learning_rate_init = 1e-4
            for e in range(epochs if not loaded else max(10, epochs // 3)):
                self.model.partial_fit(X, T)
                self.history.append(float(self.model.loss_))
                if on_epoch:
                    on_epoch(e, self.history[-1])
        self.save()
        log(f"تم حفظ النموذج: {self.path.name if HAS_KERAS else self.path.with_suffix('.joblib').name}")

    # ---- تنبؤ
    def predict(self, x):
        x = np.atleast_2d(x).astype("float32")
        if HAS_KERAS:
            rec, risk = self.model.predict(x, verbose=0)
            return rec, risk.ravel()
        out = self.model.predict(x)
        return out[:, :D], np.clip(out[:, D], 0, 1)

    def predict_student(self, f: Features, i):
        rec, risk = self.predict(f.x()[i])
        return per_model_probs(rec[0]), float(risk[0])

    # ---- تقييم صادق: k-fold على مستوى الطلاب، إخفاء نماذج منجزة، ومقارنة بخط أساس الأغلبية
    def cross_validate(self, f: Features, folds=5, epochs=40, log=print):
        n = len(f.keys)
        if n < 10:
            raise ValueError("التقييم المتقاطع يحتاج 10 طلاب على الأقل.")
        rng = np.random.default_rng(1)
        order = rng.permutation(n)
        hit = tot = base_hit = 0
        # خط الأساس: النمط الأكثر شيوعاً لكل نموذج في بيانات التدريب
        for fi in range(folds):
            te = order[fi::folds]
            tr = np.setdiff1d(order, te)
            sub = Features([f.keys[i] for i in tr], f.dist[tr], f.obs[tr], f.glob[tr], f.risk[tr], f.w[tr])
            t = Trainer(self.path.with_name("_cv_tmp.keras"))
            t.train(sub, epochs=epochs, incremental=False, log=lambda *_: None)
            maj = [np.bincount(f.dist[tr][f.obs[tr][:, m] > 0, m * K:(m + 1) * K].argmax(1), minlength=K).argmax()
                   if (f.obs[tr][:, m] > 0).any() else 0 for m in range(N)]
            for i in te:
                seen = np.where(f.obs[i] > 0)[0]
                if len(seen) < 2:
                    continue
                for m in seen:                      # نخفي نموذجاً واحداً ونتنبأ به من بقية النماذج
                    o = f.obs[i].copy(); o[m] = 0
                    pred, _ = t.predict(f.x(obs=o[None, :].repeat(len(f.keys), 0))[i])
                    p = per_model_probs(pred[0])[m]
                    truth = f.dist[i, m * K:(m + 1) * K].argmax()
                    hit += int(p.argmax() == truth); base_hit += int(maj[m] == truth); tot += 1
            log(f"طيّة {fi + 1}/{folds}")
        for p in (self.path.with_name("_cv_tmp.keras"), self.path.with_name("_cv_tmp.joblib")):
            p.unlink(missing_ok=True)
        return dict(accuracy=hit / max(tot, 1), baseline=base_hit / max(tot, 1), trials=tot)
