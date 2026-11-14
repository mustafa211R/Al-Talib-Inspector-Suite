"""طبقة الاستيعاب: سحب Google Sheet كـ XLSX وتفكيكه إلى CSV (UTF-8-SIG) داخل db/ مع نسخ احتياطي."""
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from .config import (ANALYSIS_CSV, BACKUP_DIR, DB_DIR, MODEL_IDS, MODELS_DIR, PROFILES_CSV, ensure_dirs)

_MODEL_SHEET = [re.compile(r"^Model_([A-Z])_sheet$", re.I), re.compile(r"^sheet_([A-Z])_Model$", re.I),
                re.compile(r"^([A-Z])_Model$", re.I)]


def sheet_id(url: str) -> str:
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        raise ValueError("تعذّر استخراج معرّف الملف من الرابط. تأكد أنه رابط Google Sheet.")
    return m.group(1)


def download_xlsx(url: str, timeout: int = 90) -> Path:
    """يتطلب أن تكون المشاركة: 'أي شخص لديه الرابط — مشاهد'."""
    export = f"https://docs.google.com/spreadsheets/d/{sheet_id(url)}/export?format=xlsx"
    r = requests.get(export, timeout=timeout)
    r.raise_for_status()
    if "text/html" in r.headers.get("content-type", ""):
        raise PermissionError("الرابط غير متاح للعموم. اجعل المشاركة 'أي شخص لديه الرابط' (مشاهد) ثم أعد المحاولة.")
    out = Path(tempfile.gettempdir()) / "talib_download.xlsx"
    out.write_bytes(r.content)
    return out


def backup_db() -> Path | None:
    ensure_dirs()
    files = list(DB_DIR.glob("*.csv")) + list(MODELS_DIR.glob("*.csv"))
    if not files:
        return None
    dest = BACKUP_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    (dest / "models").mkdir(parents=True)
    for f in files:
        shutil.copy2(f, dest / ("models" if f.parent == MODELS_DIR else "") / f.name)
    return dest


def _model_id(sheet: str):
    for rx in _MODEL_SHEET:
        m = rx.match(sheet.strip())
        if m and m.group(1).upper() in MODEL_IDS:
            return m.group(1).upper()
    return None


def ingest_workbook(path, log=print) -> dict:
    """يفكّك مصنف XLSX إلى CSV. يعيد ملخصاً بعدد الصفوف لكل ورقة."""
    ensure_dirs()
    bk = backup_db()
    if bk:
        log(f"نسخة احتياطية: {bk.name}")
    sheets = pd.read_excel(path, sheet_name=None, dtype=object)
    summary = {}
    for name, df in sheets.items():
        df = df.dropna(how="all")
        mid = _model_id(name)
        if mid:
            df.to_csv(MODELS_DIR / f"{mid}.csv", index=False, encoding="utf-8-sig")
            summary[mid] = len(df)
        elif name.strip().lower() in ("student_profiles", "profiles_student"):
            df.to_csv(PROFILES_CSV, index=False, encoding="utf-8-sig")
            summary["profiles"] = len(df)
        elif "analysis" in name.lower():
            df.to_csv(ANALYSIS_CSV, index=False, encoding="utf-8-sig")
            summary["analysis"] = len(df)
        else:
            log(f"تم تجاهل ورقة غير معروفة: {name}")
    log(f"تم حفظ {len(summary)} ملف CSV في db/")
    return summary


def ingest_csv(path, model_id: str | None = None, log=print) -> dict:
    """استيراد CSV مفرد: ملف ملفات الطلاب (عمود المفتاح) أو نموذج محدد."""
    ensure_dirs()
    backup_db()
    df = pd.read_csv(path, dtype=object, encoding="utf-8-sig")
    if "المفتاح" in df.columns or "key" in df.columns:
        df.to_csv(PROFILES_CSV, index=False, encoding="utf-8-sig")
        log(f"ملفات الطلاب: {len(df)} صف")
        return {"profiles": len(df)}
    mid = (model_id or _model_id(Path(path).stem) or "").upper()
    if mid not in MODEL_IDS:
        raise ValueError("حدّد رمز النموذج (A–Z) لهذا الملف.")
    df.to_csv(MODELS_DIR / f"{mid}.csv", index=False, encoding="utf-8-sig")
    log(f"نموذج {mid}: {len(df)} صف")
    return {mid: len(df)}


def fetch_and_ingest(url: str, log=print) -> dict:
    log("جاري تنزيل المصنف من Google Sheets ...")
    return ingest_workbook(download_xlsx(url), log)
