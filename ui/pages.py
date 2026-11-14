"""صفحات التطبيق الأربع: الرئيسية، البيانات والنموذج، الطلاب، المختبر البحثي."""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLineEdit, QMessageBox,
                               QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ai_talib import clinical_profiler as CP, cloud_fetcher
from ai_talib import research_lab as RL
from ai_talib.config import CATALOG, LENSES, MODEL_IDS
from ai_talib.neural_trainer import HAS_KERAS
from .dialogs import BuilderDialog, ChartDialog, FilterDialog, TableDialog, TextDialog
from .theme import make_plot, set_ticks
from .widgets import Card, StatCard, button, chips_box, label


def restyle_plot(plot, pal):
    plot.setBackground(pal["card"])
    for ax in ("left", "bottom"):
        a = plot.getPlotItem().getAxis(ax); a.setPen(pg.mkPen(pal["border"])); a.setTextPen(pg.mkPen(pal["sub"]))


class Page(QScrollArea):
    """صفحة قابلة للتمرير بهوامش مريحة."""
    def __init__(self, win, title, sub=""):
        super().__init__()
        self.win = win
        self.setWidgetResizable(True)
        body = QWidget(); body.setObjectName("page"); self.setWidget(body)
        self.lay = QVBoxLayout(body); self.lay.setContentsMargins(32, 28, 32, 28); self.lay.setSpacing(18)
        self.lay.addWidget(label(title, "h1"))
        if sub:
            self.lay.addWidget(label(sub, "sub", True))

    def on_data(self): ...
    def on_theme(self, pal): ...


# ------------------------------------------------------------------ الرئيسية
class HomePage(Page):
    def __init__(self, win):
        super().__init__(win, "مرحباً بك في AI-Talib", "منظومة الفحص والتدقيق والتحليل التنبؤي — للاستخدام الإرشادي لا العقابي")
        row = QHBoxLayout(); row.setSpacing(14)
        self.s_students, self.s_rows, self.s_flag, self.s_model = (StatCard(t) for t in
                                                                   ("الطلاب", "صفوف صالحة للتدريب", "مستبعدة / مندفعة", "النموذج العصبي"))
        for s in (self.s_students, self.s_rows, self.s_flag, self.s_model):
            row.addWidget(s)
        self.lay.addLayout(row)
        acts = QHBoxLayout(); acts.setSpacing(14)
        for title, sub, txt, fn in (
                ("اسحب من السحابة", "حدّث قاعدة db/ من رابط Google Sheet", "بدء السحب", lambda: (win.go(1), win.pages[1].pull())),
                ("استورد ملفاً", "ملف XLSX للمصنف أو CSV لملفات الطلاب", "اختيار ملف", lambda: (win.go(1), win.pages[1].import_file())),
                ("ابحث عن طالب", "افتح ملف حالة كامل بنقرة واحدة", "فتح الطلاب", lambda: win.go(2))):
            c = Card(title, sub); c.add(button(txt, fn, True)); acts.addWidget(c)
        self.lay.addLayout(acts)
        self.chart_card = Card("توزيع الصفوف لكل نموذج", "أخضر: صالح للتدريب · برتقالي: اندفاع زمني · أحمر: مكرر مستبعد")
        self.plot = make_plot(win.pal, min_h=260); self.chart_card.add(self.plot)
        self.lay.addWidget(self.chart_card); self.lay.addStretch()

    def on_theme(self, pal): restyle_plot(self.plot, pal)

    def on_data(self):
        st = self.win.store; a = st.audit; pal = self.win.pal
        if a is None:
            return
        s = a.stats
        self.s_students.set(s["students"] or len(st.students()), "طالب فريد بعد دمج صيغ الكتابة")
        self.s_rows.set(s["trainable"], f"من أصل {s['total']} صفاً")
        self.s_flag.set(f"{s['duplicates']} / {s['impulsive']}", "تكرار / اندفاع زمني")
        self.s_model.set("جاهز" if st.trainer.exists() else "غير مدرَّب", "Keras" if HAS_KERAS else "scikit-learn (بديل)")
        self.plot.clear(); c = a.clean
        if len(c):
            x = np.arange(len(MODEL_IDS)); base = np.zeros(len(MODEL_IDS))
            ok = np.array([((c.model == m) & c.trainable).sum() for m in MODEL_IDS], float)
            imp = np.array([((c.model == m) & c.impulsive).sum() for m in MODEL_IDS], float)
            dup = np.array([(a.rejected.model == m).sum() if len(a.rejected) else 0 for m in MODEL_IDS], float)
            for v, col in ((ok, pal["ok"]), (imp, pal["warn"]), (dup, pal["bad"])):
                self.plot.addItem(pg.BarGraphItem(x=x, height=v, y0=base, width=0.7, brush=col)); base = base + v
            set_ticks(self.plot, MODEL_IDS)


# ------------------------------------------------------------------ البيانات والنموذج
class DataPage(Page):
    def __init__(self, win):
        super().__init__(win, "البيانات والنموذج", "ثلاث خطوات: اجلب البيانات، دقّقها، ثم درّب الشبكة")
        c1 = Card("١ · المصدر", "الصق رابط Google Sheet (المشاركة: أي شخص لديه الرابط — مشاهد) أو استورد ملفاً محلياً")
        self.url = QLineEdit(win.settings.value("url", "")); self.url.setPlaceholderText("https://docs.google.com/spreadsheets/d/...")
        c1.add(self.url)
        r = QHBoxLayout(); r.addWidget(button("سحب وتحديث", self.pull, True)); r.addWidget(button("استيراد XLSX / CSV", self.import_file)); r.addStretch()
        c1.lay.addLayout(r); self.lay.addWidget(c1)
        c2 = Card("٢ · التدقيق والتمحيص", "تكرار ذكي · عتبة الاندفاع 1.5 ث · نصوص عشوائية · موازنة الفئات")
        c2.add(button("تشغيل التدقيق", self.audit))
        self.gate_box = QVBoxLayout(); c2.lay.addLayout(self.gate_box); self.lay.addWidget(c2)
        c3 = Card("٣ · التدريب", "تحميل الأوزان السابقة وتحديثها بمعدل تعلّم منخفض إن وُجدت")
        r = QHBoxLayout(); r.addWidget(label("عدد الدورات", "sub"))
        self.epochs = QSpinBox(); self.epochs.setRange(5, 500); self.epochs.setValue(int(win.settings.value("epochs", 60)))
        r.addWidget(self.epochs); r.addSpacing(12)
        r.addWidget(button("تدريب / تحديث النموذج", self.train, True)); r.addWidget(button("تقييم صادق (CV)", self.cv)); r.addStretch()
        c3.lay.addLayout(r)
        self.loss = make_plot(win.pal, "خسارة التدريب", 200); self.line = self.loss.plot([], [], pen=pg.mkPen(win.pal["accent"], width=3))
        c3.add(self.loss); self.lay.addWidget(c3); self.lay.addStretch()

    def on_theme(self, pal):
        restyle_plot(self.loss, pal); self.line.setPen(pg.mkPen(pal["accent"], width=3))

    def pull(self):
        url = self.url.text().strip()
        if not url:
            return self.win.toast("ألصق رابط Google Sheet أولاً")
        self.win.settings.setValue("url", url)
        self.win.run_task("سحب البيانات من السحابة", lambda log: cloud_fetcher.fetch_and_ingest(url, log), self._ingested)

    def import_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "اختر ملفاً", "", "XLSX/CSV (*.xlsx *.csv)")
        if not f:
            return
        fn = (lambda log: cloud_fetcher.ingest_workbook(f, log)) if f.lower().endswith(".xlsx") else (lambda log: cloud_fetcher.ingest_csv(f, log=log))
        self.win.run_task("استيراد الملف", fn, self._ingested)

    def _ingested(self, summary):
        self.win.toast(f"تم استيراد {len(summary)} ملفاً")
        self.audit()

    def audit(self):
        def job(log):
            log("تحميل وتدقيق البيانات ..."); self.win.store.refresh(); return self.win.store.audit
        self.win.run_task("التدقيق والتمحيص", job, lambda a: (self.win.refresh_all(), self.win.toast("اكتمل التدقيق")))

    def on_data(self):
        while self.gate_box.count():
            w = self.gate_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        a = self.win.store.audit
        if a is None:
            return
        s = a.stats
        self.gate_box.addWidget(chips_box([f"{s['total']} صفاً", f"{s['students']} طالباً", f"{s['duplicates']} مكرراً مستبعداً",
                                           f"{s['impulsive']} اندفاع زمني", f"{s['gibberish']} نصاً عشوائياً",
                                           f"{s['trainable']} صالحاً للتدريب"], "info"))

    def train(self):
        st = self.win.store
        if st.features is None:
            return self.win.toast("لا توجد بيانات — ابدأ بالخطوة ١")
        curve = []
        self.win.settings.setValue("epochs", self.epochs.value())

        def job(log):
            st.trainer.train(st.features, epochs=self.epochs.value(), log=log, on_epoch=lambda e, l: curve.append(l))
            return list(curve)

        def done(c):
            self.line.setData(list(range(len(c))), c); self.win.refresh_all()
            TextDialog(self.win, "اكتمل التدريب",
                       f"<h3>تم حفظ النموذج</h3><p>الواجهة: {st.trainer.backend}</p><p>آخر خسارة: <b>{c[-1]:.4f}</b></p>"
                       f"<p>تنبيه: عدد الطلاب {len(st.features.keys)} صغير؛ استخدم «تقييم صادق» قبل الاعتماد على أي تنبؤ.</p>").exec()
        self.win.run_task("تدريب الشبكة العصبية", job, done)

    def cv(self):
        st = self.win.store
        if st.features is None:
            return self.win.toast("لا توجد بيانات")
        def done(r):
            better = r["accuracy"] > r["baseline"]
            TextDialog(self.win, "نتيجة التقييم الصادق",
                       f"<h3>دقة استرجاع نموذج مخفي: {r['accuracy'] * 100:.1f}%</h3>"
                       f"<p>خط الأساس (النمط الأشيع): {r['baseline'] * 100:.1f}% على {r['trials']} محاولة.</p>"
                       + ("<p>النموذج يتفوّق على الأساس." if better else
                          "<p><b>النموذج لا يتفوّق على خط الأساس بعد</b> — البيانات لا تكفي أو الارتباط بين النماذج ضعيف.") + "</p>").exec()
        self.win.run_task("تقييم متقاطع", lambda log: st.trainer.cross_validate(st.features, log=log), done)


# ------------------------------------------------------------------ الطلاب
class StudentsPage(Page):
    def __init__(self, win):
        super().__init__(win, "الطلاب", "ابحث ثم انقر مرتين على الطالب لفتح ملف الحالة الكامل في نافذة مستقلة")
        self.filters = ("", "", "")
        self.rows = None
        bar = QHBoxLayout(); bar.setSpacing(10)
        self.q = QLineEdit(); self.q.setPlaceholderText("🔍  اسم الطالب (كامل أو جزئي)"); self.q.returnPressed.connect(self.search)
        bar.addWidget(self.q, 1); bar.addWidget(button("بحث", self.search, True)); bar.addWidget(button("فلتر متقدم", self.adv))
        self.lay.addLayout(bar)
        self.active = QHBoxLayout(); self.lay.addLayout(self.active)
        self.count = label("", "sub"); self.lay.addWidget(self.count)
        self.table = QTableWidget(0, 5); self.table.setHorizontalHeaderLabels(["الاسم", "الصف", "المدرسة", "النماذج", "آخر نشاط"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False); self.table.verticalHeader().setDefaultSectionSize(44)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(420); self.table.cellDoubleClicked.connect(lambda r, c: self.open_row(r))
        self.lay.addWidget(self.table, 1)
        self.lay.addWidget(button("فتح الملف المحدد", lambda: self.open_row(self.table.currentRow()), True))

    def on_data(self): self.search()

    def adv(self):
        d = FilterDialog(self.win, self.filters)
        if d.exec():
            self.filters = d.values(); self.search()

    def search(self):
        st = self.win.store
        if st.audit is None:
            return
        sch, gr, iss = self.filters
        self.rows = CP.search(st, self.q.text(), sch, gr, iss)
        while self.active.count():
            w = self.active.takeAt(0).widget()
            if w:
                w.deleteLater()
        act = [x for x in (f"المدرسة: {sch}" if sch else "", f"الصف: {gr}" if gr else "", f"المشكلة: {iss}" if iss else "") if x]
        if act:
            self.active.addWidget(chips_box(act, "info"))
        self.count.setText(f"{len(self.rows)} طالباً" if len(self.rows) else "لا نتائج — اسحب البيانات أو غيّر البحث")
        self.table.setRowCount(len(self.rows))
        for i, r in enumerate(self.rows.itertuples()):
            vals = (r.name, r.grade, r.school, "" if pd.isna(r.n_models) else int(r.n_models),
                    "" if pd.isna(r.last) else r.last.strftime("%Y-%m-%d"))
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem(str(v)))

    def open_row(self, r):
        if self.rows is None or r < 0 or r >= len(self.rows):
            return self.win.toast("اختر طالباً من القائمة")
        self.win.open_profile(self.rows.iloc[r]["skey"])


# ------------------------------------------------------------------ المختبر البحثي
class ResearchPage(Page):
    def __init__(self, win):
        super().__init__(win, "المختبر البحثي", "اختر التخصص والنموذج المحوري ثم حلّل؛ تُفتح كل نتيجة في نافذة مستقلة")
        self.res = None
        c = Card("إعداد الدراسة")
        r = QHBoxLayout(); r.setSpacing(10)
        self.lens = QComboBox(); self.lens.addItems(LENSES.keys()); self.lens.currentIndexChanged.connect(self.fill)
        self.model = QComboBox(); self.method = QComboBox(); self.method.addItems(["spearman", "pearson"])
        for t, w in (("التخصص", self.lens), ("النموذج المحوري", self.model), ("الارتباط", self.method)):
            r.addWidget(label(t, "sub")); r.addWidget(w, 1)
        r.addWidget(button("تحليل", self.analyze, True)); c.lay.addLayout(r); self.lay.addWidget(c)
        self.grid_host = QWidget(); self.grid = QGridLayout(self.grid_host); self.grid.setSpacing(14)
        self.lay.addWidget(self.grid_host)
        t = Card("أدوات الباحث"); rr = QHBoxLayout()
        rr.addWidget(button("تصدير بيانات مُعمّاة (CSV)", self.export_anon)); rr.addWidget(button("منشئ نموذج جديد", lambda: BuilderDialog(win).exec()))
        rr.addStretch(); t.lay.addLayout(rr); self.lay.addWidget(t); self.lay.addStretch()
        self.fill()

    def fill(self):
        self.model.clear()
        for m in LENSES[self.lens.currentText()]:
            self.model.addItem(f"{m} — {CATALOG[m]['title']}", m)

    def analyze(self):
        st = self.win.store
        if st.audit is None or not len(st.clean):
            return self.win.toast("لا توجد بيانات — ابدأ من صفحة البيانات")
        c, mid, lens, method = st.clean, self.model.currentData(), LENSES[self.lens.currentText()], self.method.currentText()
        d, n = RL.type_distribution(c, mid)
        t = RL.time_summary(c, lens); t = t[t["n"] > 0]
        r, p, nn = RL.correlations(c, lens, method)
        ia, alpha, na = RL.item_analysis(c, mid)
        sig = [(b, r.loc[mid, b], p.loc[mid, b], nn.loc[mid, b]) for b in r.columns
               if mid in r.index and b != mid and not np.isnan(r.loc[mid, b]) and p.loc[mid, b] < .05]
        self.res = dict(mid=mid, d=d, n=n, t=t, r=r, p=p, nn=nn, ia=ia, alpha=alpha, na=na, sig=sig, method=method)
        top = d.sort_values("pct", ascending=False).iloc[0] if n else None
        cards = [
            ("توزيع الأنماط", f"الأشيع: {top['type']} ({top['pct']:.0f}%) · n={n}" if top is not None else "لا بيانات", self.show_dist),
            ("زمن الاستجابة", f"{len(t)} نماذج في هذا التخصص", self.show_time),
            ("مصفوفة الارتباط", f"{r.shape[0]} × {r.shape[0]} · {len(sig)} علاقة دالة مع {mid}", self.show_corr),
            ("تحليل الأسئلة", "ألفا = " + (f"{alpha:.2f}" if not np.isnan(alpha) else "غير متاح") + f" (n={na})", self.show_items),
            ("قراءة النتائج", "صياغة أكاديمية جاهزة للمسودة", self.show_text)]
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        for i, (ttl, sub, fn) in enumerate(cards):
            cd = Card(ttl, sub); cd.add(button("عرض", fn, True)); self.grid.addWidget(cd, i // 2, i % 2)

    def _need(self):
        if self.res is None:
            self.win.toast("اضغط «تحليل» أولاً")
        return self.res

    def show_dist(self):
        R = self._need()
        if not R: return
        pal = self.win.pal; d = R["d"]; w = make_plot(pal); x = np.arange(len(d))
        w.addItem(pg.BarGraphItem(x=x, height=np.asarray(d["pct"]), width=0.6, brush=pal["accent"]))
        w.addItem(pg.ErrorBarItem(x=x, y=np.asarray(d["pct"]), top=np.asarray(d["hi"] - d["pct"]), bottom=np.asarray(d["pct"] - d["lo"]), beam=0.12, pen=pg.mkPen(pal["text"], width=2)))
        set_ticks(w, list(d["type"])); w.setLabel("left", "% من الطلاب")
        ChartDialog(self.win, f"توزيع أنماط «{CATALOG[R['mid']]['title']}»", w, f"n = {R['n']} · الخط الرأسي = فترة ثقة 95% (Wilson)").exec()

    def show_time(self):
        R = self._need()
        if not R: return
        pal = self.win.pal; t = R["t"]; w = make_plot(pal); x = np.arange(len(t))
        w.addItem(pg.BarGraphItem(x=x, height=np.asarray(t["mean"]), width=0.6, brush=pal["ok"]))
        w.addItem(pg.ErrorBarItem(x=x, y=np.asarray(t["mean"]), height=np.asarray(2 * t["sd"].fillna(0)), beam=0.12, pen=pg.mkPen(pal["text"], width=2)))
        set_ticks(w, list(t["model"])); w.setLabel("left", "ثانية / سؤال")
        ChartDialog(self.win, "متوسط زمن الاستجابة", w, "المتوسط ± الانحراف المعياري").exec()

    def show_corr(self):
        R = self._need()
        if not R: return
        r = R["r"]; pal = self.win.pal; w = make_plot(pal, min_h=420)
        if len(r) > 1:
            img = pg.ImageItem(np.nan_to_num(r.values.astype(float)).T); img.setColorMap(pg.colormap.get("CET-D1")); img.setLevels((-1, 1))
            w.addItem(img); set_ticks(w, list(r.columns)); set_ticks(w, list(r.index), "left")
        ChartDialog(self.win, f"مصفوفة الارتباط ({R['method']})", w, "أزرق = ارتباط عكسي، أحمر = طردي؛ الخلايا الفارغة = أقل من 5 طلاب مشتركين").exec()

    def show_items(self):
        R = self._need()
        if not R: return
        if R["ia"].empty:
            return self.win.toast("لا تفاصيل إجابات صالحة لهذا النموذج")
        TableDialog(self.win, f"تحليل أسئلة «{CATALOG[R['mid']]['title']}»", R["ia"], "نسبة اختيار كل بديل % · متوسط الزمن · ارتباط السؤال بالمجموع").exec()

    def show_text(self):
        R = self._need()
        if not R: return
        html = "<h3>قراءة النتائج</h3>" + ("".join("<p>" + RL.read_result(R["mid"], b, rr, pp, n, R["method"]) + "</p>" for b, rr, pp, n in R["sig"])
                                           or "<p>لا توجد علاقات دالة عند 0.05 (أو أن المشترك أقل من 5 طلاب).</p>")
        TextDialog(self.win, "قراءة النتائج", html).exec()

    def export_anon(self):
        st = self.win.store
        if st.audit is None or not len(st.clean):
            return self.win.toast("لا توجد بيانات")
        f, _ = QFileDialog.getSaveFileName(self, "حفظ", "research_anonymized.csv", "CSV (*.csv)")
        if f:
            RL.anonymize(st.clean).to_csv(f, index=False, encoding="utf-8-sig"); self.win.toast("تم التصدير مع تعمية الهوية")
