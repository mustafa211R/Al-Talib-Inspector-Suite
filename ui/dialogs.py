"""النوافذ المنبثقة: ملف الطالب، الرسوم، الجداول، النص، الإعدادات، السجل، منشئ النماذج."""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QPageSize, QPdfWriter, QTextDocument
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QHeaderView, QLineEdit,
                               QMessageBox, QPlainTextEdit, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
                               QTextBrowser, QVBoxLayout, QWidget)

from ai_talib import clinical_profiler as CP
from ai_talib import research_lab as RL
from .theme import make_plot, set_ticks
from .widgets import Card, Gauge, button, chip, chips_box, label


class BaseDialog(QDialog):
    def __init__(self, parent, title, w=900, h=620):
        super().__init__(parent)
        self.setWindowTitle(title); self.resize(w, h)
        self.root = QVBoxLayout(self); self.root.setContentsMargins(24, 20, 24, 20); self.root.setSpacing(14)
        self.root.addWidget(label(title, "h1"))

    def footer(self, *btns):
        row = QHBoxLayout(); row.addStretch()
        for b in btns:
            row.addWidget(b)
        row.addWidget(button("إغلاق", self.accept))
        self.root.addLayout(row)


class ChartDialog(BaseDialog):
    def __init__(self, parent, title, plot_widget, note=""):
        super().__init__(parent, title, 980, 640)
        if note:
            self.root.addWidget(label(note, "sub", True))
        c = Card(); c.add(plot_widget); self.root.addWidget(c, 1)
        self.plot = plot_widget
        self.footer(button("حفظ كصورة PNG", self.save_png))

    def save_png(self):
        import pyqtgraph.exporters as ex
        f, _ = QFileDialog.getSaveFileName(self, "حفظ الصورة", "chart.png", "PNG (*.png)")
        if f:
            e = ex.ImageExporter(self.plot.plotItem); e.parameters()["width"] = 1800; e.export(f)


class TableDialog(BaseDialog):
    def __init__(self, parent, title, df: pd.DataFrame, note=""):
        super().__init__(parent, title, 1000, 560)
        if note:
            self.root.addWidget(label(note, "sub", True))
        t = QTableWidget(len(df), len(df.columns)); t.setAlternatingRowColors(True)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); t.verticalHeader().setVisible(False)
        t.setHorizontalHeaderLabels([str(c) for c in df.columns])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for i, row in enumerate(df.itertuples(index=False)):
            for j, v in enumerate(row):
                t.setItem(i, j, QTableWidgetItem("" if pd.isna(v) else (f"{v:.2f}" if isinstance(v, float) else str(v))))
        self.root.addWidget(t, 1)
        self.footer(button("حفظ CSV", lambda: self.save(df)))

    def save(self, df):
        f, _ = QFileDialog.getSaveFileName(self, "حفظ", "table.csv", "CSV (*.csv)")
        if f:
            df.to_csv(f, index=False, encoding="utf-8-sig")


class TextDialog(BaseDialog):
    def __init__(self, parent, title, html):
        super().__init__(parent, title, 760, 520)
        c = Card(); tb = QTextBrowser(); tb.setHtml(html); tb.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        c.add(tb); self.root.addWidget(c, 1); self.footer()


class LogDialog(BaseDialog):
    def __init__(self, parent, lines):
        super().__init__(parent, "سجل العمليات", 780, 480)
        t = QPlainTextEdit(); t.setReadOnly(True); t.setPlainText("\n".join(lines)); self.root.addWidget(t, 1); self.footer()


class SettingsDialog(BaseDialog):
    def __init__(self, parent, settings):
        super().__init__(parent, "الإعدادات", 520, 360)
        self.s = settings
        form = QFormLayout(); form.setSpacing(14)
        self.theme = QComboBox(); self.theme.addItem("فاتح", "light"); self.theme.addItem("داكن", "dark")
        self.theme.setCurrentIndex(0 if settings.value("theme", "light") == "light" else 1)
        self.epochs = QSpinBox(); self.epochs.setRange(5, 500); self.epochs.setValue(int(settings.value("epochs", 60)))
        form.addRow("المظهر", self.theme); form.addRow("دورات التدريب الافتراضية", self.epochs)
        self.root.addLayout(form); self.root.addStretch()
        self.footer(button("حفظ", self.save, True))

    def save(self):
        self.s.setValue("theme", self.theme.currentData()); self.s.setValue("epochs", self.epochs.value())
        self.accept()


class FilterDialog(BaseDialog):
    def __init__(self, parent, values):
        super().__init__(parent, "فلتر متقدم", 520, 340)
        form = QFormLayout(); form.setSpacing(14)
        self.school, self.grade, self.issue = QLineEdit(values[0]), QLineEdit(values[1]), QLineEdit(values[2])
        self.issue.setPlaceholderText("مثال: كسول بسبب الملل")
        form.addRow("المدرسة", self.school); form.addRow("الصف", self.grade); form.addRow("نوع المشكلة", self.issue)
        self.root.addLayout(form); self.root.addStretch()
        self.footer(button("تطبيق", self.accept, True))

    def values(self):
        return self.school.text(), self.grade.text(), self.issue.text()


class BuilderDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "منشئ نموذج سيكومتري جديد", 760, 640)
        form = QFormLayout(); form.setSpacing(12)
        self.id, self.title, self.icon = QLineEdit("AA"), QLineEdit(), QLineEdit("🧪")
        self.types = QPlainTextEdit(); self.types.setPlaceholderText("4 أسطر — كل سطر: اسم النمط ; وصفه ; توصية")
        self.qs = QPlainTextEdit(); self.qs.setPlaceholderText("سطر لكل سؤال: نص السؤال ; بديل1 ; بديل2 ; بديل3 ; بديل4 (بترتيب الأنماط)")
        form.addRow("رمز النموذج", self.id); form.addRow("العنوان", self.title); form.addRow("أيقونة", self.icon)
        form.addRow("الأنماط", self.types); form.addRow("الأسئلة", self.qs)
        self.root.addLayout(form, 1); self.footer(button("حفظ ملف التعريف (JS)", self.save, True))

    def save(self):
        try:
            types = [tuple(x.strip() for x in l.split(";")) for l in self.types.toPlainText().splitlines() if l.strip()]
            qs = [(p[0], list(p[1:])) for p in (tuple(x.strip() for x in l.split(";")) for l in self.qs.toPlainText().splitlines() if l.strip())]
            if any(len(t) != 3 for t in types):
                raise ValueError("كل نمط: اسم ; وصف ; توصية")
            js = RL.build_model_js(self.id.text().strip(), self.title.text().strip(), self.icon.text().strip(), types, qs)
            f, _ = QFileDialog.getSaveFileName(self, "حفظ", f"model_{self.id.text()}.js", "JS (*.js)")
            if f:
                open(f, "w", encoding="utf-8").write(js); self.accept()
        except Exception as ex:
            QMessageBox.warning(self, "خطأ", str(ex))


class ProfileDialog(BaseDialog):
    """ملف الطالب كنافذة منبثقة بطاقات بدل شاشة مقسّمة."""
    def __init__(self, parent, P, pal):
        super().__init__(parent, "ملف الحالة", 1040, 760)
        self.P = P
        self.root.itemAt(0).widget().setText("ملف الحالة")
        # رأس: صورة رمزية + بيانات + مقياس الخطر
        head = Card(); hl = QHBoxLayout(); head.lay.addLayout(hl)
        av = label((P["name"] or "؟")[:1], "avatar"); av.setFixedSize(64, 64); av.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl.addWidget(av)
        info = QVBoxLayout(); info.addWidget(label(P["name"], "h1"))
        info.addWidget(label(f"{P['grade']} — {P['school']}", "sub"))
        row = QHBoxLayout()
        row.addWidget(chip(f"{P['n_done']} نموذج منجز", "info"))
        if P["avg_time"]:
            row.addWidget(chip(f"{P['avg_time']:.1f} ث / سؤال", "info"))
        if P["mean_trust"] is not None:
            row.addWidget(chip(f"موثوقية {P['mean_trust'] * 100:.0f}%", "ok" if P["mean_trust"] > .7 else "warn"))
        row.addStretch(); info.addLayout(row)
        hl.addLayout(info, 1)
        g = Gauge(); g.set_value(None if P["risk"] is None else P["risk"] * 100, pal); g.setToolTip(P["risk_src"] + " — مؤشر أولي غير مُعايَر")
        hl.addWidget(g)
        # المحتوى القابل للتمرير
        sc = QScrollArea(); sc.setWidgetResizable(True)
        body = QWidget(); body.setObjectName("page"); bl = QVBoxLayout(body); bl.setSpacing(14); bl.setContentsMargins(0, 0, 6, 0)
        if P["safety"]:
            b = QFrame(); b.setObjectName("banner"); v = QVBoxLayout(b); v.setContentsMargins(18, 14, 18, 14)
            v.addWidget(label("⚠ تنبيه أمان — يتطلب تدخلاً بشرياً اليوم", "h2"))
            v.addWidget(label("ورد في نص الطالب ما يستدعي اهتماماً فورياً (" + "، ".join(P["safety"]) +
                              "). تواصل معه مباشرة وبهدوء.", None, True)); bl.addWidget(b)
        self.root.addWidget(head); self.root.addWidget(sc, 1)
        sc.setWidget(body)
        # البطاقة المعرفية: جدول + رسم درجات
        if P["models"]:
            c = Card("البطاقة المعرفية والسلوكية", "الدرجة من −1 (يحتاج انتباه) إلى +1 (نقطة قوة)")
            t = QTableWidget(len(P["models"]), 4); t.setHorizontalHeaderLabels(["المحور", "النمط الغالب", "زمن/سؤال", "الموثوقية"])
            t.verticalHeader().setVisible(False); t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            t.setAlternatingRowColors(True); t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            for i, m in enumerate(P["models"]):
                for j, v in enumerate((f"{m['id']} — {m['title']}", m["type"] + (" ⚡" if m["impulsive"] else ""),
                                       f"{m['avg']:.1f}", m["rel"])):
                    t.setItem(i, j, QTableWidgetItem(str(v)))
            t.setFixedHeight(min(len(P["models"]), 8) * 36 + 44); c.add(t)
            pl = make_plot(pal, min_h=170); x = np.arange(len(P["models"]))
            br = [pal["ok"] if m["score"] > .2 else pal["bad"] if m["score"] < -.2 else pal["warn"] for m in P["models"]]
            pl.addItem(pg.BarGraphItem(x=x, height=[m["score"] for m in P["models"]], width=0.6, brushes=br))
            set_ticks(pl, [m["id"] for m in P["models"]]); pl.setYRange(-1, 1); c.add(pl); bl.addWidget(c)
        c = Card("نقاط القوة"); c.add(chips_box(P["strengths"], "ok")); bl.addWidget(c)
        c = Card("ما يحتاج انتباهاً"); c.add(chips_box(P["attention"], "warn"))
        if P["causes"]:
            c.add(label("الجذور المحتملة", "sub"))
            for t_ in P["causes"]:
                c.add(label("• " + t_, None, True))
        bl.addWidget(c)
        hn = P["honesty"]
        c = Card("الصدق والتناقض")
        for n in hn["notes"] or ["لا مؤشرات تناقض لافتة."]:
            c.add(label("• " + n, None, True))
        if hn["problems"]:
            c.add(label("ما كتبه الطالب: «" + hn["problems"] + "»", "sub", True))
        if hn["themes"]:
            c.add(chips_box(hn["themes"], "info"))
        bl.addWidget(c)
        c = Card("التنبؤ المستقبلي", "مؤشر أولي وليس نتيجة مُعايَرة بحالات فعلية")
        for t_ in P["forecast"]:
            c.add(label("• " + t_, None, True))
        bl.addWidget(c)
        c = Card("خطة التدخل"); cols = QHBoxLayout(); cols.setSpacing(12)
        for ttl, key in (("رسالة للمعلم", "teacher"), ("رسالة للمرشد", "counselor"), ("رسالة لولي الأمر", "parent")):
            cc = Card(ttl)
            for t_ in P["plan"][key]:
                cc.add(label("• " + t_, None, True))
            cols.addWidget(cc, 1)
        c.lay.addLayout(cols); bl.addWidget(c)
        if P["twins"]:
            c = Card("توائم سلوكية", "طلاب بنمط مشابه — راجع ما نجح معهم")
            c.add(chips_box([f"{t['name']} · {t['sim'] * 100:.0f}% ({t['common']} نماذج)" for t in P["twins"]], "info")); bl.addWidget(c)
        bl.addWidget(label(CP.DISCLAIMER, "sub", True)); bl.addStretch()
        self.footer(button("PDF سري للمرشد", lambda: self.export(False)), button("PDF مُعمّى", lambda: self.export(True)))

    def export(self, anon):
        f, _ = QFileDialog.getSaveFileName(self, "حفظ PDF", "case_report.pdf", "PDF (*.pdf)")
        if not f:
            return
        doc = QTextDocument(); doc.setHtml(CP.to_html(self.P, anonymous=anon))
        w = QPdfWriter(f); w.setPageSize(QPageSize(QPageSize.PageSizeId.A4)); doc.print_(w)
