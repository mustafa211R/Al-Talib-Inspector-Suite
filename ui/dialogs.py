"""النوافذ المنبثقة: ملف الطالب، الرسوم، الجداول، النص، الإعدادات، السجل، منشئ النماذج."""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QPageSize, QPdfWriter, QTextDocument
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QButtonGroup, QGridLayout, QPushButton, QStackedWidget, QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QHeaderView, QLineEdit,
                               QMessageBox, QPlainTextEdit, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
                               QTextBrowser, QVBoxLayout, QWidget)

from ai_talib import clinical_profiler as CP
from ai_talib import research_lab as RL
from ai_talib.insights import analyze_text
from .theme import make_plot, set_ticks
from .widgets import Card, FlowLayout, Gauge, button, chip, chips_box, label


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


class StudentListDialog(BaseDialog):
    """قائمة طلاب (مثلاً المتأثرون بسبب جذري)، والنقر المزدوج يفتح الملف."""
    def __init__(self, parent, title, skeys, note=""):
        super().__init__(parent, title, 820, 560)
        self.win = parent
        if note:
            self.root.addWidget(label(note, "sub", True))
        st = parent.store.students().set_index("skey")
        sk = [k for k in skeys if k in st.index]
        t = QTableWidget(len(sk), 3); t.setHorizontalHeaderLabels(["الاسم", "الصف", "المدرسة"]); t.setAlternatingRowColors(True)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for i, k in enumerate(sk):
            for j, c in enumerate(("name", "grade", "school")):
                t.setItem(i, j, QTableWidgetItem(str(st.loc[k, c])))
        t.cellDoubleClicked.connect(lambda r, c: parent.open_profile(sk[r]))
        self.root.addWidget(t, 1); self.root.addWidget(label("انقر مرتين على الطالب لفتح ملفه", "sub")); self.footer()


def _clear(layout):
    while layout.count():
        it = layout.takeAt(0)
        if it.widget():
            it.widget().deleteLater()
        elif it.layout():
            _clear(it.layout())


def _gauge_block(title, value, pal, caption, text, invert=False):
    w = QWidget(); v = QVBoxLayout(w); v.setSpacing(4)
    g = Gauge(caption, invert); g.set_value(value, pal); v.addWidget(g)
    v.addWidget(label(title, "h2")); v.addWidget(label(text, "sub", True))
    return w


def _tint(item, pal, kind):
    item.setBackground(QColor(pal[kind + "_bg"]))


class ProfileDialog(BaseDialog):
    """ملف الطالب: «عام» (كل الأنماط) أو «نمط محدد» (كل التفاصيل عن محور واحد)."""
    def __init__(self, parent, P, pal):
        super().__init__(parent, "ملف الحالة", 1080, 780)
        self.P, self.pal = P, pal
        head = Card(); hl = QHBoxLayout(); head.lay.addLayout(hl)
        av = label((P["name"] or "؟")[:1], "avatar"); av.setFixedSize(64, 64); av.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl.addWidget(av)
        info = QVBoxLayout(); info.addWidget(label(P["name"], "h1")); info.addWidget(label(f"{P['grade']} — {P['school']}", "sub"))
        row = QHBoxLayout(); row.addWidget(chip(f"{P['n_done']} نموذج منجز", "info"))
        if P["avg_time"]:
            row.addWidget(chip(f"{P['avg_time']:.1f} ث / سؤال", "info"))
        if P["mean_trust"] is not None:
            row.addWidget(chip(f"موثوقية {P['mean_trust'] * 100:.0f}%", "ok" if P["mean_trust"] > .7 else "warn"))
        row.addStretch(); info.addLayout(row); hl.addLayout(info, 1)
        g = Gauge(); g.set_value(None if P["risk"] is None else P["risk"] * 100, pal)
        g.setToolTip(P["risk_src"] + " — مؤشر أولي غير مُعايَر"); hl.addWidget(g)
        self.root.addWidget(head)
        # مفتاح التبديل
        tog = QHBoxLayout(); tog.setSpacing(0); self.grp = QButtonGroup(self)
        self.b_gen, self.b_spec = QPushButton("عام"), QPushButton("نمط محدد")
        for i, b in enumerate((self.b_gen, self.b_spec)):
            b.setObjectName("seg"); b.setCheckable(True); b.setCursor(Qt.CursorShape.PointingHandCursor); self.grp.addButton(b, i); tog.addWidget(b)
        self.b_gen.setToolTip("المعلومات الحالية عن جميع الأنماط بشكل عام"); self.b_spec.setToolTip("كل التفاصيل عن نمط (محور) واحد تختاره")
        self.b_gen.setChecked(True); tog.addStretch(); self.root.addLayout(tog)
        self.stack = QStackedWidget(); self.root.addWidget(self.stack, 1)
        self.stack.addWidget(self._scroll(self._general())); self.stack.addWidget(self._scroll(self._specific()))
        self.grp.idClicked.connect(self.stack.setCurrentIndex)
        if not P["detail"]:
            self.b_spec.setEnabled(False); self.b_spec.setToolTip("لا توجد نتائج نماذج تفصيلية لهذا الطالب (ملف الطلاب فقط)")
        self.footer(button("PDF سري للمرشد", lambda: self.export(False)), button("PDF مُعمّى", lambda: self.export(True)))

    @staticmethod
    def _scroll(body):
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(body); return sc

    def _page(self):
        body = QWidget(); body.setObjectName("page"); bl = QVBoxLayout(body); bl.setSpacing(14); bl.setContentsMargins(0, 6, 6, 0)
        return body, bl

    # ------------------------------------------------------------ عام
    def _general(self):
        P, pal = self.P, self.pal
        body, bl = self._page()
        if P["safety"]:
            from PySide6.QtWidgets import QFrame
            b = QFrame(); b.setObjectName("banner"); v = QVBoxLayout(b); v.setContentsMargins(18, 14, 18, 14)
            v.addWidget(label("⚠ تنبيه أمان — يتطلب تدخلاً بشرياً اليوم", "h2"))
            v.addWidget(label("ورد في نص الطالب ما يستدعي اهتماماً فورياً (" + "، ".join(P["safety"]) + "). تواصل معه مباشرة وبهدوء.", None, True)); bl.addWidget(b)
        if P["models"]:
            c = Card("البطاقة المعرفية والسلوكية", "الدرجة من −1 (يحتاج انتباه) إلى +1 (نقطة قوة)")
            t = QTableWidget(len(P["models"]), 4); t.setHorizontalHeaderLabels(["المحور", "النمط الغالب", "زمن/سؤال", "الموثوقية"])
            t.verticalHeader().setVisible(False); t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); t.setAlternatingRowColors(True)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            for i, m in enumerate(P["models"]):
                for j, v in enumerate((f"{m['id']} — {m['title']}", m["type"] + (" ⚡" if m["impulsive"] else ""), f"{m['avg']:.1f}", m["rel"])):
                    t.setItem(i, j, QTableWidgetItem(str(v)))
            t.setFixedHeight(min(len(P["models"]), 8) * 36 + 44); c.add(t)
            pl = make_plot(pal, min_h=170); x = np.arange(len(P["models"]))
            br = [pal["ok"] if m["score"] > .2 else pal["bad"] if m["score"] < -.2 else pal["warn"] for m in P["models"]]
            pl.addItem(pg.BarGraphItem(x=x, height=[m["score"] for m in P["models"]], width=0.6, brushes=br))
            set_ticks(pl, [m["id"] for m in P["models"]]); pl.setYRange(-1, 1); c.add(pl); bl.addWidget(c)
        b = P.get("behavior")
        if b:
            bl.addWidget(self._behavior_card(b))
        c = Card("نقاط القوة"); c.add(chips_box(P["strengths"], "ok")); bl.addWidget(c)
        c = Card("ما يحتاج انتباهاً"); c.add(chips_box(P["attention"], "warn"))
        if P["causes"]:
            c.add(label("الجذور المحتملة", "sub"))
            for t_ in P["causes"]:
                c.add(label("• " + t_, None, True))
        bl.addWidget(c)
        hn = P["honesty"]; c = Card("الصدق والتناقض")
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
        return body

    def _behavior_card(self, b):
        pal = self.pal
        c = Card("الموثوقية والاندفاع وتجميل الصورة",
                 "ثلاثة مؤشرات تخبرك بمدى الوثوق بإجابات الطالب. تجميل الصورة مؤشر لإدارة الانطباع وليس كاشف كذب.")
        row = QHBoxLayout(); row.setSpacing(16)
        row.addWidget(_gauge_block("تجميل الصورة (الكذب المحتمل)", b["faking_mean"], pal, "المتوسط",
                                   f"المستوى: {CP.faking_level(b['faking_mean'])} · الأعلى {b['faking_max']:.0f}% · "
                                   f"{b['faking_hi']} نماذج ≥ 40%. يرتفع حين تكون إجاباته المتأنية أفضل من العفوية "
                                   f"({b['slow_better']} نماذج بهذا النمط)."), 1)
        row.addWidget(_gauge_block("الاندفاع", b["imp_ratio"] * 100, pal, "من الإجابات",
                                   f"{b['imp_ratio'] * 100:.0f}% من إجاباته اندفاعية، و{b['imp_models']} نماذج دون عتبة 1.5 ث/سؤال. "
                                   "النماذج المندفعة لا تُعتمد في تدريب التشخيص المعرفي."), 1)
        row.addWidget(_gauge_block("الموثوقية", b["trust_mean"], pal, "نقاط الثقة",
                                   "= 100% − 50% للاندفاع − 20% لنص عشوائي − 0.3 × تجميل الصورة، لكل نموذج ثم المتوسط.", invert=True), 1)
        c.lay.addLayout(row)
        per = b["per"]
        t = QTableWidget(len(per), 6)
        t.setHorizontalHeaderLabels(["المحور", "تجميل الصورة", "الاندفاع (اندفاع/متزن/بطيء)", "زمن/سؤال", "الموثوقية", "الخصومات"])
        t.verticalHeader().setVisible(False); t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for i, x in enumerate(per):
            vals = (f"{x['id']} — {x['title']}", f"{x['faking']:.0f}% ({CP.faking_level(x['faking'])})",
                    f"{x['imp_n']} / {x['bal_n']} / {x['slow_n']}", f"{x['avg']:.1f}", f"{x['reliability']} · {x['trust']:.0f}%",
                    "، ".join(f"{n} −{v:g}%" for n, v in x["penalties"]) or "لا خصم")
            for j, v in enumerate(vals):
                it = QTableWidgetItem(str(v)); t.setItem(i, j, it)
            if x["faking"] >= 40: _tint(t.item(i, 1), pal, "warn")
            if x["impulsive"]: _tint(t.item(i, 2), pal, "bad")
            if x["trust"] < 60: _tint(t.item(i, 4), pal, "bad")
        t.setFixedHeight(min(len(per), 8) * 36 + 44); c.add(t)
        return c

    # ------------------------------------------------------------ نمط محدد
    def _specific(self):
        body, bl = self._page()
        c = Card("اختر النمط (المحور)", "انقر على أحد الأنماط لعرض كل ما يخصه: الأسئلة والمشاكل والموثوقية والاندفاع")
        host = QWidget(); fl = FlowLayout(host); self.pick = QButtonGroup(self); self.pick.setExclusive(True)
        for i, (mid, d) in enumerate(self.P["detail"].items()):
            b = QPushButton(f"{mid} · {d['title']}: {d['type']}"); b.setObjectName("pick"); b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor); self.pick.addButton(b, i); fl.addWidget(b)
        c.add(host); bl.addWidget(c)
        self.spec_lay = QVBoxLayout(); self.spec_lay.setSpacing(14); bl.addLayout(self.spec_lay); bl.addStretch()
        self.keys = list(self.P["detail"])
        self.pick.idClicked.connect(lambda i: self.show_model(self.keys[i]))
        if self.keys:
            self.pick.button(0).setChecked(True); self.show_model(self.keys[0])
        return body

    def show_model(self, mid):
        d, pal = self.P["detail"][mid], self.pal
        L = self.spec_lay; _clear(L)
        kind = "ok" if d["level"] == 1 else "warn" if d["level"] == -1 else "info"
        c = Card(f"{d['title']} — {d['type']}")
        r = QHBoxLayout(); r.addWidget(chip(d["type"], kind))
        r.addWidget(chip(f"الموثوقية: {d['reliability']}", "info"))
        if d["impulsive"]:
            r.addWidget(chip("⚡ اندفاع زمني", "bad"))
        if pd.notna(d["ts"]):
            r.addWidget(chip(str(d["ts"])[:16], "info"))
        r.addStretch(); c.lay.addLayout(r); L.addWidget(c)
        # التوزيع
        c = Card("توزيع الأنماط في هذا المحور"); pl = make_plot(pal, min_h=200); x = np.arange(len(d["types"]))
        pct = [100 * v for v in (d["dist"] or [0] * 4)]
        pl.addItem(pg.BarGraphItem(x=x, height=pct, width=0.6, brushes=[pal["accent"] if i == d["ti"] else pal["border"] for i in x]))
        set_ticks(pl, d["types"]); pl.setLabel("left", "%"); c.add(pl); L.addWidget(c)
        # السلوك
        imp, bal, slow = d["lat"]; tot = imp + bal + slow
        c = Card("السلوك أثناء الاختبار"); row = QHBoxLayout(); row.setSpacing(16)
        row.addWidget(_gauge_block("تجميل الصورة", d["faking"], pal, CP.faking_level(d["faking"]), f"سريع/متأنٍّ: {d['dual']}"), 1)
        row.addWidget(_gauge_block("الاندفاع", 100 * imp / tot if tot else 0, pal, "من إجابات النموذج",
                                   f"اندفاع {imp} · متزن {bal} · بطيء {slow} — متوسط {d['avg']:.1f} ث، المجموع {d['total']:.0f} ث"), 1)
        row.addWidget(_gauge_block("الموثوقية", d["trust"], pal, d["reliability"], "نقاط الثقة بعد خصم الاندفاع والنص العشوائي وتجميل الصورة", True), 1)
        c.lay.addLayout(row)
        if d["alerts"]:
            c.add(label("تنبيهات: " + d["alerts"], "sub", True))
        L.addWidget(c)
        # الأسئلة
        qs = d["questions"]
        c = Card("الأسئلة وإجابات الطالب", "الزمن بالثواني · الصف الأحمر = إجابة اندفاعية")
        if qs:
            t = QTableWidget(len(qs), 6); t.setHorizontalHeaderLabels(["#", "السؤال", "إجابته", "النمط", "الزمن", "الحالة"])
            t.verticalHeader().setVisible(False); t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); t.setWordWrap(True)
            h = t.horizontalHeader(); h.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch); h.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            for i, q in enumerate(qs):
                for j, v in enumerate((q["n"], q["q"] or f"({q['kind']})", q["answer"], q["type"], f"{q['sec']:.1f}", q["speed"])):
                    t.setItem(i, j, QTableWidgetItem(str(v)))
                if "اندفاع" in q["speed"]:
                    for j in range(6): _tint(t.item(i, j), pal, "bad")
            t.resizeRowsToContents(); t.setFixedHeight(min(sum(t.rowHeight(i) for i in range(len(qs))) + 44, 420)); c.add(t)
        else:
            c.add(label("لا توجد تفاصيل إجابات محفوظة لهذا النموذج.", "sub"))
        L.addWidget(c)
        # المشاكل
        c = Card("المشاكل التي كتبها الطالب")
        if d["problems"]:
            c.add(label("«" + d["problems"] + "»", None, True))
            themes, risk = analyze_text(d["problems"])
            if d["gibberish"]:
                c.add(chip("نص عشوائي — مستبعد من التحليل", "bad"))
            if themes:
                c.add(chips_box(themes, "info"))
            if risk:
                c.add(chips_box(["⚠ " + x for x in risk], "bad"))
        else:
            c.add(label("لم يكتب شيئاً في هذا النموذج.", "sub"))
        L.addWidget(c)
        for ttl, txt in (("النتيجة والتحليل", d["result"]), ("التوقع", d["forecast"]), ("توصية هذا النمط", d["plan"])):
            if txt:
                c = Card(ttl); c.add(label(txt, None, True)); L.addWidget(c)

    def export(self, anon):
        f, _ = QFileDialog.getSaveFileName(self, "حفظ PDF", "case_report.pdf", "PDF (*.pdf)")
        if not f:
            return
        doc = QTextDocument(); doc.setHtml(CP.to_html(self.P, anonymous=anon))
        w = QPdfWriter(f); w.setPageSize(QPageSize(QPageSize.PageSizeId.A4)); doc.print_(w)
