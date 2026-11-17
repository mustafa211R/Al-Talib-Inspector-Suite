"""AI-Talib Inspector Suite — نافذة رئيسية حديثة: شريط تنقل جانبي + صفحات + نوافذ منبثقة."""
import sys
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QStackedWidget, QVBoxLayout, QWidget)

import pyqtgraph as pg
from ai_talib import clinical_profiler as CP
from ai_talib.config import ensure_dirs
from ai_talib.store import Store
from ui.dialogs import LogDialog, ProfileDialog, SettingsDialog
from ui.pages import CausesPage, DataPage, HomePage, InsightsPage, IssuesPage, ResearchPage, StudentsPage
from ui.theme import PALETTES, qss
from ui.widgets import BusyDialog, label

APP_DIR = Path(__file__).resolve().parent


class Worker(QThread):
    done, failed, log = Signal(object), Signal(str), Signal(str)

    def __init__(self, fn):
        super().__init__(); self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(self.log.emit))
        except Exception as ex:
            self.failed.emit(f"{ex}\n{traceback.format_exc()}")


class Task(QObject):
    """مهمة خلفية مع نافذة انتظار؛ الفتحات دوال كائن يعيش في خيط الواجهة."""
    def __init__(self, win, title, fn, then):
        super().__init__(win)
        self.win, self.then = win, then
        self.dlg = BusyDialog(win, title)
        self.w = Worker(fn)
        self.w.log.connect(self.on_log); self.w.done.connect(self.on_done); self.w.failed.connect(self.on_fail)

    def start(self):
        self.dlg.show(); self.w.start()

    def on_log(self, s):
        self.win.log(s); self.dlg.say(s)

    def on_done(self, res):
        self.dlg.accept(); self.w.wait()
        try:
            if self.then:
                self.then(res)
        except Exception as ex:
            self.win.log(traceback.format_exc()); QMessageBox.warning(self.win, "خطأ", str(ex))
        self.win._tasks.discard(self)

    def on_fail(self, msg):
        self.dlg.accept(); self.w.wait(); self.win.log(msg)
        QMessageBox.warning(self.win, "تعذّر التنفيذ", msg.splitlines()[0])
        self.win._tasks.discard(self)


class MainWindow(QMainWindow):
    NAV = [("🏠", "الرئيسية"), ("🗂", "البيانات والنموذج"), ("👩‍🎓", "الطلاب"), ("💬", "المشكلات ونقاط الانتباه"),
           ("🧩", "الأسباب الجذرية"), ("🧠", "الاستقراء والاستدلال"), ("🔬", "المختبر البحثي")]

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.settings = QSettings("AI-Talib", "InspectorSuite")
        self.store, self._tasks, self._log = Store(), set(), []
        self.theme = self.settings.value("theme", "light")
        self.pal = PALETTES[self.theme]
        self.setWindowTitle("AI-Talib Inspector Suite"); self.resize(1280, 820); self.setMinimumSize(1000, 680)

        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        # شريط التنقل (يظهر يميناً تلقائياً في الواجهة العربية)
        side = QFrame(); side.setObjectName("sidebar"); side.setFixedWidth(230)
        sl = QVBoxLayout(side); sl.setContentsMargins(14, 22, 14, 16); sl.setSpacing(2)
        sl.addWidget(label("AI-Talib", "h1")); sl.addWidget(label("Inspector Suite", "sub")); sl.addSpacing(18)
        self.stack = QStackedWidget(); self.group = QButtonGroup(self); self.group.setExclusive(True)
        from ui.widgets import button
        self.pages = [HomePage(self), DataPage(self), StudentsPage(self), IssuesPage(self), CausesPage(self), InsightsPage(self), ResearchPage(self)]
        for i, ((ic, name), page) in enumerate(zip(self.NAV, self.pages)):
            b = button(f"{ic}   {name}"); b.setObjectName("nav"); b.setCheckable(True)
            b.clicked.connect(lambda _=False, i=i: self.go(i))
            self.group.addButton(b, i); sl.addWidget(b); self.stack.addWidget(page)
        sl.addStretch()
        for txt, fn in (("⚙   الإعدادات", self.open_settings), ("📜   سجل العمليات", lambda: LogDialog(self, self._log).exec())):
            b = button(txt); b.setObjectName("nav"); b.clicked.connect(fn); sl.addWidget(b)
        sl.addWidget(label("بيانات محلية في db/ · للإرشاد لا للعقاب", "sub", True))
        root.addWidget(side); root.addWidget(self.stack, 1)

        self.toast_lbl = QLabel("", central); self.toast_lbl.setObjectName("toast"); self.toast_lbl.hide()
        self.apply_theme(self.theme)
        self.go(0)
        try:                                           # تحميل ما هو موجود في db/ عند الفتح
            self.store.refresh(); self.refresh_all()
        except Exception as ex:
            self.log(f"تعذّر التحميل الأولي: {ex}")

    # ---------------------------------------------------------- خدمات للصفحات
    def go(self, i):
        self.stack.setCurrentIndex(i); self.group.button(i).setChecked(True)

    def log(self, s):
        self._log.append(f"[{datetime.now():%H:%M:%S}] {s}")

    def toast(self, msg, ms=2600):
        t = self.toast_lbl; t.setText(msg); t.adjustSize()
        c = self.centralWidget(); t.move((c.width() - t.width()) // 2, c.height() - t.height() - 28)
        t.raise_(); t.show(); QTimer.singleShot(ms, t.hide)

    def run_task(self, title, fn, then=None):
        t = Task(self, title, fn, then); self._tasks.add(t); t.start()

    def refresh_all(self):
        for p in self.pages:
            p.on_data()

    def open_profile(self, skey):
        P = CP.build_profile(self.store, skey)
        ProfileDialog(self, P, self.pal).exec()

    def open_settings(self):
        if SettingsDialog(self, self.settings).exec():
            self.apply_theme(self.settings.value("theme", "light"))
            self.toast("تم حفظ الإعدادات")

    def apply_theme(self, theme):
        self.theme, self.pal = theme, PALETTES[theme]
        self.app.setStyleSheet(qss(theme))
        pg.setConfigOptions(antialias=True)
        for p in self.pages:
            p.on_theme(self.pal)


def main():
    ensure_dirs()
    sys.excepthook = lambda *a: (APP_DIR.joinpath("error.log").open("a", encoding="utf-8").write(
        "".join(traceback.format_exception(*a)) + "\n"), print("".join(traceback.format_exception(*a))))
    app = QApplication(sys.argv)
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    app.setStyle("Fusion")
    f = QFont(); f.setFamilies(["Segoe UI", "Tahoma", "Arial"]); f.setPointSize(10); app.setFont(f)
    w = MainWindow(app); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
