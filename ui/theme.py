"""الثيمات (فاتح/داكن) وقالب QSS بأسلوب Fluent الحديث."""
import pyqtgraph as pg
from PySide6.QtCore import Qt

PALETTES = {
    "light": dict(bg="#f3f5f9", card="#ffffff", text="#1b2330", sub="#6b7686", border="#e3e7ee", accent="#2f6df6",
                  accent_hover="#255ae0", accent_soft="#e6eeff", accent_text="#ffffff", nav="#ffffff", hover="#eef2f9",
                  input="#f7f9fc", ok="#13803d", ok_bg="#dcf5e5", warn="#a45b00", warn_bg="#fdeccc",
                  bad="#b42318", bad_bg="#fde1de", info="#1d4ed8", info_bg="#e0e9ff"),
    "dark": dict(bg="#0e1116", card="#171b22", text="#e8ecf2", sub="#8b95a5", border="#262c36", accent="#4c8dff",
                 accent_hover="#6aa0ff", accent_soft="#1c2a44", accent_text="#ffffff", nav="#12151b", hover="#1e242e",
                 input="#1c212a", ok="#4ade80", ok_bg="#14301f", warn="#fbbf24", warn_bg="#3a2c0c",
                 bad="#f87171", bad_bg="#3d1717", info="#7aa7ff", info_bg="#1a2a4a"),
}

_QSS = """
* { font-family: "Segoe UI","Tahoma","Arial"; font-size: 10.5pt; color: @text; }
QMainWindow, QDialog, QWidget#page { background: @bg; }
QScrollArea { border: none; background: transparent; }
QFrame#card { background: @card; border: 1px solid @border; border-radius: 14px; }
QFrame#card QLabel, QFrame#card QCheckBox { background: transparent; }
QFrame#sidebar { background: @nav; border-left: 1px solid @border; }
QPushButton { background: @card; border: 1px solid @border; border-radius: 10px; padding: 8px 18px; }
QPushButton:hover { background: @hover; }
QPushButton:disabled { color: @sub; }
QPushButton[primary="true"] { background: @accent; color: @accent_text; border: none; font-weight: 600; }
QPushButton[primary="true"]:hover { background: @accent_hover; }
QPushButton#nav { border: none; background: transparent; text-align: right; padding: 12px 16px; font-size: 11pt; }
QPushButton#nav:hover { background: @hover; }
QPushButton#nav:checked { background: @accent_soft; color: @accent; font-weight: 700; }
QPushButton#seg { border: 1px solid @border; padding: 9px 28px; font-weight: 600; background: @card; }
QPushButton#seg:checked { background: @accent; color: @accent_text; border: 1px solid @accent; }
QPushButton#pick { border: 1px solid @border; border-radius: 16px; padding: 7px 16px; background: @card; }
QPushButton#pick:checked { background: @accent_soft; color: @accent; border: 1px solid @accent; font-weight: 700; }
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextBrowser {
  background: @input; border: 1px solid @border; border-radius: 10px; padding: 7px 10px; selection-background-color: @accent; }
QTextBrowser { background: transparent; border: none; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border: 1px solid @accent; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox QAbstractItemView { background: @card; border: 1px solid @border; selection-background-color: @accent_soft; selection-color: @text; }
QTableWidget { background: @card; border: 1px solid @border; border-radius: 12px; gridline-color: transparent;
  alternate-background-color: @input; selection-background-color: @accent_soft; selection-color: @text; }
QTableWidget::item { padding: 6px; border: none; }
QHeaderView::section { background: @card; color: @sub; border: none; border-bottom: 1px solid @border; padding: 10px; font-weight: 600; }
QScrollBar:vertical { width: 10px; background: transparent; margin: 2px; }
QScrollBar::handle:vertical { background: @border; border-radius: 5px; min-height: 30px; }
QScrollBar:horizontal { height: 10px; background: transparent; margin: 2px; }
QScrollBar::handle:horizontal { background: @border; border-radius: 5px; min-width: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QProgressBar { border: none; background: @border; border-radius: 4px; max-height: 8px; text-align: center; }
QProgressBar::chunk { background: @accent; border-radius: 4px; }
QLabel#h1 { font-size: 20pt; font-weight: 700; }
QLabel#h2 { font-size: 13pt; font-weight: 700; }
QLabel#sub { color: @sub; }
QLabel#stat { font-size: 24pt; font-weight: 700; }
QLabel#avatar { background: @accent; color: white; border-radius: 32px; font-size: 22pt; font-weight: 700; }
QLabel#toast { background: @text; color: @bg; border-radius: 12px; padding: 11px 22px; font-weight: 600; }
QLabel[chip="ok"] { background: @ok_bg; color: @ok; border-radius: 13px; padding: 5px 13px; }
QLabel[chip="warn"] { background: @warn_bg; color: @warn; border-radius: 13px; padding: 5px 13px; }
QLabel[chip="bad"] { background: @bad_bg; color: @bad; border-radius: 13px; padding: 5px 13px; }
QLabel[chip="info"] { background: @info_bg; color: @info; border-radius: 13px; padding: 5px 13px; }
QFrame#banner { background: @bad_bg; border: 1px solid @bad; border-radius: 12px; }
QToolTip { background: @card; color: @text; border: 1px solid @border; padding: 6px; }
"""


def qss(theme: str) -> str:
    s = _QSS
    for k in sorted(PALETTES[theme], key=len, reverse=True):
        s = s.replace("@" + k, PALETTES[theme][k])
    return s


def make_plot(pal, title="", min_h=240):
    w = pg.PlotWidget(background=pal["card"])
    w.setLayoutDirection(Qt.LayoutDirection.LeftToRight)       # المحاور لا تنعكس مع الواجهة العربية
    pi = w.getPlotItem()
    for ax in ("left", "bottom"):
        a = pi.getAxis(ax)
        a.setPen(pg.mkPen(pal["border"])); a.setTextPen(pg.mkPen(pal["sub"]))
    if title:
        pi.setTitle(title, color=pal["text"], size="12pt")
    pi.showGrid(y=True, alpha=0.15)
    w.setMenuEnabled(False); w.setMinimumHeight(min_h)
    return w


def set_ticks(plot, labels, axis="bottom"):
    plot.getAxis(axis).setTicks([[(i, str(l)) for i, l in enumerate(labels)]])
