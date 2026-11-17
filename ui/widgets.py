"""عناصر واجهة قابلة لإعادة الاستخدام: بطاقة، شريحة، مقياس، تخطيط متدفق، نافذة انتظار."""
from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout, QLabel, QLayout, QProgressBar, QPushButton,
                               QVBoxLayout, QWidget)


def label(text="", obj=None, wrap=False):
    l = QLabel(text)
    if obj:
        l.setObjectName(obj)
    l.setWordWrap(wrap)
    return l


def button(text, fn=None, primary=False):
    b = QPushButton(text)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    if primary:
        b.setProperty("primary", True)
    if fn:
        b.clicked.connect(fn)
    return b


class Card(QFrame):
    def __init__(self, title="", sub=""):
        super().__init__()
        self.setObjectName("card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(20, 18, 20, 18); self.lay.setSpacing(12)
        if title:
            self.lay.addWidget(label(title, "h2"))
        if sub:
            self.lay.addWidget(label(sub, "sub", True))

    def add(self, w):
        self.lay.addWidget(w)
        return w


def chip(text, kind="info"):
    l = QLabel(text)
    l.setProperty("chip", kind)
    return l


class FlowLayout(QLayout):
    """يرتّب الشرائح في أسطر متكسّرة ويحترم اتجاه RTL."""
    def __init__(self, parent=None, spacing=8):
        super().__init__(parent)
        self._items, self._sp = [], spacing

    def addItem(self, it): self._items.append(it)
    def count(self): return len(self._items)
    def itemAt(self, i): return self._items[i] if 0 <= i < len(self._items) else None
    def takeAt(self, i): return self._items.pop(i) if 0 <= i < len(self._items) else None
    def expandingDirections(self): return Qt.Orientation(0)
    def hasHeightForWidth(self): return True
    def heightForWidth(self, w): return self._do(QRect(0, 0, w, 0), True)
    def setGeometry(self, r): super().setGeometry(r); self._do(r, False)
    def sizeHint(self): return self.minimumSize()

    def minimumSize(self):
        s = QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do(self, rect, test):
        rtl = QApplication.layoutDirection() == Qt.LayoutDirection.RightToLeft
        off, y, lh = 0, rect.y(), 0
        for it in self._items:
            sz = it.sizeHint()
            if off + sz.width() > rect.width() and lh > 0:
                off, y, lh = 0, y + lh + self._sp, 0
            if not test:
                x = rect.right() + 1 - off - sz.width() if rtl else rect.x() + off
                it.setGeometry(QRect(QPoint(x, y), sz))
            off += sz.width() + self._sp
            lh = max(lh, sz.height())
        return y + lh - rect.y()


def chips_box(items, kind):
    w = QWidget()
    fl = FlowLayout(w)
    if not items:
        fl.addWidget(label("—", "sub"))
    for t in items:
        fl.addWidget(chip(t, kind))
    return w


class StatCard(Card):
    def __init__(self, title, value="—", hint=""):
        super().__init__()
        self.lay.setSpacing(4)
        self.lay.addWidget(label(title, "sub"))
        self.v = label(str(value), "stat"); self.lay.addWidget(self.v)
        self.h = label(hint, "sub", True); self.lay.addWidget(self.h)

    def set(self, value, hint=""):
        self.v.setText(str(value)); self.h.setText(hint)


class Gauge(QWidget):
    def __init__(self, caption="مؤشر الخطر", invert=False):
        super().__init__()
        self.value, self.caption, self.invert = None, caption, invert
        self.pal = {}
        self.setMinimumSize(150, 130)

    def set_value(self, v, pal):
        self.value, self.pal = v, pal
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.pal or {"border": "#ddd", "text": "#222", "sub": "#777", "ok": "#2a2", "warn": "#e90", "bad": "#d33"}
        side = min(self.width(), self.height()) - 24
        rect = QRectF((self.width() - side) / 2, 12, side, side)
        pen = QPen(QColor(pal["border"]), 12); pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen); p.drawArc(rect, 225 * 16, -270 * 16)
        if self.value is not None:
            v = 100 - self.value if self.invert else self.value
            col = pal["ok"] if v < 40 else pal["warn"] if v < 65 else pal["bad"]
            pen.setColor(QColor(col)); p.setPen(pen)
            p.drawArc(rect, 225 * 16, int(-270 * 16 * self.value / 100))
        f = QFont(self.font()); f.setPointSize(20); f.setBold(True); p.setFont(f); p.setPen(QColor(pal["text"]))
        p.drawText(rect.adjusted(0, -8, 0, 0), Qt.AlignmentFlag.AlignCenter, "—" if self.value is None else f"{self.value:.0f}%")
        f.setPointSize(9); f.setBold(False); p.setFont(f); p.setPen(QColor(pal["sub"]))
        p.drawText(rect.adjusted(0, 44, 0, 0), Qt.AlignmentFlag.AlignCenter, self.caption)


class BusyDialog(QDialog):
    """نافذة انتظار منبثقة أثناء المهام الطويلة."""
    def __init__(self, parent, title):
        super().__init__(parent)
        self.setWindowTitle(title); self.setModal(True); self.setMinimumWidth(420)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        lay = QVBoxLayout(self); lay.setContentsMargins(28, 24, 28, 24); lay.setSpacing(14)
        lay.addWidget(label(title, "h2"))
        self.msg = label("جاري العمل ...", "sub", True); lay.addWidget(self.msg)
        bar = QProgressBar(); bar.setRange(0, 0); lay.addWidget(bar)

    def say(self, s):
        self.msg.setText(s)
