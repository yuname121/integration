#!/usr/bin/env python3
"""Generate the theory-first SafeNest system learning guide.

This edition is intentionally different from the implementation-audit guide:
it develops a mental model through narrative, diagrams, and typeset equations.
Reader exercises, rubrics, and test checklists are excluded from the core text.
Each conceptual page is bounded explicitly; overflow raises instead of silently
splitting a unit across pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Flowable, Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "SafeNest_시스템이해_이론학습서_20260728.pdf"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")

PAGE_W, PAGE_H = A4
LEFT = 18 * mm
RIGHT = 18 * mm
BODY_W = PAGE_W - LEFT - RIGHT
TOP_Y = PAGE_H - 22 * mm
BOTTOM_Y = 27 * mm

INK = colors.HexColor("#10233F")
SLATE = colors.HexColor("#334155")
MUTED = colors.HexColor("#64748B")
LINE = colors.HexColor("#CBD5E1")
PAPER = colors.HexColor("#F8FAFC")
WHITE = colors.white
BLUE = colors.HexColor("#1D4ED8")
CYAN = colors.HexColor("#0891B2")
TEAL = colors.HexColor("#0F766E")
GREEN = colors.HexColor("#15803D")
AMBER = colors.HexColor("#B45309")
RED = colors.HexColor("#B91C1C")
PURPLE = colors.HexColor("#6D28D9")
PALE_BLUE = colors.HexColor("#EFF6FF")
PALE_CYAN = colors.HexColor("#ECFEFF")
PALE_GREEN = colors.HexColor("#F0FDF4")
PALE_AMBER = colors.HexColor("#FFFBEB")
PALE_RED = colors.HexColor("#FEF2F2")
PALE_PURPLE = colors.HexColor("#F5F3FF")

pdfmetrics.registerFont(TTFont("KoreanTheory", str(FONT_PATH)))
pdfmetrics.registerFontFamily(
    "KoreanTheory",
    normal="KoreanTheory",
    bold="KoreanTheory",
    italic="KoreanTheory",
    boldItalic="KoreanTheory",
)

STYLES = {
    "body": ParagraphStyle(
        "theory_body",
        fontName="KoreanTheory",
        fontSize=9.1,
        leading=14.2,
        textColor=SLATE,
        wordWrap="CJK",
        allowWidows=0,
        allowOrphans=0,
    ),
    "small": ParagraphStyle(
        "theory_small",
        fontName="KoreanTheory",
        fontSize=7.6,
        leading=11.0,
        textColor=MUTED,
        wordWrap="CJK",
    ),
    "h2": ParagraphStyle(
        "theory_h2",
        fontName="KoreanTheory",
        fontSize=11.2,
        leading=15,
        textColor=INK,
        wordWrap="CJK",
    ),
    "callout": ParagraphStyle(
        "theory_callout",
        fontName="KoreanTheory",
        fontSize=8.5,
        leading=12.5,
        textColor=INK,
        wordWrap="CJK",
    ),
    "table": ParagraphStyle(
        "theory_table",
        fontName="KoreanTheory",
        fontSize=7.4,
        leading=10.0,
        textColor=SLATE,
        wordWrap="CJK",
    ),
    "table_head": ParagraphStyle(
        "theory_table_head",
        fontName="KoreanTheory",
        fontSize=7.5,
        leading=9.8,
        textColor=WHITE,
        alignment=TA_CENTER,
        wordWrap="CJK",
    ),
}


def rich(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def plain(value: object, style: str = "table") -> Paragraph:
    return Paragraph(escape(str(value)).replace("\n", "<br/>"), STYLES[style])


class LayoutError(RuntimeError):
    pass


class EquationFlowable(Flowable):
    """A centered, properly typeset mathematical expression."""

    def __init__(self, formula: str, number: str, max_width: float = BODY_W):
        super().__init__()
        self.formula = formula
        self.number = number
        buf = BytesIO()
        math_to_image(
            f"${formula}$",
            buf,
            prop=FontProperties(size=19),
            dpi=360,
            format="png",
            color="#10233F",
        )
        self.png = buf.getvalue()
        with PILImage.open(BytesIO(self.png)) as im:
            px_w, px_h = im.size
        natural_w = px_w / 360.0 * 72.0
        natural_h = px_h / 360.0 * 72.0
        usable_w = max_width - 34 * mm
        scale = min(1.0, usable_w / max(natural_w, 1.0))
        self.image_w = natural_w * scale
        self.image_h = natural_h * scale
        self.width = max_width
        self.height = max(18 * mm, self.image_h + 10 * mm)

    def wrap(self, availWidth, availHeight):
        return min(self.width, availWidth), self.height

    def draw(self):
        c = self.canv
        c.setFillColor(PALE_PURPLE)
        c.setStrokeColor(colors.HexColor("#C4B5FD"))
        c.setLineWidth(0.7)
        c.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=1)
        x = (self.width - self.image_w) / 2
        y = (self.height - self.image_h) / 2
        c.drawImage(
            ImageReader(BytesIO(self.png)),
            x,
            y,
            width=self.image_w,
            height=self.image_h,
            mask="auto",
        )
        c.setFillColor(PURPLE)
        c.setFont("KoreanTheory", 7.2)
        c.drawRightString(self.width - 5 * mm, self.height - 5.2 * mm, f"({self.number})")


class DiagramFlowable(Flowable):
    """Small vector diagrams used to explain causal structure and signal flow."""

    def __init__(self, kind: str, data: dict, width: float = BODY_W, height: float = 42 * mm):
        super().__init__()
        self.kind = kind
        self.data = data
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return min(self.width, availWidth), self.height

    def _text(self, x, y, text, size=7.2, color=INK, align="center"):
        c = self.canv
        c.setFillColor(color)
        c.setFont("KoreanTheory", size)
        lines = str(text).split("\n")
        for idx, line in enumerate(lines):
            yy = y - idx * (size + 2)
            if align == "left":
                c.drawString(x, yy, line)
            else:
                c.drawCentredString(x, yy, line)

    def _arrow(self, x1, y1, x2, y2, color=MUTED):
        c = self.canv
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.0)
        c.line(x1, y1, x2, y2)
        import math

        angle = math.atan2(y2 - y1, x2 - x1)
        ah = 4.2
        for delta in (2.55, -2.55):
            c.line(x2, y2, x2 + ah * math.cos(angle + delta), y2 + ah * math.sin(angle + delta))

    def _box(self, x, y, w, h, title, sub="", tone=0):
        palette = [(PALE_BLUE, BLUE), (PALE_CYAN, CYAN), (PALE_GREEN, GREEN), (PALE_PURPLE, PURPLE)]
        bg, stroke = palette[tone % len(palette)]
        c = self.canv
        c.setFillColor(bg)
        c.setStrokeColor(stroke)
        c.setLineWidth(0.8)
        c.roundRect(x, y, w, h, 5, fill=1, stroke=1)
        self._text(x + w / 2, y + h * 0.62, title, 7.4, INK)
        if sub:
            self._text(x + w / 2, y + h * 0.31, sub, 6.5, MUTED)

    def draw(self):
        c = self.canv
        kind = self.kind
        if kind == "flow":
            labels = self.data["labels"]
            n = len(labels)
            arrow_w = 7 * mm
            box_w = (self.width - 8 * mm - arrow_w * (n - 1)) / n
            box_h = self.height - 12 * mm
            x = 4 * mm
            y = 6 * mm
            for idx, label in enumerate(labels):
                title, sub = label if isinstance(label, tuple) else (label, "")
                self._box(x, y, box_w, box_h, title, sub, idx)
                if idx < n - 1:
                    self._arrow(x + box_w + 1.2 * mm, y + box_h / 2, x + box_w + arrow_w - 1.2 * mm, y + box_h / 2)
                x += box_w + arrow_w
        elif kind == "layers":
            labels = self.data["labels"]
            gap = 2.2 * mm
            h = (self.height - 8 * mm - gap * (len(labels) - 1)) / len(labels)
            for idx, label in enumerate(labels):
                title, sub = label if isinstance(label, tuple) else (label, "")
                y = self.height - 4 * mm - (idx + 1) * h - idx * gap
                self._box(12 * mm, y, self.width - 24 * mm, h, title, sub, idx)
                if idx < len(labels) - 1:
                    self._arrow(self.width / 2, y - 0.6 * mm, self.width / 2, y - gap + 0.6 * mm)
        elif kind == "timeline":
            markers = self.data["markers"]
            x0, x1 = 12 * mm, self.width - 12 * mm
            y = self.height * 0.48
            c.setStrokeColor(INK)
            c.setLineWidth(1.5)
            c.line(x0, y, x1, y)
            self._arrow(x1 - 4, y, x1, y, INK)
            for idx, (frac, title, sub) in enumerate(markers):
                x = x0 + frac * (x1 - x0)
                c.setStrokeColor(BLUE if idx < len(markers) - 1 else RED)
                c.setLineWidth(1.2)
                c.line(x, y - 4 * mm, x, y + 4 * mm)
                self._text(x, y + 9 * mm, title, 7.2, INK)
                self._text(x, y - 8 * mm, sub, 6.5, MUTED)
        elif kind == "fan_in":
            sources = self.data["sources"]
            center = self.data["center"]
            output = self.data["output"]
            src_w, src_h = 34 * mm, 8.5 * mm
            left_x = 2 * mm
            for idx, src in enumerate(sources):
                y = self.height - 7 * mm - (idx + 1) * src_h - idx * 1.3 * mm
                self._box(left_x, y, src_w, src_h, src, "", idx)
                self._arrow(left_x + src_w, y + src_h / 2, self.width * 0.46, self.height / 2)
            self._box(self.width * 0.46, self.height / 2 - 10 * mm, 42 * mm, 20 * mm, center, "결합 규칙", 3)
            self._arrow(self.width * 0.46 + 42 * mm, self.height / 2, self.width - 37 * mm, self.height / 2)
            self._box(self.width - 36 * mm, self.height / 2 - 9 * mm, 34 * mm, 18 * mm, output, "", 2)
        elif kind == "state":
            nodes = {
                "NORMAL": (8 * mm, self.height / 2 - 8 * mm, PALE_GREEN, GREEN),
                "CAUTION": (57 * mm, self.height / 2 - 8 * mm, PALE_AMBER, AMBER),
                "DANGER": (108 * mm, self.height / 2 - 8 * mm, PALE_RED, RED),
                "FAULT": (57 * mm, 3 * mm, PAPER, MUTED),
            }
            for name, (x, y, bg, stroke) in nodes.items():
                c.setFillColor(bg); c.setStrokeColor(stroke); c.roundRect(x, y, 38 * mm, 16 * mm, 5, fill=1, stroke=1)
                self._text(x + 19 * mm, y + 8.5 * mm, name, 7.5, INK)
            self._arrow(46 * mm, self.height / 2, 56 * mm, self.height / 2, AMBER)
            self._arrow(95 * mm, self.height / 2, 107 * mm, self.height / 2, RED)
            self._arrow(57 * mm, self.height / 2 - 4 * mm, 47 * mm, self.height / 2 - 4 * mm, GREEN)
            self._arrow(108 * mm, self.height / 2 - 4 * mm, 96 * mm, self.height / 2 - 4 * mm, AMBER)
            self._arrow(76 * mm, self.height / 2 - 8 * mm, 76 * mm, 20 * mm, MUTED)
            self._arrow(114 * mm, 11 * mm, 97 * mm, 11 * mm, MUTED)
            self._text(82 * mm, self.height - 4 * mm, "진입 문턱과 해제 문턱을 분리", 6.8, MUTED)
        elif kind == "schema":
            c.setFillColor(PALE_BLUE); c.setStrokeColor(BLUE); c.roundRect(3 * mm, 3 * mm, self.width - 6 * mm, self.height - 6 * mm, 6, fill=1, stroke=1)
            self._text(10 * mm, self.height - 9 * mm, "SensorEnvelopeV1", 8.2, BLUE, "left")
            cols = [
                ("Identity", "device_id\nsession_id\nsequence"),
                ("Time", "measured_at\nreceived_at\nage budget"),
                ("Meaning", "value·unit\nsemantic\ncalibration"),
                ("Health", "status\nquality\nreason"),
            ]
            w = (self.width - 18 * mm) / 4
            for idx, (title, sub) in enumerate(cols):
                self._box(6 * mm + idx * (w + 2 * mm), 8 * mm, w, self.height - 23 * mm, title, sub, idx)
        elif kind == "plot":
            self._draw_plot(self.data.get("plot", "lowpass"))

    def _axes(self, xlabel, ylabel):
        c = self.canv
        x0, y0 = 16 * mm, 9 * mm
        x1, y1 = self.width - 10 * mm, self.height - 8 * mm
        c.setStrokeColor(MUTED); c.setLineWidth(0.7)
        c.line(x0, y0, x1, y0); c.line(x0, y0, x0, y1)
        self._text(x1, y0 - 4 * mm, xlabel, 6.5, MUTED)
        self._text(x0 + 2 * mm, y1 + 1.5 * mm, ylabel, 6.5, MUTED, "left")
        return x0, y0, x1, y1

    def _draw_plot(self, kind):
        import math

        c = self.canv
        x0, y0, x1, y1 = self._axes(
            "시간 / sample" if kind not in {"spectrum", "quantization"} else ("frequency" if kind == "spectrum" else "실수 x"),
            "응답" if kind not in {"spectrum", "quantization"} else ("에너지" if kind == "spectrum" else "정수 q"),
        )
        w, h = x1 - x0, y1 - y0
        if kind == "lowpass":
            c.setStrokeColor(LINE); c.setLineWidth(1.0); c.line(x0, y0 + 0.82 * h, x1, y0 + 0.82 * h)
            pts = []
            for i in range(41):
                t = i / 40 * 10
                val = 1 - math.exp(-t / 3.5)
                pts.append((x0 + i / 40 * w, y0 + 0.82 * h * val))
            c.setStrokeColor(CYAN); c.setLineWidth(2.2)
            for a, b in zip(pts, pts[1:]): c.line(a[0], a[1], b[0], b[1])
            self._text(x0 + 0.69 * w, y0 + 0.62 * h, "IIR: 부드럽지만 지연", 6.8, CYAN)
        elif kind == "spectrum":
            c.setFillColor(PALE_BLUE); c.rect(x0 + 0.12 * w, y0, 0.40 * w, h, fill=1, stroke=0)
            for frac, amp, tone in ((0.12, 0.25, MUTED), (0.33, 0.92, BLUE), (0.50, 0.36, MUTED), (0.72, 0.18, MUTED)):
                c.setStrokeColor(tone); c.setLineWidth(3 if tone == BLUE else 1.2)
                c.line(x0 + frac * w, y0, x0 + frac * w, y0 + amp * h)
            self._text(x0 + 0.32 * w, y0 + 0.72 * h, "호흡 대역", 6.8, BLUE)
        elif kind == "quantization":
            c.setStrokeColor(BLUE); c.setLineWidth(1.6)
            last = None
            for i in range(9):
                xa = x0 + i / 9 * w; xb = x0 + (i + 1) / 9 * w
                yy = y0 + (i + 0.5) / 9 * h
                c.line(xa, yy, xb, yy)
                if last is not None: c.line(xa, last, xa, yy)
                last = yy
            c.setStrokeColor(RED); c.setDash(3, 2)
            c.line(x0 + 0.88 * w, y0, x0 + 0.88 * w, y1); c.setDash()
            self._text(x0 + 0.82 * w, y0 + 0.83 * h, "포화", 6.8, RED)
        elif kind == "hysteresis":
            c.setStrokeColor(RED); c.setLineWidth(2.0)
            c.line(x0 + 0.15 * w, y0 + 0.25 * h, x0 + 0.80 * w, y0 + 0.25 * h)
            c.line(x0 + 0.80 * w, y0 + 0.25 * h, x0 + 0.80 * w, y0 + 0.76 * h)
            c.setStrokeColor(GREEN)
            c.line(x0 + 0.68 * w, y0 + 0.76 * h, x0 + 0.15 * w, y0 + 0.76 * h)
            c.line(x0 + 0.68 * w, y0 + 0.76 * h, x0 + 0.68 * w, y0 + 0.25 * h)
            self._text(x0 + 0.80 * w, y0 + 0.14 * h, "진입", 6.8, RED)
            self._text(x0 + 0.68 * w, y0 + 0.88 * h, "해제", 6.8, GREEN)
        elif kind == "co2":
            # Q_v가 커지면 time constant V/Q_v와 steady-state excess G/Q_v가 모두 작아진다.
            for tau, steady, tone, label in (
                (0.40, 0.86, RED, "환기 약함: 높은 정상상태"),
                (0.18, 0.48, CYAN, "환기 강함: 낮은 정상상태"),
            ):
                pts = []
                for i in range(41):
                    t = i / 40
                    val = steady * (1 - math.exp(-t / tau))
                    pts.append((x0 + t * w, y0 + val * h))
                c.setStrokeColor(tone); c.setLineWidth(2)
                for a, b in zip(pts, pts[1:]): c.line(a[0], a[1], b[0], b[1])
                self._text(x0 + 0.70 * w, y0 + (0.75 if steady > 0.8 else 0.39) * h, label, 6.4, tone)
        elif kind == "phase":
            c.setStrokeColor(BLUE); c.setLineWidth(2.0)
            pts = []
            for i in range(81):
                t = i / 80
                val = 0.5 + 0.34 * math.sin(2 * math.pi * 3 * t)
                pts.append((x0 + t * w, y0 + val * h))
            for a, b in zip(pts, pts[1:]): c.line(a[0], a[1], b[0], b[1])
            self._text(x0 + 0.73 * w, y0 + 0.84 * h, "흉부 변위가 위상 변조로 나타남", 6.6, BLUE)


class PageWriter:
    def __init__(self, output: Path):
        output.parent.mkdir(parents=True, exist_ok=True)
        self.c = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
        self.c.setTitle("SafeNest 시스템 이해 이론학습서")
        self.c.setSubject("전자공학 고학년 팀원을 위한 이론·원리 중심 시스템 학습서")
        self.c.setAuthor("SafeNest System Understanding")
        self.page_no = 0
        self.y = TOP_Y
        self.title = ""
        self.part = ""
        self.source = ""

    def _draw(self, item: Flowable, gap_after: float = 2.5 * mm):
        width, height = item.wrap(BODY_W, self.y - BOTTOM_Y)
        if height > self.y - BOTTOM_Y + 0.01:
            raise LayoutError(
                f"page {self.page_no} overflow in '{self.title}': "
                f"need {height/mm:.1f} mm, have {(self.y-BOTTOM_Y)/mm:.1f} mm"
            )
        item.drawOn(self.c, LEFT, self.y - height)
        self.y -= height + gap_after

    def cover(self):
        self.page_no += 1
        c = self.c
        c.setFillColor(INK); c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        c.setFillColor(BLUE); c.rect(0, PAGE_H - 13 * mm, PAGE_W, 13 * mm, fill=1, stroke=0)
        c.setFillColor(CYAN); c.circle(PAGE_W - 31 * mm, PAGE_H - 45 * mm, 17 * mm, fill=1, stroke=0)
        c.setFillColor(PURPLE); c.circle(PAGE_W - 16 * mm, PAGE_H - 60 * mm, 10 * mm, fill=1, stroke=0)
        c.setFillColor(WHITE); c.setFont("KoreanTheory", 8)
        c.drawString(20 * mm, PAGE_H - 31 * mm, "SafeNest ON-DEVICE AI SYSTEM")
        c.setFont("KoreanTheory", 29)
        c.drawString(20 * mm, PAGE_H - 62 * mm, "시스템 이해")
        c.drawString(20 * mm, PAGE_H - 78 * mm, "이론학습서")
        c.setFillColor(colors.HexColor("#CBD5E1")); c.setFont("KoreanTheory", 11)
        c.drawString(20 * mm, PAGE_H - 94 * mm, "원리·수식·도식으로 연결하는 전자공학 고학년용 교재")
        c.setFillColor(colors.HexColor("#1E3A5F"))
        c.roundRect(20 * mm, 68 * mm, PAGE_W - 40 * mm, 74 * mm, 7, fill=1, stroke=0)
        c.setFillColor(WHITE); c.setFont("KoreanTheory", 9.3)
        lines = [
            "물리 현상 → 센서 → 신호처리 → Tensor·Edge AI",
            "→ 위험 융합 → 상태기계 → 확장 가능한 시스템 계약",
        ]
        for idx, line in enumerate(lines):
            c.drawString(29 * mm, 121 * mm - idx * 9 * mm, line)
        c.setFillColor(colors.HexColor("#BFDBFE")); c.setFont("KoreanTheory", 8)
        c.drawString(29 * mm, 89 * mm, "학습 목표: 대본 없이 전체 구조와 원리를 설명할 수 있는 mental model")
        c.drawString(29 * mm, 80 * mm, "범위: 이론과 현재 구현 연결 / 실습·시험·rubric은 본문에서 제외")
        c.setFillColor(colors.HexColor("#94A3B8")); c.setFont("KoreanTheory", 7.5)
        c.drawString(20 * mm, 18 * mm, "v3.0 · 59-page theory-first edition")
        c.drawRightString(PAGE_W - 20 * mm, 18 * mm, "SafeNest System Understanding")
        c.bookmarkPage("cover-theory")
        c.addOutlineEntry("표지", "cover-theory", level=0, closed=False)
        c.showPage()

    def start(self, number: str, title: str, subtitle: str, part: str, unit: str = "", source: str = ""):
        self.page_no += 1
        self.y = TOP_Y
        self.title = title
        self.part = part
        self.source = source
        c = self.c
        c.setFillColor(WHITE); c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        c.setFillColor(INK); c.setFont("KoreanTheory", 6.8)
        c.drawString(LEFT, PAGE_H - 11 * mm, "SafeNest 시스템 이해 이론학습서")
        c.setFillColor(MUTED); c.drawRightString(PAGE_W - RIGHT, PAGE_H - 11 * mm, part)
        c.setStrokeColor(LINE); c.line(LEFT, PAGE_H - 14 * mm, PAGE_W - RIGHT, PAGE_H - 14 * mm)
        c.setFillColor(BLUE); c.setFont("KoreanTheory", 8)
        label = f"THEORY UNIT {number}"
        if unit:
            label += f" · {unit}"
        c.drawString(LEFT, self.y, label)
        self.y -= 9 * mm
        c.setFillColor(INK); c.setFont("KoreanTheory", 18)
        c.drawString(LEFT, self.y, title)
        self.y -= 7 * mm
        c.setFillColor(MUTED); c.setFont("KoreanTheory", 8)
        c.drawString(LEFT, self.y, subtitle)
        self.y -= 6 * mm
        c.setStrokeColor(BLUE); c.setLineWidth(1.5); c.line(LEFT, self.y, LEFT + 24 * mm, self.y)
        c.setStrokeColor(LINE); c.setLineWidth(0.5); c.line(LEFT + 24 * mm, self.y, PAGE_W - RIGHT, self.y)
        self.y -= 5 * mm
        key = f"theory-{self.page_no:02d}-{number}"
        c.bookmarkPage(key)
        c.addOutlineEntry(f"{number}. {title}", key, level=0, closed=False)

    def h2(self, text: str):
        self._draw(rich(f"<b>{text}</b>", "h2"), gap_after=0.7 * mm)

    def body(self, text: str, gap: float = 2.6 * mm):
        self._draw(rich(text, "body"), gap_after=gap)

    def small(self, text: str, gap: float = 2.0 * mm):
        self._draw(rich(text, "small"), gap_after=gap)

    def bullets(self, items: list[str], gap: float = 2.5 * mm):
        rows = [[rich('<font color="#1D4ED8">•</font>', "body"), rich(item, "body")] for item in items]
        table = Table(rows, colWidths=[5 * mm, BODY_W - 5 * mm])
        table.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 1),
            ("TOPPADDING", (0,0), (-1,-1), 0.5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 1.5),
        ]))
        self._draw(table, gap_after=gap)

    def equation(self, formula: str, number: str, meaning: str | None = None):
        self._draw(EquationFlowable(formula, number), gap_after=1.2 * mm)
        if meaning:
            self.small(f"<b>수식의 의미</b> · {meaning}", gap=2.4 * mm)

    def diagram(self, kind: str, data: dict, height: float = 42 * mm, caption: str | None = None):
        self._draw(DiagramFlowable(kind, data, height=height), gap_after=1.0 * mm)
        if caption:
            self.small(f"<b>도식 해석</b> · {caption}", gap=2.4 * mm)

    def callout(self, label: str, text: str, tone: str = "blue", gap: float = 2.5 * mm):
        palette = {
            "blue": (PALE_BLUE, BLUE), "cyan": (PALE_CYAN, CYAN), "green": (PALE_GREEN, GREEN),
            "amber": (PALE_AMBER, AMBER), "red": (PALE_RED, RED), "purple": (PALE_PURPLE, PURPLE),
        }
        bg, accent = palette[tone]
        inner = Table(
            [[rich(f'<font color="{accent.hexval()}"><b>{label}</b></font>', "callout")], [rich(text, "callout")]],
            colWidths=[BODY_W - 10 * mm],
        )
        inner.setStyle(TableStyle([
            ("LEFTPADDING", (0,0), (-1,-1), 0), ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("TOPPADDING", (0,0), (-1,-1), 1), ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ]))
        outer = Table([[inner]], colWidths=[BODY_W])
        outer.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), bg), ("BOX", (0,0), (-1,-1), 0.6, accent),
            ("LINEBEFORE", (0,0), (0,0), 4, accent), ("LEFTPADDING", (0,0), (-1,-1), 7),
            ("RIGHTPADDING", (0,0), (-1,-1), 7), ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        self._draw(outer, gap_after=gap)

    def summary(self, items: list[str]):
        text = "<br/>".join(f"<b>{idx+1}.</b> {item}" for idx, item in enumerate(items))
        self.callout("말로 설명할 핵심", text, "green", gap=2.2 * mm)

    def reference_table(self, headers: list[str], rows: list[list[object]], widths: list[float]):
        total = sum(widths)
        data = [[plain(v, "table_head") for v in headers]]
        data += [[plain(v) for v in row] for row in rows]
        table = Table(data, colWidths=[BODY_W * w / total for w in widths])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), INK), ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("GRID", (0,0), (-1,-1), 0.35, LINE), ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, PAPER]),
            ("LEFTPADDING", (0,0), (-1,-1), 4), ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 3.5), ("BOTTOMPADDING", (0,0), (-1,-1), 3.5),
        ]))
        self._draw(table, gap_after=2.4 * mm)

    def finish(self):
        c = self.c
        if self.source:
            c.setFillColor(PAPER); c.rect(LEFT, 18 * mm, BODY_W, 6.5 * mm, fill=1, stroke=0)
            c.setFillColor(MUTED); c.setFont("KoreanTheory", 5.9)
            src = self.source if len(self.source) <= 145 else self.source[:142] + "..."
            c.drawString(LEFT + 2 * mm, 20.3 * mm, f"SafeNest 연결 근거 · {src}")
        c.setStrokeColor(LINE); c.line(LEFT, 14 * mm, PAGE_W - RIGHT, 14 * mm)
        c.setFillColor(MUTED); c.setFont("KoreanTheory", 6.5)
        c.drawString(LEFT, 9.5 * mm, "2026-07-28 · 이론학습서 v3.0")
        c.drawRightString(PAGE_W - RIGHT, 9.5 * mm, str(self.page_no))
        c.showPage()

    def save(self):
        self.c.save()


@dataclass
class TheoryPage:
    number: str
    title: str
    subtitle: str
    part: str
    unit: str
    source: str
    blocks: list[tuple]


def page(number, title, subtitle, part, unit, source, *blocks):
    return TheoryPage(number, title, subtitle, part, unit, source, list(blocks))


def render_page(w: PageWriter, p: TheoryPage):
    w.start(p.number, p.title, p.subtitle, p.part, p.unit, p.source)
    for block in p.blocks:
        kind, *args = block
        if kind == "body": w.body(*args)
        elif kind == "h2": w.h2(*args)
        elif kind == "small": w.small(*args)
        elif kind == "bullets": w.bullets(*args)
        elif kind == "equation": w.equation(*args)
        elif kind == "diagram": w.diagram(*args)
        elif kind == "callout": w.callout(*args)
        elif kind == "summary": w.summary(*args)
        elif kind == "table": w.reference_table(*args)
        else: raise ValueError(f"unknown block kind: {kind}")
    w.finish()


def build_pages() -> list[TheoryPage]:
    pages: list[TheoryPage] = []

    pages.append(page(
        "00", "이 학습서의 읽는 법", "표를 외우는 대신 현상·변환·판단의 인과관계를 하나의 mental model로 만든다.",
        "학습 안내", "READING GUIDE", "README.md; models/model_manifest.json; integrated_node/safenest_risk_engine.py",
        ("body", "이 문서의 목표는 코드 파일의 위치를 암기하는 것이 아니다. 센서가 어떤 물리량을 관측하고, 그 값이 어떤 시간축과 단위를 거쳐 신호와 tensor가 되며, 마지막에 어떤 규칙과 상태기계로 연결되는지를 설명할 수 있어야 한다. 따라서 각 단원은 <b>현상 → 정의 → 도식 → 수식 → SafeNest 연결 → 한계</b>의 순서로 전개한다."),
        ("diagram", "flow", {"labels": [("현상", "무엇이 변하는가"), ("측정", "센서가 보는 것"), ("변환", "신호·tensor"), ("판단", "AI·규칙"), ("상태", "위험·건강")]}, 35 * mm, "한 단계의 출력은 다음 단계의 입력 의미를 제한한다. 중간 의미를 생략하면 shape가 맞아도 시스템은 틀릴 수 있다."),
        ("body", "핵심 개념은 두 쪽을 한 학습 단위로 묶었다. 첫 쪽에서는 정의와 원리를 세우고, 둘째 쪽에서는 수식의 의미와 현재 시스템의 대응 관계를 설명한다. 수식 아래에는 기호·단위·가정을 문장으로 풀어 쓴다."),
        ("callout", "본문에서 제외한 것", "실습 절차, 점검 문제, 채점표, 테스트 명령은 이론의 흐름을 끊기 때문에 넣지 않았다. 코드와 테스트는 이론이 현재 시스템에 어떻게 나타나는지 확인하는 근거로만 사용한다.", "amber"),
        ("summary", ["SafeNest는 센서 모음이 아니라 물리계에서 판단 상태까지 이어지는 변환 시스템이다.", "각 단계는 값뿐 아니라 단위·시간·상태 의미를 다음 단계에 전달해야 한다.", "이론을 이해했다는 것은 수식과 도식을 사용해 전체 인과관계를 말로 복원할 수 있다는 뜻이다."]),
    ))

    pages.append(page(
        "01", "전체 시스템을 먼저 본다", "세부 모델을 보기 전에 신호가 어디에서 생기고 어디에서 의미가 바뀌는지 고정한다.",
        "학습 안내", "SYSTEM MAP", "integrated_node/virtual_sensor_streamer.py:171-199; integrated_node/safenest_risk_engine.py:120-415",
        ("body", "SafeNest의 입력은 Thermal frame, CO₂ 농도와 습도, mmWave 호흡 정보, PIR motion이다. 이 값들은 곧바로 위험도가 아니다. 먼저 입력 계약을 통과하고, 필요한 경우 window·정규화·양자화를 거친 뒤 AI 또는 규칙 기반 부분 판단이 된다. 부분 판단은 위험 융합과 상태기계를 거쳐 사용자에게 전달된다."),
        ("diagram", "layers", {"labels": [("물리계", "사람·공간·환기·움직임"), ("센서 관측", "thermal·CO₂·mmWave·PIR"), ("신호 표현", "frame·ppm·phase·motion"), ("추론과 규칙", "TFLite class·부분 위험도"), ("융합과 상태", "risk·health·reason"), ("출력", "경보·UI·telemetry")]}, 76 * mm, "위쪽은 현실 세계이고 아래쪽으로 갈수록 소프트웨어가 부여한 해석이 강해진다."),
        ("body", "시스템을 설명할 때는 항상 위에서 아래로 내려가야 한다. 예를 들어 ‘mmWave AI가 무호흡을 감지한다’고 바로 말하면 phase의 물리 의미, 10 Hz sampling, 300-sample window, 모델 class와 2초 확인 규칙이 모두 생략된다. 정확한 설명은 이 연결을 복원하는 일이다."),
        ("callout", "두 개의 병렬 축", "SafeNest는 <b>관측 위험</b>과 <b>판단 경로의 건강 상태</b>를 함께 다룬다. 위험이 낮아도 sensor나 model 경로가 고장 났다면 안전하다고 단정할 수 없다.", "blue"),
    ))

    pages.append(page(
        "02", "기호와 계층을 먼저 고정한다", "같은 문자와 같은 class 번호가 다른 계층을 뜻하지 않도록 표기를 분리한다.",
        "학습 안내", "NOTATION", "models/model_manifest.json; risk/risk_rules.py; integrated_node/safenest_risk_engine.py",
        ("table", ["기호", "이 문서의 의미", "단위·범위"], [
            ["x(t), x[k]", "연속시간 물리 신호 / sampling된 이산 신호", "물리량에 따라 다름"],
            ["f_s, N", "sample rate / window sample 수", "Hz / samples"],
            ["X[m]", "N-point DFT의 m번째 주파수 성분", "입력 단위에 의존"],
            ["q_int8", "양자화된 정수 tensor 원소", "-128…127"],
            ["s_q, z_q", "양자화 scale / zero point", "실수/정수"],
            ["S_i, w_i", "i번째 부분 위험도 / 가중치", "0…1 / 합 1"],
            ["R_raw, R[k]", "융합 원시 위험도 / 필터링된 위험도", "0…100"],
            ["Q_i", "센서 경로의 품질 또는 가용성", "현재 coarse gate"],
        ], [0.17, 0.56, 0.27]),
        ("body", "<b>class 번호는 model namespace 안에서만 의미가 있다.</b> thermal.class=2는 FALL이고 mmwave.class=2는 APNEA다. 또한 q_int8은 tensor의 저장 정수이고 Q_i는 센서 품질이다. 둘을 같은 q로 생각하면 양자화와 품질 가중치가 뒤섞인다."),
        ("callout", "용어 사용 원칙", "처음 등장하는 전문 용어는 그 자리에서 정의한다. 뒤에서 나오는 수식은 이 페이지의 namespace를 사용하고, 값·단위·시간축·상태의 계층을 섞지 않는다.", "purple"),
        ("summary", ["기호는 계산을 줄여 쓰기 위한 약속이며 물리 의미와 단위를 잃어서는 안 된다.", "같은 숫자나 class index라도 생산한 model과 처리 계층이 다르면 의미가 다르다.", "시스템 설명에서는 값과 함께 namespace·unit·timebase를 말해야 한다."]),
    ))

    pages.append(page(
        "P1", "PART I · 시스템을 보는 기본 틀", "복잡한 시스템은 구성요소 목록보다 계층·계약·시간·상태의 관계로 이해한다.",
        "PART I · 시스템 기초", "PART OPENER", "README.md; models/model_manifest.json; integrated_node/safenest_risk_engine.py",
        ("body", "센서와 AI를 많이 연결한다고 자동으로 하나의 시스템이 되는 것은 아니다. 각 구성요소가 생산하는 값의 의미, 그 값이 유효한 시간, 내부에 남는 상태, 고장 시 downstream에 전달되는 정보가 합쳐져야 시스템이 된다. PART I은 이후 모든 장을 읽는 공통 문법을 만든다."),
        ("diagram", "layers", {"labels": [("계층", "현실→신호→판단"), ("계약", "shape보다 의미"), ("시간", "sampling·age·latency"), ("상태", "history·timer·transition"), ("두 출력축", "risk와 health")]}, 70 * mm, "시스템 확장으로 복잡도가 증가할수록 이 다섯 축을 먼저 고정해야 한다."),
        ("h2", "이 장의 중심 질문"),
        ("bullets", ["값이 같아도 시간과 단위가 다르면 왜 다른 정보인가?", "순수 함수와 과거를 기억하는 stateful component는 무엇이 다른가?", "위험도가 낮다는 말과 시스템이 정상이라는 말은 왜 분리해야 하는가?"]),
        ("summary", ["시스템은 component의 합이 아니라 contract와 state transition의 연결이다.", "확장은 새 기능을 추가하기 전에 기존 의미 경계를 명확히 하는 작업이다."]),
    ))

    pages.append(page(
        "03", "계층은 의미가 바뀌는 경계다", "물리량에서 경보까지 내려갈수록 관측값에 해석과 정책이 추가된다.",
        "PART I · 시스템 기초", "UNIT 01 · 1/2", "integrated_node/safenest_risk_engine.py:120-415; inference/*_interpreter.py",
        ("body", "<b>추상화 계층</b>은 세부를 숨기는 장치이면서 동시에 의미가 바뀌는 경계다. 물리계의 흉부 변위는 radar complex signal이 되고, 특정 range bin의 phase가 되며, 300-sample tensor와 model class를 거쳐 호흡 위험 증거가 된다. 각 계층은 이전 계층 전체가 아니라 필요한 표현만 전달한다."),
        ("diagram", "flow", {"labels": [("물리량", "흉부 변위"), ("센서신호", "I/Q·rFFT"), ("표현", "resp_phase"), ("Tensor", "[1,300,1]"), ("판단", "class·rule")]}, 39 * mm, "화살표마다 정보는 압축되고 새로운 가정이 추가된다."),
        ("h2", "추상화가 필요한 이유"),
        ("body", "Downstream이 radar chirp 전체를 알지 않아도 resp_phase를 사용할 수 있어야 구성요소를 교체하고 재사용할 수 있다. 그러나 upstream이 phase unwrap과 clutter 제거를 했다는 사실까지 숨기면 같은 shape의 다른 신호가 model에 들어갈 수 있다. 따라서 추상화는 세부 구현을 숨기되 semantic contract는 숨기지 않아야 한다."),
        ("callout", "SafeNest에서의 경계", "Adapter는 생산자별 raw 형식을 canonical time grid와 signal semantic으로 바꾸는 경계다. Interpreter는 canonical signal을 tensor로 바꾸는 경계이며, RiskRules는 class와 sensor value를 정책 의미로 바꾸는 경계다.", "blue"),
    ))

    pages.append(page(
        "04", "인과관계와 상태를 함께 본다", "현재 출력은 현재 입력뿐 아니라 과거 상태와 update order에도 의존한다.",
        "PART I · 시스템 기초", "UNIT 01 · 2/2", "integrated_node/safenest_risk_engine.py:73-96,326-344; risk/risk_rules.py:82-146",
        ("body", "입력만으로 출력이 결정되는 순수 함수라면 같은 입력은 언제나 같은 결과를 낸다. 하지만 SafeNest에는 window, moving mean, IIR state, 이전 status, apnea timer가 남는다. 따라서 동일한 packet도 직전 history와 timer가 다르면 다른 결과를 낼 수 있다."),
        ("equation", r"s_{k+1}=f(s_k,x_k),\qquad y_k=g(s_k,x_k)", "1.1", "x_k는 현재 입력, s_k는 buffer·filter·timer를 포함한 내부 상태, y_k는 현재 출력이다."),
        ("diagram", "flow", {"labels": [("현재 입력", "x_k"), ("상태 갱신", "f(s_k,x_k)"), ("새 상태", "s_{k+1}"), ("출력", "y_k")]}, 34 * mm, "입력과 상태가 함께 다음 출력의 원인이 된다."),
        ("body", "이 식은 update order가 시스템 의미의 일부임을 보여준다. 예를 들어 emergency에서는 raw risk history를 갱신하지 않고 IIR state를 100으로 바꾼다. 이후 복귀 속도는 emergency 이전 history와 새 IIR state를 함께 반영한다."),
        ("summary", ["추상화 계층은 정보가 압축되고 의미가 바뀌는 경계다.", "Stateful system의 출력은 현재 입력과 과거 상태의 함수다.", "시스템을 설명할 때는 값의 경로뿐 아니라 state owner와 update order를 포함해야 한다."]),
    ))

    pages.append(page(
        "05", "기계 계약은 구조를 고정한다", "shape·dtype·key는 값이 해석되기 전에 확인할 수 있는 최소 조건이다.",
        "PART I · 시스템 기초", "UNIT 02 · 1/2", "models/model_manifest.json; inference/mmwave_interpreter.py:113-158",
        ("body", "<b>기계 계약</b>은 프로그램이 자동으로 검사할 수 있는 구조적 조건이다. message의 key, tensor shape, dtype, quantization parameter, schema version이 여기에 속한다. 기계 계약이 없으면 잘못된 입력이 downstream 계산으로 들어가 더 늦고 모호한 오류를 만든다."),
        ("diagram", "schema", {}, 49 * mm, "식별·시간·의미·건강 정보가 하나의 envelope에서 기계적으로 검증되어야 한다."),
        ("h2", "계약 검사의 순서"),
        ("body", "검사는 대체로 schema version과 필수 key에서 시작해 type·shape·finite 여부, unit·semantic, 시간 순서와 freshness로 진행한다. 앞 단계가 실패하면 뒤의 수치 계산을 수행하지 않는 것이 좋다. 이 순서가 예외를 containment하고 reason을 일관되게 만든다."),
        ("callout", "형태가 맞는 것의 의미", "[1,300,1] int8이라는 사실은 model이 호출될 수 있다는 뜻일 뿐, 10 Hz의 30초 unwrapped·clutter-removed phase라는 뜻까지 보장하지 않는다.", "amber"),
    ))

    pages.append(page(
        "06", "의미 계약은 숫자의 뜻을 고정한다", "같은 배열이라도 단위·전처리·시간축이 다르면 다른 신호다.",
        "PART I · 시스템 기초", "UNIT 02 · 2/2", "models/mmwave/sensor_stats_metadata_v0.1.0.json; models/model_manifest.json",
        ("body", "<b>Semantic contract</b>는 숫자가 무엇을 뜻하는지 정한다. mmWave model의 입력은 단순한 300개 실수가 아니라 10 Hz로 sampling된 30초 resp_phase이며 metadata에는 unwrapped·clutter-removed 의미가 기록돼 있다. 값의 발생 과정이 달라지면 shape가 같아도 학습 분포와 입력 의미가 달라진다."),
        ("diagram", "flow", {"labels": [("같은 shape", "300 values"), ("단위", "rad 또는 rpm"), ("시간축", "10 Hz·30 s"), ("전처리", "unwrap·clutter"), ("의미", "model input")]}, 39 * mm, "기계 계약은 첫 상자만 확인하고 의미 계약은 나머지 상자를 확인한다."),
        ("h2", "의미 계약이 깨지는 전형적인 경우"),
        ("body", "breath_rpm을 300번 복제한 배열은 resp_phase window가 아니다. RPM은 한 구간의 주기성을 요약한 값이고 phase는 시간에 따라 변하는 파형이다. 둘은 같은 실수 배열로 저장될 수 있지만 정보의 차원과 물리적 의미가 다르다."),
        ("callout", "계약의 완성", "생산자는 unit·sample rate·filter·calibration·semantic version을 기록하고, 소비자는 manifest와 metadata를 기준으로 검증해야 한다. 파일 hash는 artifact를 고정하지만 신호 의미를 대신하지 않는다.", "purple"),
        ("summary", ["기계 계약은 key·shape·dtype처럼 자동 검사 가능한 구조를 고정한다.", "의미 계약은 단위·시간축·전처리·물리량의 뜻을 고정한다.", "실행 가능성과 올바른 해석은 별개의 조건이며 둘 다 만족해야 한다."]),
    ))

    pages.append(page(
        "07", "Sampling은 시간을 숫자열로 바꾼다", "sample 개수와 실제 측정 시간은 같지 않으므로 time grid를 명시해야 한다.",
        "PART I · 시스템 기초", "UNIT 03 · 1/2", "adapters/mmwave_stream_adapter.py:31-94; models/model_manifest.json:73-96",
        ("body", "연속시간 신호 x(t)를 일정 간격으로 읽으면 이산 신호 x[k]가 된다. 이상적인 sampling에서는 모든 시각이 첫 시각과 sample rate로 결정된다. 실제 stream에는 jitter, gap, 중복 timestamp가 생길 수 있으므로 N개의 값이 있다는 사실만으로 이상적인 시간 grid를 보장할 수 없다."),
        ("equation", r"t_k=t_0+\frac{k}{f_s},\qquad T_{\mathrm{cov}}=\frac{N-1}{f_s}", "1.2", "f_s는 초당 sample 수이고 T_cov는 첫 sample과 마지막 sample 사이의 실제 coverage다."),
        ("diagram", "timeline", {"markers": [(0.0, "k=0", "t₀"), (0.33, "k=100", "10.0 s"), (0.67, "k=200", "20.0 s"), (1.0, "k=299", "29.9 s")]}, 34 * mm, "10 Hz에서 300개 sample은 첫 점을 0으로 두면 마지막 점이 29.9초에 놓인다."),
        ("body", "현재 mmWave 계약은 f_s=10 Hz, N=300이다. 그러나 Stream Adapter는 buffer 길이와 큰 gap은 보지만 모든 sample이 0.1초 grid에 가까운지와 전체 coverage가 29.9초인지까지 강제하지 않는다. 시간 의미는 shape보다 강한 조건이다."),
    ))

    pages.append(page(
        "08", "Freshness는 판단 시점의 유효성을 정한다", "측정 시각과 도착 시각을 분리해야 오래된 값을 새 값처럼 쓰지 않는다.",
        "PART I · 시스템 기초", "UNIT 03 · 2/2", "adapters/mmwave_stream_adapter.py:84-110; integrated_node/safenest_risk_engine.py:120-169",
        ("body", "<b>Freshness</b>는 값이 존재하는지가 아니라 현재 판단에 사용할 만큼 최근인지 묻는다. 센서가 측정한 시각 measured_at과 process가 받은 시각 received_at이 다르면 transport 지연과 queue 지연을 분리할 수 있다. 현재 packet에 timestamp 하나만 있으면 이 두 시간을 구분하기 어렵다."),
        ("equation", r"a_i(t)=t-t^{\mathrm{meas}}_i,\qquad \mathrm{valid}_i(t)=[a_i(t)\leq B_i]", "1.3", "a_i는 i번째 sensor 값의 age, B_i는 그 sensor에 허용된 freshness budget이다."),
        ("diagram", "flow", {"labels": [("측정", "measured_at"), ("전송", "transport"), ("도착", "received_at"), ("대기", "queue"), ("판단", "evaluated_at")]}, 36 * mm, "전체 latency와 sensor age는 같은 숫자가 아니며 각각 기록해야 한다."),
        ("body", "mmWave Adapter는 window가 2초보다 오래되면 stale로 보지만 다른 modality에는 같은 수준의 per-sensor age 계약이 없다. 비동기 node로 확장하면 같은 packet에 묶였다는 이유만으로 동시에 측정됐다고 가정해서는 안 된다."),
        ("summary", ["Sampling은 연속시간 신호를 time grid 위의 숫자열로 바꾸는 과정이다.", "N개 sample과 N/f_s초라는 의미는 timestamp 품질이 보장될 때만 성립한다.", "Freshness는 판단 시점에서 각 sensor 값의 age가 허용 budget 안인지 검사하는 조건이다."]),
    ))

    pages.append(page(
        "09", "상태의 소유권과 생명주기", "buffer·filter·timer가 누구에게 속하고 언제 reset되는지가 시스템 의미를 결정한다.",
        "PART I · 시스템 기초", "UNIT 04 · 1/2", "integrated_node/safenest_risk_engine.py:73-96; risk/risk_rules.py:82-85",
        ("body", "<b>State owner</b>는 과거 정보를 보관하고 갱신하며 초기화할 책임이 있는 객체다. SafeNestRiskEngine instance 하나에는 mmWave buffer, rFFT history, CO₂ history, raw risk history, IIR state, 이전 status가 함께 있다. RiskRulesEvaluator는 apnea와 no-motion timer를 보관한다."),
        ("diagram", "state", {}, 43 * mm, "위험 상태 전이뿐 아니라 FAULT 진입과 복귀도 명시적인 lifecycle로 다뤄야 한다."),
        ("h2", "Reset은 단순한 메모리 삭제가 아니다"),
        ("body", "새 device나 새 session이 시작될 때 이전 사람의 30초 window와 CO₂ slope가 남아 있으면 현재 입력의 의미가 오염된다. process restart가 우연히 state를 비우는 것과, session boundary에서 의도적으로 reset하는 것은 다르다. 전자는 운영 사건이고 후자는 시스템 계약이다."),
        ("callout", "현재 구조의 함의", "한 RiskEngine instance에는 device/session별 key가 없다. 여러 stream을 같은 instance에 넣으면 history·timer·hysteresis가 섞이므로 instance-per-stream 또는 keyed StateStore가 필요하다.", "red"),
    ))

    pages.append(page(
        "10", "위험과 건강은 서로 다른 축이다", "관측 위험이 낮은 것과 그 판단을 신뢰할 수 있는 것은 같은 문장이 아니다.",
        "PART I · 시스템 기초", "UNIT 04 · 2/2", "risk/risk_rules.py:258-335; integrated_node/safenest_risk_engine.py:360-415",
        ("body", "위험도는 현재 관측된 증거가 얼마나 위험한지를 나타낸다. 시스템 건강은 그 증거를 생산한 센서·model·시간 경로가 얼마나 가용한지를 나타낸다. sensor가 모두 빠진 경우 risk_score가 0일 수 있지만 이는 안전 증거가 아니라 판단 가능한 관측이 없다는 뜻이다."),
        ("diagram", "fan_in", {"sources": ["호흡 증거", "CO₂ 증거", "심박 증거", "자세 증거", "움직임 증거"], "center": "위험 융합", "output": "RISK"}, 57 * mm, "부분 위험도는 하나의 RISK 축으로 합쳐지지만 health와 reason은 별도로 보존해야 한다."),
        ("body", "따라서 출력은 최소한 status_str, system_status, reasons의 세 축을 가져야 한다. NORMAL+DEGRADED는 낮은 위험 증거와 제한된 판단 범위가 동시에 존재한다는 뜻이다. DANGER+DEGRADED도 가능하며 응급 증거가 다른 경로의 고장을 지우지 않는다."),
        ("callout", "소비자의 해석 순서", "먼저 FAULT인지 확인하고, 다음으로 DEGRADED 정책을 적용한 뒤, 마지막에 risk level을 해석한다. risk_score 하나만 읽어 safe/unsafe를 결정하면 관측 불능을 안전으로 오해할 수 있다.", "amber"),
        ("summary", ["State는 특정 owner가 보관하며 device·session lifecycle에 따라 reset되어야 한다.", "위험도는 관측된 hazard의 크기이고 health는 판단 경로의 가용성이다.", "출력 소비자는 risk·health·reason을 함께 읽어야 한다."]),
    ))

    pages.append(page(
        "P2", "PART II · 센서와 물리적 측정", "센서는 사람의 상태를 직접 읽지 않고 특정 물리 현상을 제한된 방식으로 관측한다.",
        "PART II · 센서와 물리", "PART OPENER", "thermal_prep.py; integrated_node/virtual_sensor_streamer.py; mr60/sensor_receiver.py",
        ("body", "센서 데이터의 의미를 이해하려면 먼저 ‘무엇을 직접 측정하는가’를 물어야 한다. Thermal은 적외선 복사 분포, CO₂ sensor는 공간의 기체 농도, PIR은 적외선 flux의 시간 변화, mmWave radar는 송수신파의 주파수·위상 관계를 관측한다. 사람의 자세·재실·호흡은 이 관측을 해석한 결과다."),
        ("diagram", "flow", {"labels": [("물리 현상", "radiation·gas·motion"), ("센서 변환", "전기 신호"), ("보정", "offset·gain"), ("표현", "frame·ppm·phase"), ("추론", "상태 의미")]}, 40 * mm, "센서는 해석의 출발점이지 정답 자체가 아니다."),
        ("h2", "센서 이론에서 반드시 구분할 것"),
        ("bullets", ["측정 물리량과 software가 저장한 숫자 표현", "공간 해상도와 시간 해상도", "보정 가능한 systematic error와 random noise", "센서의 detection과 사람이 원하는 semantic label"]),
        ("callout", "이 장의 관점", "각 센서를 ‘입력값 목록’으로 설명하지 않고 에너지 또는 물질의 흐름, sampling, calibration, failure mode의 관점에서 설명한다.", "blue"),
    ))

    pages.append(page(
        "11", "Thermal sensor는 복사 분포를 본다", "온도와 적외선 intensity는 관련 있지만 동일한 데이터 표현은 아니다.",
        "PART II · 센서와 물리", "UNIT 05 · 1/2", "thermal_prep.py:19-130; inference/thermal_interpreter.py:106-148",
        ("body", "절대온도가 0 K보다 높은 물체는 전자기 복사를 방출한다. Thermal array는 특정 적외선 파장대에서 들어오는 복사 에너지를 pixel별 전기 신호로 바꾼다. 이 신호는 대상 온도뿐 아니라 방사율 emissivity, 반사된 주변 복사, 광학계, sensor response와 보정 상태의 영향을 함께 받는다."),
        ("equation", r"P_{\mathrm{rad}}\approx \varepsilon\sigma A\left(T^4-T_{\mathrm{env}}^4\right)", "2.1", "이 식은 복사 에너지와 절대온도의 관계를 보여주는 개념 모델이다. 실제 thermal camera는 제한된 파장대와 calibration model을 사용한다."),
        ("diagram", "flow", {"labels": [("대상", "T·emissivity"), ("적외선 복사", "incident power"), ("pixel array", "response"), ("NUC·보정", "offset·gain"), ("frame", "온도/강도 grid")]}, 38 * mm, "온도 grid를 얻으려면 pixel response를 물리 온도로 바꾸는 calibration chain이 필요하다."),
        ("body", "<b>NUC</b>(non-uniformity correction)는 같은 복사가 들어와도 pixel마다 다른 offset과 gain을 보이는 편차를 줄인다. 설치 방향과 FOV는 사람의 형상이 frame에서 차지하는 위치와 크기를 바꾸므로 model 입력 의미의 일부다."),
    ))

    pages.append(page(
        "12", "현재 Thermal 입력은 normalized intensity다", "저장소의 grayscale 학습 체인과 실제 온도 측정 체인을 구분한다.",
        "PART II · 센서와 물리", "UNIT 05 · 2/2", "thermal_prep.py:19-130; models/model_manifest.json:6-35; inference/thermal_interpreter.py:106-159",
        ("body", "현재 학습 원천은 PNG/JPEG를 grayscale로 읽고 80×62로 resize한 뒤 255로 나눈 영상이다. runtime manifest의 의미도 normalized_thermal_frame이다. 따라서 thermal_80x62 배열을 ‘섭씨 온도 4,960개’라고 설명하면 현재 artifact의 학습 의미와 맞지 않는다."),
        ("diagram", "flow", {"labels": [("이미지", "PNG/JPEG"), ("grayscale", "8-bit"), ("resize", "80×62"), ("정규화", "0…1"), ("2D CNN", "3 classes")]}, 37 * mm, "현재 경로는 radiometric temperature보다 영상 형상과 intensity contrast를 사용한다."),
        ("equation", r"x_{\mathrm{norm}}=\frac{x-x_{\min}}{x_{\max}-x_{\min}}", "2.2", "runtime에서 값이 0…1 범위를 벗어나면 한 frame의 최소·최대로 다시 정규화할 수 있다."),
        ("body", "Frame별 min-max는 절대 offset과 gain을 제거해 contrast에 강할 수 있지만 장면마다 숫자의 의미가 달라진다. Uniform frame에서는 분모가 작아지는 경계도 고려해야 한다. 실제 radiometric sensor를 연결하려면 온도 단위와 calibration을 보존하는 Adapter를 별도로 설계하는 편이 안전하다."),
        ("summary", ["Thermal array는 적외선 복사를 pixel response로 바꾸며 온도 외에도 방사율·반사·광학·보정의 영향을 받는다.", "현재 SafeNest 모델은 radiometric °C가 아니라 grayscale 기반 normalized frame을 학습했다.", "온도 sensor를 연결할 때는 기존 intensity semantic과 새 temperature semantic을 분리해야 한다."]),
    ))

    pages.append(page(
        "13", "CO₂ 농도는 발생과 환기의 균형이다", "같은 사람이 있어도 공간 부피와 환기량이 다르면 농도 변화가 달라진다.",
        "PART II · 센서와 물리", "UNIT 06 · 1/2", "integrated_node/safenest_risk_engine.py:92-94,195-230; risk/risk_rules.py:148-174",
        ("body", "공간을 하나의 완전 혼합 control volume으로 보면 CO₂ 농도 변화는 내부 발생량과 환기에 의한 유출입의 차이로 설명할 수 있다. 이 모델은 실제 방의 위치별 농도 차이를 생략하지만 농도와 slope를 해석하는 기본 인과관계를 제공한다."),
        ("diagram", "fan_in", {"sources": ["사람의 발생량 G", "외기 농도 C_out", "공간 부피 V", "환기 유량 Q_v"], "center": "공간 물질수지", "output": "C(t)"}, 54 * mm, "농도 C(t)는 사람의 존재만이 아니라 공간과 환기 조건의 함수다."),
        ("equation", r"V\frac{dC}{dt}=G-Q_v\left(C-C_{\mathrm{out}}\right)", "2.3", "C는 체적분율, G는 CO₂ 발생 부피유량, Q_v는 환기 유량, V는 공간 부피다."),
        ("body", "식의 왼쪽은 공간 안에 축적되는 CO₂이고 오른쪽 첫 항은 내부 발생, 둘째 항은 실내외 농도차에 따른 환기 제거다. ppm은 체적분율에 10⁶을 곱한 표현이므로 농도 차와 slope의 단위를 일관되게 유지해야 한다."),
    ))

    pages.append(page(
        "14", "CO₂의 시간응답은 1차 시스템과 닮아 있다", "환기가 강할수록 정상상태 농도와 도달 시간이 함께 낮아진다.",
        "PART II · 센서와 물리", "UNIT 06 · 2/2", "integrated_node/safenest_risk_engine.py:203-230; risk/risk_config.json:12-16",
        ("body", "G, Q_v, V, C_out이 일정하다고 두면 농도는 정상상태를 향해 지수적으로 접근한다. 시간상수는 공간 공기가 환기로 교체되는 시간척도이며, 같은 농도 변화율도 window 길이와 sensor update rate가 다르면 다른 feature가 된다."),
        ("equation", r"C(t)=C_{\mathrm{ss}}+\left(C_0-C_{\mathrm{ss}}\right)e^{-t/\tau},\quad C_{\mathrm{ss}}=C_{\mathrm{out}}+\frac{G}{Q_v},\quad \tau=\frac{V}{Q_v}", "2.4", "C_ss는 정상상태 농도이고 τ는 공간 부피를 환기 유량으로 나눈 시간상수다."),
        ("diagram", "plot", {"plot": "co2"}, 45 * mm, "환기가 강하면 더 낮은 정상상태에 더 빨리 접근하고, 환기가 약하면 농도가 더 높게 축적된다."),
        ("body", "현재 엔진은 최대 30개의 timestamp·ppm history에서 양 끝점 slope를 계산한다. 이는 ‘30분’이나 ‘5분’이 아니라 호출 주기에 따라 달라지는 30 samples다. CO₂ 1000/2500 ppm과 slope 15 ppm/min은 정책값이며, occupancy AI의 class와 전역 위험 override는 별개다."),
        ("summary", ["CO₂ 농도는 발생량·환기량·공간 부피·외기 농도의 물질수지로 설명된다.", "시간상수 V/Q_v는 농도가 얼마나 빨리 정상상태로 가는지 결정한다.", "CO₂ slope는 window의 시간척도를 포함한 feature이므로 sample 수만 같다고 같은 의미가 아니다."]),
    ))

    pages.append(page(
        "15", "PIR은 절대 위치보다 변화를 감지한다", "Pyroelectric element는 적외선 flux의 시간 변화에 반응한다.",
        "PART II · 센서와 물리", "UNIT 07 · 1/2", "integrated_node/virtual_sensor_streamer.py:117-159; risk/risk_rules.py:213-256",
        ("body", "PIR(passive infrared) sensor는 스스로 빛을 내지 않고 주변에서 들어오는 적외선 복사의 변화를 감지한다. 사람과 배경의 복사 차이가 Fresnel lens의 여러 zone을 지나며 pyroelectric element에 시간 변화로 나타난다. 따라서 움직이지 않는 사람은 신호가 작아질 수 있다."),
        ("equation", r"v_{\mathrm{PIR}}(t)\propto \frac{d\Phi_{\mathrm{IR}}(t)}{dt}", "2.5", "Φ_IR은 sensor에 들어오는 적외선 flux다. PIR 출력은 절대 복사량보다 변화율에 민감하다는 개념을 나타낸다."),
        ("diagram", "flow", {"labels": [("사람 이동", "IR contrast"), ("Fresnel zone", "공간 변조"), ("dual element", "차동 응답"), ("증폭·필터", "pulse"), ("motion", "0/1 event")]}, 38 * mm, "PIR의 motion=0은 ‘사람 없음’이 아니라 최근 검출 가능한 변화가 없다는 뜻에 가깝다."),
        ("body", "Detection 범위는 lens pattern, 설치 높이와 방향, 대상 속도, 배경 온도에 의존한다. 정지 재실을 판단하려면 PIR만으로 충분하지 않으며 mmWave presence 같은 다른 관측과 결합해야 한다."),
    ))

    pages.append(page(
        "16", "No-motion은 presence와 함께 해석한다", "움직임이 없다는 증거가 위험 의미를 가지려면 사람이 있다는 조건이 먼저 필요하다.",
        "PART II · 센서와 물리", "UNIT 07 · 2/2", "risk/risk_rules.py:213-256; integrated_node/safenest_risk_engine.py:306-315",
        ("body", "SafeNest의 no-motion timer는 PIR motion=0만으로 시작하지 않는다. mmWave에서 presence가 확인된 상태에서 motion=0이 지속될 때만 의미 있는 정지 상태로 누적한다. Presence가 false이거나 motion이 다시 1이 되면 timer를 reset한다."),
        ("diagram", "timeline", {"markers": [(0.0, "presence=1", "motion=0"), (0.33, "5 s", "누적"), (0.67, "10 s", "누적"), (1.0, "15 s", "LONG_NO_MOTION")]}, 36 * mm, "15초는 물리 상수가 아니라 현재 provisional policy의 지속시간 문턱이다."),
        ("body", "여기서 presence는 여러 센서의 fusion 결과가 아니라 현재 코드의 mmWave presence 하나다. 따라서 radar blind spot이나 stale presence가 있으면 no-motion 의미도 함께 약해진다. ‘정지=위험’이 아니라 ‘presence가 있는 상태에서 관측 가능한 움직임이 일정 시간 없다’가 정확한 설명이다."),
        ("callout", "센서 융합의 원리", "서로 다른 failure mode를 가진 센서를 결합하면 단일 센서보다 의미가 강해질 수 있다. 다만 한 센서의 조건을 다른 센서의 ground truth처럼 사용하면 그 센서의 오류도 함께 전파된다.", "amber"),
        ("summary", ["PIR은 절대 위치가 아니라 적외선 flux의 시간 변화를 감지한다.", "motion=0은 사람 부재가 아니라 검출 가능한 움직임이 없다는 관측이다.", "SafeNest는 mmWave presence가 있을 때만 no-motion 시간을 위험 증거로 누적한다."]),
    ))

    pages.append(page(
        "17", "FMCW radar는 beat frequency로 거리를 분리한다", "주파수 sweep과 지연된 반사파의 차이가 target range 정보가 된다.",
        "PART II · 센서와 물리", "UNIT 08 · 1/4", "mr60/sensor_receiver.py; docs/roadmap_and_setup/safenest_mmwave_latest_development_direction_20260726.md",
        ("body", "FMCW(frequency-modulated continuous wave) radar는 chirp 동안 송신 주파수를 거의 선형으로 변화시킨다. target에서 반사된 파는 왕복 지연 τ만큼 늦게 도착한다. 현재 송신파와 지연된 수신파를 mixer에서 곱하면 두 주파수의 차이인 beat frequency가 생긴다."),
        ("equation", r"f_{\mathrm{tx}}(t)=f_0+St,\qquad S=\frac{B}{T_c}", "2.6", "B는 chirp bandwidth, T_c는 chirp duration, S는 주파수 sweep slope다."),
        ("equation", r"f_b\approx S\tau,\qquad R=\frac{c\tau}{2}=\frac{cf_b}{2S}", "2.7", "왕복 시간 때문에 거리 R에는 1/2가 들어간다. 정지 target에서는 beat frequency가 주로 range를 나타낸다."),
        ("diagram", "flow", {"labels": [("chirp 송신", "slope S"), ("target 반사", "delay τ"), ("수신", "delayed chirp"), ("mix", "difference"), ("beat", "f_b→range")]}, 36 * mm, "시간 지연을 직접 재는 대신 주파수 차이로 바꾸어 거리를 분리한다."),
    ))

    pages.append(page(
        "18", "Range FFT는 여러 거리를 bin으로 분리한다", "한 chirp 안의 ADC sample을 주파수축으로 바꾸면 target 거리가 분리된다.",
        "PART II · 센서와 물리", "UNIT 08 · 2/4", "integrated_node/safenest_risk_engine.py:17-71; mr60/sensor_receiver.py",
        ("body", "Mixer 출력에는 여러 target의 beat tone이 함께 들어 있다. Chirp 내부의 fast-time ADC sample에 FFT를 적용하면 서로 다른 beat frequency가 서로 다른 bin에 나타난다. Calibration이 맞으면 각 frequency bin을 거리 bin으로 mapping할 수 있다."),
        ("equation", r"X[m]=\sum_{n=0}^{N_r-1}x[n]e^{-j2\pi mn/N_r}", "2.8", "x[n]은 chirp 안의 ADC 또는 complex baseband sample이고 X[m]은 m번째 beat-frequency bin이다."),
        ("diagram", "flow", {"labels": [("ADC samples", "fast time"), ("window", "leakage 억제"), ("range FFT", "frequency bins"), ("calibration", "f_b→R"), ("range profile", "complex X[m]")]}, 38 * mm, "Range profile의 각 bin은 크기와 위상을 가진 complex 값이다."),
        ("body", "거리 해상도는 이상적으로 bandwidth가 클수록 좋아져 ΔR≈c/(2B)로 표현된다. 그러나 실제 탐지 성능은 window, SNR, multipath, antenna geometry, calibration에도 의존한다. 현재 저장소의 일부 rFFT는 실제 sensor raw가 아니라 시각화용 합성 신호이므로 실측 radar DSP와 구분해야 한다."),
        ("summary", ["FMCW는 지연된 chirp와 현재 chirp의 beat frequency를 이용해 거리를 계산한다.", "Range FFT는 여러 beat tone을 분리해 complex range bin을 만든다.", "거리 bin은 target 자체가 아니라 특정 거리 구간의 반사 신호 표현이다."]),
    ))

    pages.append(page(
        "19", "위상은 파장보다 작은 변위에 민감하다", "같은 range bin 안의 미세한 흉부 운동은 complex phase 변화로 나타난다.",
        "PART II · 센서와 물리", "UNIT 08 · 3/4", "models/mmwave/sensor_stats_metadata_v0.1.0.json; inference/mmwave_interpreter.py:159-198",
        ("body", "호흡에 의한 흉부 변위는 range resolution보다 작을 수 있어 bin index가 바뀌지 않는다. 그러나 왕복 경로 길이는 미세하게 변하고 complex signal의 phase가 이동한다. Radar는 이 phase modulation을 이용해 sub-bin displacement를 관측할 수 있다."),
        ("equation", r"\Delta\phi=\frac{4\pi}{\lambda}\Delta R,\qquad \Delta R=\frac{\lambda}{4\pi}\Delta\phi", "2.9", "전자파가 target까지 갔다가 돌아오므로 경로 변화가 2ΔR이고 phase 식에 4π가 나타난다."),
        ("diagram", "plot", {"plot": "phase"}, 44 * mm, "주기적인 흉부 변위가 시간에 따른 phase 파형으로 표현된다."),
        ("body", "측정 phase는 보통 -π와 π 사이에서 wrap된다. 연속 운동을 복원하려면 ±π 경계를 지날 때 2π를 보정하는 phase unwrap이 필요하다. 작은 noise와 target switching도 unwrap error를 만들 수 있으므로 range-bin tracking과 signal quality가 중요하다."),
    ))

    pages.append(page(
        "20", "호흡 입력은 긴 신호 체인의 결과다", "Resp_phase 300개는 radar raw에서 여러 선택과 보정을 거친 canonical signal이다.",
        "PART II · 센서와 물리", "UNIT 08 · 4/4", "models/mmwave/sensor_stats_metadata_v0.1.0.json; adapters/mmwave_stream_adapter.py; models/model_manifest.json",
        ("diagram", "flow", {"labels": [("range FFT", "complex bins"), ("clutter 제거", "background"), ("chest bin", "target"), ("phase", "angle"), ("unwrap", "continuous"), ("10 Hz", "resample"), ("window", "300×1")]}, 39 * mm, "Model input은 마지막 300개 숫자만이 아니라 이 전체 처리 의미를 포함한다."),
        ("body", "현재 metadata의 input semantic은 <b>resp_phase_unwrapped_clutter_removed</b>이고 sample rate는 10 Hz, window는 300 samples다. 따라서 producer가 phase를 제공할 때는 rad 단위, unwrap 방식, clutter 제거 방식, sample clock과 session 경계를 함께 보장해야 한다."),
        ("body", "호흡 rate는 phase 파형의 주기성을 요약한 값이고 resp_phase는 시간 신호다. 정상 RPM이 들어온다는 사실만으로 model window가 준비된 것은 아니다. 반대로 model class는 특정 30초 파형의 분류 결과이지 현재 한 순간의 물리적 호흡량이 아니다."),
        ("callout", "현재 연결 경계", "독립 MR60 receiver, 표시용 rFFT, 통합 Engine의 resp_phase 입력은 완전히 같은 실측 chain으로 연결돼 있지 않다. 이론상 필요한 chain과 현재 연결 상태를 구분해야 한다.", "red"),
        ("summary", ["미세 변위는 complex phase 변화로 나타나며 Δφ=4πΔR/λ로 연결된다.", "호흡 model 입력은 clutter 제거·chest-bin 선택·phase unwrap·10 Hz sampling을 거친 시간 신호다.", "RPM·phase·AI class는 서로 다른 추상화 계층의 값이므로 바꾸어 쓸 수 없다."]),
    ))

    pages.append(page(
        "P3", "PART III · Sampling과 신호처리", "신호처리는 noise를 줄이는 대신 정보 손실·지연·상태 기억을 만든다.",
        "PART III · DSP와 시간", "PART OPENER", "adapters/mmwave_stream_adapter.py; integrated_node/safenest_risk_engine.py:17-71,326-344",
        ("body", "Digital signal processing은 센서의 연속 물리 현상을 유한한 sample과 제한된 계산으로 해석하는 과정이다. Sampling rate, window 길이, FFT bin, filter coefficient는 단순한 구현 숫자가 아니라 관측 가능한 현상과 응답시간을 결정한다."),
        ("diagram", "layers", {"labels": [("Sampling", "연속→이산"), ("Window", "관측 구간"), ("주파수 변환", "DFT·band energy"), ("배경 제거", "clutter"), ("시간 필터", "mean·IIR"), ("상태 안정화", "hysteresis") ]}, 76 * mm, "각 단계는 noise를 줄이지만 동시에 latency와 memory를 추가한다."),
        ("h2", "이 장의 중심 원리"),
        ("body", "신호처리 parameter는 독립적으로 선택할 수 없다. 긴 window는 주파수 분해능을 높이지만 startup latency와 변화 추적 지연을 늘린다. 작은 IIR α는 출력 변동을 줄이지만 경보 진입과 해제를 늦춘다. 좋은 설계는 noise와 latency의 trade-off를 요구사항에 맞춰 설명한다."),
        ("summary", ["DSP parameter는 계산 편의가 아니라 관측 가능 범위와 시간 응답을 결정한다.", "Noise 억제와 빠른 경보는 일반적으로 trade-off 관계다."]),
    ))

    pages.append(page(
        "21", "Sampling theorem이 관측 가능한 주파수를 정한다", "Sample rate보다 빠르게 변하는 성분은 원래 주파수로 구분되지 않는다.",
        "PART III · DSP와 시간", "UNIT 09 · 1/2", "models/model_manifest.json:73-96; adapters/mmwave_stream_adapter.py:31-47",
        ("body", "연속시간 sinusoid를 f_s로 sampling하면 주파수축이 f_s 간격으로 반복된다. 원 신호의 spectrum이 반복된 spectrum과 겹치면 높은 주파수 성분이 낮은 주파수처럼 보이는 aliasing이 생긴다. 대역 제한 신호를 손실 없이 복원하려면 가장 높은 유효 주파수가 Nyquist frequency보다 작아야 한다."),
        ("equation", r"f_{\mathrm{signal,max}}<f_N=\frac{f_s}{2}", "3.1", "f_N은 Nyquist frequency다. 실제 시스템은 경계 밖 성분을 줄이는 anti-alias filter와 여유 대역이 필요하다."),
        ("equation", r"f_{\mathrm{alias}}=\left|f_0-kf_s\right|\quad\mathrm{for\ a\ suitable\ integer}\ k", "3.2", "f_s를 넘는 주파수는 sampling 후 더 낮은 apparent frequency로 접혀 보일 수 있다."),
        ("body", "SafeNest mmWave phase는 10 Hz로 계약되어 Nyquist frequency가 5 Hz다. 호흡의 주파수 대역은 이보다 훨씬 낮지만 sensor 내부의 고주파 noise나 motion 성분이 충분히 억제되지 않으면 낮은 대역 해석에 영향을 줄 수 있다."),
        ("callout", "Sample rate의 의미", "10 Hz는 초당 10개의 배열 원소가 있다는 말이 아니라 측정 시각이 평균 0.1초 간격이며 jitter·gap·clock drift가 허용 범위 안이라는 시간 계약이다.", "blue"),
    ))

    pages.append(page(
        "22", "Window는 무엇을 한 번의 판단으로 묶을지 정한다", "길이는 주파수 분해능과 startup latency, stride는 판단 갱신 간격을 결정한다.",
        "PART III · DSP와 시간", "UNIT 09 · 2/2", "adapters/mmwave_csv_adapter.py:40-120; adapters/mmwave_stream_adapter.py:31-94",
        ("body", "Window는 연속 stream에서 유한 구간을 잘라 하나의 feature 또는 model input으로 만드는 연산이다. N과 f_s가 정해지면 관측 구간과 DFT frequency spacing이 함께 정해진다. Window를 길게 하면 가까운 주파수를 구분하기 쉬워지지만 첫 판단까지 기다리는 시간이 늘어난다."),
        ("equation", r"T_{\mathrm{window}}\approx\frac{N}{f_s},\qquad \Delta f=\frac{f_s}{N}", "3.3", "N=300, f_s=10 Hz이면 약 30초 window이고 frequency spacing은 0.0333 Hz다."),
        ("diagram", "timeline", {"markers": [(0.0, "첫 sample", "buffer 1/300"), (0.34, "10 s", "101/300"), (0.67, "20 s", "201/300"), (1.0, "29.9 s", "300/300") ]}, 36 * mm, "첫 AI invoke는 이상적인 10 Hz grid에서 약 29.9초 뒤 가능하다."),
        ("body", "Rolling window를 매 sample 갱신하면 연속 두 window는 299/300을 공유해 99.67%가 겹친다. 출력 갱신은 0.1초마다 가능하지만 독립 정보가 0.1초마다 새로 생기는 것은 아니다. Stride와 overlap은 계산량·상관성·응답성을 함께 결정한다."),
        ("summary", ["Sampling rate는 관측 가능한 주파수 범위와 시간 grid를 정한다.", "Window 길이는 frequency spacing과 startup latency를 동시에 결정한다.", "많이 겹치는 rolling window의 연속 출력은 강하게 상관되어 있다."]),
    ))

    pages.append(page(
        "23", "DFT는 시간 파형을 주파수 성분으로 분해한다", "각 bin은 window 안에서 특정 주파수의 complex sinusoid와 얼마나 닮았는지 나타낸다.",
        "PART III · DSP와 시간", "UNIT 10 · 1/2", "integrated_node/safenest_risk_engine.py:54-71; numpy.fft 사용 경로",
        ("body", "이산 Fourier 변환 DFT는 N개의 시간 sample을 N개의 complex frequency coefficient로 바꾼다. 각 coefficient의 magnitude는 해당 sinusoid 성분의 크기와 관련되고 phase는 시간 정렬 정보를 담는다. 실수 신호에는 양·음 주파수 대칭이 있어 rFFT는 비음수 주파수 절반만 계산한다."),
        ("equation", r"X[m]=\sum_{n=0}^{N-1}x[n]e^{-j2\pi mn/N},\qquad f_m=\frac{m}{N}f_s", "3.4", "m번째 bin의 중심 주파수는 m f_s/N이다."),
        ("diagram", "plot", {"plot": "spectrum"}, 45 * mm, "유한 window에서는 연속 spectrum을 Δf 간격의 bin으로 관측한다."),
        ("body", "Window 안에 정수 개 주기가 들어맞지 않으면 한 tone의 energy가 여러 bin으로 퍼지는 spectral leakage가 생긴다. Hann 같은 window function은 leakage를 줄이지만 main-lobe 폭을 넓혀 분해능에 영향을 준다. 따라서 FFT 결과는 항상 N, f_s, window function과 함께 해석해야 한다."),
    ))

    pages.append(page(
        "24", "Band energy는 주파수 대역의 활동량을 요약한다", "PSD와 단순 FFT 제곱합은 normalization과 단위가 다르다.",
        "PART III · DSP와 시간", "UNIT 10 · 2/2", "integrated_node/safenest_risk_engine.py:54-71",
        ("body", "호흡처럼 관심 주파수 범위가 알려진 경우 특정 bin들의 magnitude squared를 합해 대역 activity를 비교할 수 있다. 이 값은 target bin을 선택하는 proxy로 유용하지만 sampling rate, window 길이, window function, FFT normalization이 없으면 물리 단위를 가진 power spectral density라고 부르기 어렵다."),
        ("equation", r"E_B=\sum_{m:\,f_m\in B}\left|X[m]\right|^2", "3.5", "B는 관심 주파수 대역이고 E_B는 해당 bin들의 비정규화 energy proxy다."),
        ("body", "현재 adaptive chest-bin 경로는 10 Hz에서 rFFT history 30개를 사용하므로 시간 window가 약 3초이고 Δf는 약 0.333 Hz다. 0.1–0.5 Hz 대역 안에 매우 적은 bin만 들어와 호흡 대역의 세밀한 구조를 표현하기 어렵다. 30초 model window의 0.0333 Hz spacing과 구분해야 한다."),
        ("callout", "수식 이름의 정확성", "|FFT|²을 합했다는 이유만으로 PSD라고 부르지 않는다. PSD는 window energy, sample rate, bin width에 대한 normalization을 포함해 단위 주파수당 power를 나타낸다.", "purple"),
        ("summary", ["DFT는 유한한 시간 신호를 Δf=f_s/N 간격의 complex frequency bin으로 바꾼다.", "Spectral leakage와 window function은 bin energy 분포를 바꾼다.", "현재 chest-bin 지표는 정규화 PSD가 아니라 호흡 대역 FFT 제곱합 proxy다."]),
    ))

    pages.append(page(
        "25", "Clutter 제거는 정적 반사를 배경으로 분리한다", "벽·가구처럼 지속되는 반사를 추정해 움직이는 target의 변화를 강조한다.",
        "PART III · DSP와 시간", "UNIT 11 · 1/2", "integrated_node/safenest_risk_engine.py:25-46",
        ("body", "Radar range profile에는 사람뿐 아니라 벽, 바닥, 좌석, sensor enclosure의 반사가 함께 들어 있다. 이 중 시간에 따라 거의 변하지 않는 성분을 clutter 또는 background로 보고 여러 frame의 평균으로 추정할 수 있다. 현재 frame에서 background를 빼면 변화 성분이 강조된다."),
        ("equation", r"C[b]=\frac{1}{K}\sum_{k=0}^{K-1}X_k[b],\qquad \widetilde{X}_k[b]=X_k[b]-C[b]", "3.6", "C[b]는 b번째 range bin의 complex clutter estimate이고 X̃는 background-subtracted signal이다."),
        ("diagram", "flow", {"labels": [("range profile", "X_k[b]"), ("K-frame 평균", "C[b]"), ("complex subtraction", "X-C"), ("변화 성분", "X tilde"), ("target 분석", "energy·phase")]}, 37 * mm, "Magnitude가 아니라 complex 값을 빼야 amplitude와 phase background를 함께 제거할 수 있다."),
        ("body", "Calibration 중 사람이 움직이면 그 일부가 background에 흡수될 수 있다. 반대로 환경이 변했는데 clutter map을 갱신하지 않으면 residual이 커진다. 따라서 calibration은 ‘처음 K frame 평균’이라는 계산뿐 아니라 공간이 비어 있다는 조건, 갱신 정책, version과 reset 조건을 포함해야 한다."),
    ))

    pages.append(page(
        "26", "Chest bin 선택은 target 위치를 추정하는 문제다", "여러 거리 중 호흡 대역 활동이 가장 뚜렷한 bin을 선택한다.",
        "PART III · DSP와 시간", "UNIT 11 · 2/2", "integrated_node/safenest_risk_engine.py:48-71,283-292",
        ("body", "사람의 흉부 반사가 들어 있는 거리 bin을 알면 그 bin의 phase를 시간에 따라 추적할 수 있다. Adaptive 방식은 각 거리 bin의 시간 history를 FFT하고 호흡 대역 energy를 계산해 가장 큰 bin을 선택한다. 이는 거리 amplitude가 가장 큰 bin과 다를 수 있다."),
        ("equation", r"b^*=\underset{b}{\arg\max}\;\sum_{m:\,f_m\in B}\left|\mathcal{F}_t\left\{\widetilde{X}_t[b]\right\}[m]\right|^2", "3.7", "각 range bin의 시간 변화에서 호흡 대역 energy가 최대인 b*를 chest bin으로 선택한다."),
        ("diagram", "flow", {"labels": [("range bins", "b=0…B-1"), ("시간 history", "X_t[b]"), ("time FFT", "per bin"), ("band energy", "0.1…0.5 Hz"), ("argmax", "chest bin") ]}, 38 * mm, "거리축 FFT와 시간축 FFT는 서로 다른 차원에 적용되는 연산이다."),
        ("body", "사람이 움직여 range bin이 바뀌면 단순 argmax가 frame마다 튈 수 있다. Tracking에는 거리 연속성, SNR, presence, hold time이 필요하다. 현재 표시용 rFFT 경로와 model의 resp_phase producer가 완전히 연결되지 않았으므로 chest bin telemetry가 곧 model input source라는 뜻은 아니다."),
        ("summary", ["Clutter map은 정적 complex 반사를 추정하고 subtraction은 변화 성분을 강조한다.", "Chest bin은 각 거리 bin의 시간 history에서 호흡 대역 energy가 가장 큰 위치다.", "Calibration 조건과 target tracking이 없으면 background와 chest 위치가 쉽게 오염될 수 있다."]),
    ))

    pages.append(page(
        "27", "Moving mean은 짧은 변동을 평균낸다", "최근 값을 같은 가중치로 묶어 noise를 줄이지만 변화의 시작을 퍼뜨린다.",
        "PART III · DSP와 시간", "UNIT 12 · 1/4", "integrated_node/safenest_risk_engine.py:93-95,326-336",
        ("body", "N-point moving mean은 최근 N개의 입력을 같은 가중치로 평균한다. 독립적인 zero-mean noise에는 variance 감소 효과가 있지만 step 변화는 N samples에 걸쳐 점진적으로 나타난다. Window가 완전히 차기 전에는 사용 가능한 sample 수만 평균해 startup 응답이 정상상태와 다르다."),
        ("equation", r"M[k]=\frac{1}{N_m}\sum_{i=0}^{N_m-1}r[k-i],\qquad N_m=6", "3.8", "현재 엔진은 최대 6개의 raw risk를 평균한다. 시작 직후에는 deque에 들어 있는 값만 사용한다."),
        ("diagram", "flow", {"labels": [("raw risk", "r[k]"), ("6-sample buffer", "history"), ("동일 가중 평균", "1/6"), ("M[k]", "smoothed input") ]}, 36 * mm, "한 개의 spike는 여섯 출력에 걸쳐 영향을 남길 수 있다."),
        ("body", "Moving mean은 memory가 유한한 FIR filter다. N_m이 커지면 noise는 더 줄지만 급격한 위험 증가와 감소가 더 천천히 보인다. 시스템의 packet rate가 바뀌면 같은 6 samples도 물리 시간 길이가 달라지므로 filter 의미를 시간 단위로 함께 설명해야 한다."),
    ))

    pages.append(page(
        "28", "IIR은 과거 출력을 재귀적으로 기억한다", "현재 값과 이전 filter state를 섞어 지수형 memory를 만든다.",
        "PART III · DSP와 시간", "UNIT 12 · 2/4", "integrated_node/safenest_risk_engine.py:333-336",
        ("body", "1차 IIR low-pass는 현재 입력 M[k]와 이전 출력 R[k-1]을 가중합한다. α가 클수록 새 입력을 빠르게 반영하고, α가 작을수록 과거 출력이 오래 남는다. Moving mean 뒤에 IIR이 이어지면 두 단계의 smoothing delay가 누적된다."),
        ("equation", r"R[k]=(1-\alpha)R[k-1]+\alpha M[k],\qquad \alpha=0.25", "3.9", "현재 엔진은 새 moving mean을 25%, 이전 filtered risk를 75% 반영한다."),
        ("equation", r"\tau_{\mathrm{packet}}\approx-\frac{1}{\ln(1-\alpha)}=3.48", "3.10", "IIR 단독 step response가 약 63%에 도달하는 시간척도를 packet 수로 근사한 값이다."),
        ("diagram", "plot", {"plot": "lowpass"}, 43 * mm, "지수형 응답은 noise를 완화하지만 threshold 도달 시점을 늦춘다."),
        ("body", "Packet period가 0.1초라면 3.48 packets는 약 0.35초지만 이는 IIR 단독 근사다. 앞의 6-point mean, model window, timer와 transport latency가 함께 있으므로 전체 경보 지연은 이 숫자보다 크다."),
    ))

    pages.append(page(
        "29", "Hysteresis는 상태 경계의 왕복을 막는다", "진입과 해제 문턱을 다르게 두어 threshold 근처의 flicker를 줄인다.",
        "PART III · DSP와 시간", "UNIT 12 · 3/4", "integrated_node/safenest_risk_engine.py:338-344; risk/risk_config.json:18-28",
        ("body", "하나의 threshold만 사용하면 noise가 경계 위아래를 오갈 때 상태가 빠르게 전환된다. Hysteresis는 위험 상태로 들어가는 문턱과 빠져나오는 문턱을 다르게 둬 상태에 memory를 부여한다. 그래서 현재 위험도뿐 아니라 이전 상태가 다음 상태를 결정한다."),
        ("diagram", "plot", {"plot": "hysteresis"}, 47 * mm, "높은 진입 문턱과 낮은 해제 문턱 사이가 상태 유지 영역이다."),
        ("equation", r"R_{\mathrm{off}}<R_{\mathrm{on}}", "3.11", "상태가 위험으로 진입하는 R_on보다 해제되는 R_off를 낮게 두어 안정적인 전이를 만든다."),
        ("body", "현재 엔진은 일반 진입에 40/75를 사용하고 이전 상태가 DANGER이면 65/35를 사용해 복귀한다. Config의 normal_max=35와 engine의 일반 CAUTION 진입 40이 다르므로 이론상 하나의 policy source로 통일해야 문턱 의미를 설명할 수 있다."),
        ("summary", ["Moving mean은 최근 raw risk를 동일 가중 평균하는 유한 memory filter다.", "IIR은 이전 출력 state를 재귀적으로 사용해 지수형 memory를 만든다.", "Hysteresis는 진입과 해제 문턱을 분리해 threshold 근처 상태 flicker를 줄인다."]),
    ))

    pages.append(page(
        "30", "경보 latency는 여러 시간의 합이다", "Model 실행시간 하나만으로 sensor-to-alarm 요구를 설명할 수 없다.",
        "PART III · DSP와 시간", "UNIT 12 · 4/4", "integrated_node/safenest_risk_engine.py:242-344; risk/risk_rules.py:87-146",
        ("body", "End-to-end latency는 물리 현상이 생긴 시점부터 실제 actuator나 UI가 반응할 때까지의 시간이다. Sensor update, packet transport, window startup, model compute, 지속시간 확인, filter·hysteresis, output 전달이 모두 포함된다."),
        ("equation", r"T_{\mathrm{alarm}}=T_{\mathrm{sample}}+T_{\mathrm{transport}}+T_{\mathrm{window}}+T_{\mathrm{compute}}+T_{\mathrm{confirm}}+T_{\mathrm{actuate}}", "3.12", "각 항은 별도의 timestamp로 관측하거나 upper bound를 정해야 한다."),
        ("diagram", "timeline", {"markers": [(0.0, "현상 발생", "physical"), (0.18, "sensor", "sample"), (0.38, "window", "ready"), (0.58, "inference", "class"), (0.80, "confirm", "timer"), (1.0, "alarm", "actuate") ]}, 38 * mm, "병렬 경로마다 window와 confirmation이 달라 earliest alarm time도 달라진다."),
        ("body", "Thermal FALL은 단일 frame에서 즉시 override될 수 있고 upstream apnea flag도 valid time 경로에서 즉시다. RPM 또는 mmWave class2 candidate는 2초 확인을 사용하며 mmWave AI는 그 전에 약 30초 window startup이 필요하다. ‘2초 경보’라는 문장은 어느 시작점부터 재는지 정의해야 한다."),
        ("callout", "설계 원칙", "Noise를 줄이는 filter와 debounce는 안전에 필요할 수 있지만 latency budget을 소비한다. 각 단계의 지연을 숨기지 말고 hazard별 허용시간 안에서 배분해야 한다.", "red"),
        ("summary", ["경보 latency는 sampling·transport·window·compute·confirm·actuation 시간을 합한 end-to-end 속성이다.", "각 sensor와 판단 경로는 서로 다른 startup과 confirmation 시간을 가진다.", "필터의 안정성과 경보의 신속성은 하나의 latency budget 안에서 함께 설계해야 한다."]),
    ))

    pages.append(page(
        "P4", "PART IV · Edge AI의 입력과 출력", "Model은 물리계를 직접 보지 않고 계약된 tensor 표현을 계산한다.",
        "PART IV · Edge AI", "PART OPENER", "models/model_manifest.json; inference/*_interpreter.py",
        ("body", "On-device AI를 이해할 때 architecture 이름보다 중요한 것은 입력 semantic, preprocessing, tensor contract와 출력 해석이다. Model은 sensor 자체가 아니라 정해진 feature 또는 frame 배열을 입력받는다. 따라서 upstream 변환이 학습 때와 같아야 model class가 같은 뜻을 가진다."),
        ("diagram", "flow", {"labels": [("센서값", "physical unit"), ("feature", "selected meaning"), ("정규화", "training scale"), ("양자화", "int8 grid"), ("TFLite", "graph invoke"), ("출력", "class score") ]}, 40 * mm, "각 화살표는 model 성능의 일부이며 wrapper와 metadata가 함께 계약해야 한다."),
        ("h2", "이 장에서 분리할 개념"),
        ("bullets", ["Feature와 raw sensor value", "표준화와 INT8 양자화", "Tensor shape와 tensor semantic", "Class score와 실제 확률·정확도", "Model inference와 최종 safety policy"]),
        ("callout", "AI의 위치", "AI는 전체 시스템의 한 component다. 현재 SafeNest의 최종 위험도는 AI class뿐 아니라 sensor rule, timer, override, filter와 health propagation에 의해 결정된다.", "blue"),
    ))

    pages.append(page(
        "31", "Feature는 물리 신호에서 선택한 표현이다", "Model이 학습한 것은 sensor 전체가 아니라 특정 변환을 거친 입력 공간이다.",
        "PART IV · Edge AI", "UNIT 13 · 1/2", "models/model_manifest.json; inference/co2_interpreter.py:80-95; inference/mmwave_interpreter.py:159-198",
        ("body", "<b>Feature</b>는 raw data에서 판단에 도움이 된다고 선택하거나 계산한 표현이다. CO₂ model은 slope·humidity·ppm 세 값을 사용하고, mmWave model은 300-sample resp_phase window를 사용한다. Thermal model은 spatial frame 자체에서 feature를 내부적으로 학습한다."),
        ("equation", r"\mathbf{u}=h\!\left(x_{\mathrm{raw}},\,t,\,c\right)", "4.1", "h는 calibration·filter·window·geometry를 포함한 feature 변환이고 c는 그 변환에 필요한 context다."),
        ("diagram", "flow", {"labels": [("raw", "sensor data"), ("calibration", "unit·offset"), ("window", "time context"), ("feature h", "selected info"), ("tensor", "fixed shape") ]}, 38 * mm, "Feature engineering은 정보 선택이므로 무엇을 버렸는지도 설명해야 한다."),
        ("body", "Feature가 같은 이름을 가져도 계산 window와 단위가 다르면 다른 입력이다. 예를 들어 CO₂ slope는 ppm/min이라는 단위뿐 아니라 endpoint를 잡는 시간 구간에 의존한다. 따라서 feature contract에는 formula와 함께 sampling, window, missing-value 처리, calibration version이 필요하다."),
    ))

    pages.append(page(
        "32", "Tensor는 feature를 runtime 형식으로 고정한다", "Shape는 배열 구조를, semantic은 각 축과 원소의 뜻을 설명한다.",
        "PART IV · Edge AI", "UNIT 13 · 2/2", "models/model_manifest.json; inference/thermal_interpreter.py; inference/mmwave_interpreter.py",
        ("body", "Tensor는 dtype과 shape를 가진 다차원 배열이다. 그러나 [1,300,1]은 batch·time·channel이라는 축 구조만 말할 뿐 각 time sample이 rad 단위 resp_phase라는 사실은 말하지 않는다. Model input contract는 tensor 구조와 signal semantic을 함께 가져야 한다."),
        ("diagram", "layers", {"labels": [("Axis contract", "batch·time·height·width·channel"), ("Shape", "[1,300,1] / [1,62,80,1]"), ("Dtype", "float32 또는 int8"), ("Numeric contract", "scale·zero point·finite"), ("Semantic", "unit·sampling·preprocess") ]}, 65 * mm, "아래로 갈수록 같은 배열이 무엇을 의미하는지 더 구체적으로 고정한다."),
        ("body", "Wrapper는 shape를 reshape할 수 있지만 의미를 추론해서 복구할 수는 없다. (300,)과 (1,300,1)은 같은 원소 순서를 가진다면 기계적으로 바꿀 수 있지만 breath_rpm 300개를 resp_phase로 바꾸는 것은 불가능하다."),
        ("callout", "Silent mismatch", "Shape가 맞아 invoke가 성공하면 오류가 보이지 않을 수 있다. 의미 mismatch는 crash보다 위험할 수 있으므로 metadata와 producer-consumer contract test가 필요하다.", "red"),
        ("summary", ["Feature는 raw sensor data에서 판단에 필요한 정보를 선택한 표현이다.", "Tensor contract는 shape·dtype뿐 아니라 각 축·단위·시간·전처리 semantic을 포함한다.", "Wrapper는 배열 모양을 바꿀 수 있지만 잘못된 물리 의미를 올바른 feature로 바꿀 수 없다."]),
    ))

    pages.append(page(
        "33", "표준화는 feature 축의 척도를 맞춘다", "학습 통계로 중심과 scale을 바꾸어 optimization과 입력 분포를 안정화한다.",
        "PART IV · Edge AI", "UNIT 14 · 1/2", "models/co2/co2_scaling_metadata_v0.1.0.json; models/mmwave/sensor_stats_metadata_v0.1.0.json",
        ("body", "서로 다른 feature는 단위와 범위가 다르다. CO₂ ppm은 수백에서 수천이고 humidity는 수십, slope는 더 작은 범위를 가질 수 있다. Z-score는 각 feature에서 학습 평균을 빼고 학습 표준편차로 나누어 무차원 좌표로 바꾼다."),
        ("equation", r"z_i=\frac{x_i-\mu_i}{\sigma_i}", "4.2", "μ_i와 σ_i는 runtime batch가 아니라 training data에서 고정한 feature별 통계다."),
        ("diagram", "flow", {"labels": [("물리값 x", "ppm·%·rad"), ("학습 평균", "μ_i"), ("학습 scale", "σ_i"), ("표준값 z", "dimensionless"), ("model input", "distribution") ]}, 38 * mm, "Runtime이 같은 μ와 σ를 사용해야 학습 때와 같은 좌표계가 유지된다."),
        ("body", "Runtime 입력 분포가 학습 분포에서 멀어지면 |z|가 커진다. 이는 단순히 ‘큰 값’이 아니라 out-of-distribution 가능성을 나타낸다. Metadata가 없다고 임의 통계를 사용하면 invoke는 되더라도 model이 다른 좌표계에서 계산하게 된다."),
    ))

    pages.append(page(
        "34", "INT8 양자화는 실수축을 정수 격자로 바꾼다", "Scale과 zero point가 실수와 정수 사이의 affine mapping을 정한다.",
        "PART IV · Edge AI", "UNIT 14 · 2/2", "inference/*_interpreter.py; models/model_manifest.json",
        ("body", "Full-INT8 model은 activation과 weight를 8-bit 정수로 표현해 메모리와 integer 연산을 줄인다. 실수 0이 정수 zero point에 대응하고, scale은 정수 한 칸이 나타내는 실수 간격이다. Quantization은 압축이므로 rounding error와 표현 범위 제한을 만든다."),
        ("equation", r"q=\mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x}{s_q}\right)+z_q,\,-128,\,127\right)", "4.3", "현재 wrapper는 np.rint의 ties-to-even rounding을 사용하고 int8 범위로 clip한다."),
        ("equation", r"\widehat{x}=s_q\left(q-z_q\right)", "4.4", "Dequantization은 정수 q를 대표 실수 값 x-hat으로 되돌리지만 원래 x를 완전히 복원하지는 못한다."),
        ("diagram", "plot", {"plot": "quantization"}, 42 * mm, "연속 실수축이 계단 모양 정수 격자로 mapping되고 범위 밖은 양 끝에서 포화된다."),
        ("summary", ["Z-score는 학습 평균과 scale로 feature를 무차원 좌표로 바꾼다.", "INT8 양자화는 scale과 zero point로 실수축을 256개 정수 level에 대응시킨다.", "표준화 통계와 양자화 parameter가 학습·변환·runtime에서 일치해야 한다."]),
    ))

    pages.append(page(
        "35", "양자화 오차와 saturation을 구분한다", "범위 안의 rounding error와 범위 밖의 clipping error는 크기와 의미가 다르다.",
        "PART IV · Edge AI", "UNIT 15 · 1/2", "models/model_manifest.json; inference/co2_interpreter.py:80-103; inference/thermal_interpreter.py:139-159",
        ("body", "입력이 int8 표현 범위 안에 있으면 가장 가까운 grid point로 반올림되어 오차가 대략 scale 절반 이내에 놓인다. 반면 범위를 넘으면 q가 -128 또는 127에 고정되어 입력이 더 커져도 tensor 값이 변하지 않는다. 이를 saturation 또는 clipping이라 한다."),
        ("equation", r"x_{\min}=s_q(-128-z_q),\qquad x_{\max}=s_q(127-z_q)", "4.5", "이 범위는 양자화 직전의 실수 domain에서 int8 tensor가 표현할 수 있는 구간이다."),
        ("equation", r"\left|x-\widehat{x}\right|\leq\frac{s_q}{2}\quad\mathrm{only\ when\ not\ saturated}", "4.6", "Scale/2 오차 bound는 clipping되지 않은 값에만 적용된다."),
        ("body", "CO₂ model은 먼저 Z-score를 계산한 뒤 input quantization을 적용한다. 이때 representable z 범위를 원래 ppm·humidity·slope 단위로 역변환하면 실제 운영 범위와 겹치는지 확인할 수 있다. 범위가 지나치게 좁으면 converter의 representative dataset이나 중복 preprocessing을 점검해야 한다."),
        ("callout", "정확도와 속도의 관계", "INT8이라는 형식만으로 4배 빠르다고 단정할 수 없다. 실제 latency는 operator 지원, delegate, memory movement, CPU와 input size에 의존하며 target hardware에서 측정해야 한다.", "amber"),
    ))

    pages.append(page(
        "36", "TFLite invoke는 고정된 계산 graph를 실행한다", "Interpreter는 tensor 계약을 준비하고 graph를 호출한 뒤 출력을 해석한다.",
        "PART IV · Edge AI", "UNIT 15 · 2/2", "inference/common.py; inference/*_interpreter.py; models/model_manifest.json",
        ("body", "TFLite runtime은 model artifact에서 operator graph와 tensor metadata를 읽고 input tensor를 할당한다. Wrapper는 application의 sensor 표현을 tensor로 바꾸고 set_tensor→invoke→get_tensor 순서로 graph를 실행한다. 이 경로의 성공은 graph가 계산됐다는 뜻이지 semantic correctness를 증명하지 않는다."),
        ("diagram", "flow", {"labels": [("Artifact", "tflite+hash"), ("Interpreter", "allocate"), ("Input", "validate·quantize"), ("invoke", "operator graph"), ("Output", "dequantize"), ("Prediction", "class·score") ]}, 40 * mm, "Wrapper는 application contract와 model tensor contract 사이의 번역기다."),
        ("h2", "초기화 시 고정해야 하는 것"),
        ("body", "Model path와 hash, input/output shape·dtype·scale·zero point, class map, metadata version을 한 번에 검증해야 한다. 하나라도 다르면 fail fast 해야 잘못된 artifact가 조용히 실행되는 것을 막을 수 있다. Runtime exception은 component fault로 containment해 health와 reason에 전달해야 한다."),
        ("callout", "Artifact와 의미의 관계", "SHA-256은 정확히 같은 파일임을 증명하지만 그 파일이 올바른 data와 preprocessing으로 학습됐는지는 증명하지 않는다. Artifact integrity와 model validity는 서로 다른 증거 층이다.", "purple"),
        ("summary", ["범위 안의 quantization은 rounding error를 만들고 범위 밖은 saturation을 만든다.", "TFLite wrapper는 입력 검증·양자화·invoke·출력 해석을 담당한다.", "Artifact hash와 tensor invoke 성공만으로 semantic 또는 현장 성능이 증명되지는 않는다."]),
    ))

    pages.append(page(
        "37", "출력 score는 정확도와 같은 값이 아니다", "Model의 정규화 출력은 선택 class의 상대적 강도이며 현장 확률은 calibration이 필요하다.",
        "PART IV · Edge AI", "UNIT 16 · 1/3", "inference/*_interpreter.py; models/model_manifest.json",
        ("body", "Classification model은 각 class에 score를 내고 가장 큰 항을 prediction으로 선택한다. Softmax output은 합이 1이지만 그 숫자가 실제 사건 확률과 일치하려면 독립 data에서 calibration이 확인되어야 한다. 높은 score는 입력이 학습 분포 밖인지도 알려주지 않는다."),
        ("equation", r"p_i=\frac{e^{a_i}}{\sum_j e^{a_j}},\qquad \widehat{c}=\underset{i}{\arg\max}\;p_i", "4.7", "a_i는 logit이고 p_i는 class 간 정규화 score다. p_i가 calibrated probability라는 보장은 별도다."),
        ("body", "현재 wrapper는 dequantized output의 음수를 clip하고 합으로 다시 나누는 경로가 있다. 따라서 문서에서는 confidence보다 <b>normalized class score</b>라고 부르는 것이 안전하다. Accuracy는 많은 labeled samples에서 맞은 비율이고 한 sample의 score와 다른 개념이다."),
        ("callout", "Calibration", "Reliability curve는 score 구간별 평균 confidence와 실제 정답 비율을 비교한다. 두 값이 일치해야 0.8 score를 약 80% 사건 확률처럼 해석할 근거가 생긴다.", "blue"),
    ))

    pages.append(page(
        "38", "세 model은 서로 다른 질문에 답한다", "Tensor 형식이 모두 int8이어도 입력 의미와 class namespace는 공유되지 않는다.",
        "PART IV · Edge AI", "UNIT 16 · 2/3", "models/model_manifest.json",
        ("table", ["Model", "입력 semantic", "출력 class", "현재 시스템 역할"], [
            ["Thermal", "normalized frame [1,62,80,1]", "NOT_HUMAN / NORMAL / FALL", "FALL score≥0.8이면 emergency"],
            ["CO₂", "slope·humidity·ppm [1,3]", "VACANT / OCCUPIED", "class는 meta에 기록; risk는 농도 rule"],
            ["mmWave", "10 Hz resp_phase [1,300,1]", "NORMAL / RAPID_OR_ABNORMAL / APNEA", "class1은 호흡 caution, class2는 timer candidate"],
        ], [0.16, 0.31, 0.28, 0.25]),
        ("body", "이 표는 비교와 reference를 위한 예외적인 표다. Thermal class2와 mmWave class2는 숫자는 같지만 각각 FALL과 APNEA다. CO₂ AI의 OCCUPIED class는 현재 최종 위험도에 직접 가중되지 않고 model_meta에 기록되며 환경 위험은 ppm rule이 계산한다."),
        ("diagram", "fan_in", {"sources": ["Thermal class", "CO₂ class/meta", "mmWave class", "Sensor rules"], "center": "Policy layer", "output": "Decision"}, 51 * mm, "Model output은 policy가 의미를 부여한 뒤에야 시스템 decision이 된다."),
        ("callout", "Namespace 원칙", "Class index, score threshold, preprocessing version은 model_id와 함께 기록한다. ‘class 2 감지’처럼 model 이름을 생략한 표현은 시스템 문서에서 사용하지 않는다.", "amber"),
    ))

    pages.append(page(
        "39", "AI와 규칙은 서로 다른 지식을 표현한다", "Model은 data에서 pattern을 학습하고 rule은 명시적 정책과 물리 한계를 표현한다.",
        "PART IV · Edge AI", "UNIT 16 · 3/3", "risk/risk_rules.py; integrated_node/safenest_risk_engine.py",
        ("body", "AI model은 복잡한 pattern을 압축할 수 있지만 왜 특정 threshold가 안전한지 스스로 정의하지 않는다. Rule은 정상 호흡 범위, CO₂ 문턱, 지속시간, emergency override처럼 사람이 검토해야 하는 정책을 명시적으로 표현한다. Hybrid system은 두 종류의 지식을 결합한다."),
        ("diagram", "flow", {"labels": [("센서 evidence", "value·signal"), ("AI pattern", "learned"), ("Rule constraint", "explicit"), ("Policy", "combine"), ("State", "risk·health") ]}, 39 * mm, "AI가 모든 센서를 자동으로 융합하는 구조가 아니라 model별 class와 rule score를 policy가 결합한다."),
        ("h2", "AI가 대신할 수 없는 질문"),
        ("body", "어떤 오경보가 허용 가능한지, sensor 고장 시 fail-safe와 fail-operational 중 무엇을 택할지, 경보를 operator acknowledgement까지 latch할지, threshold를 누가 승인할지는 system safety의 문제다. Model accuracy가 높아도 이 정책이 자동으로 결정되지는 않는다."),
        ("callout", "현재 모델의 증거 수준", "세 artifact는 로컬 실행과 tensor 계약 수준의 증거가 중심이다. 실센서 독립 label, Raspberry Pi 장시간 성능, 현장 위험 사건에 대한 안전 성능은 별도의 검증 대상이다.", "red"),
        ("summary", ["Class score는 model 내부의 상대적 출력이며 calibration 없이 현장 사건 확률로 단정할 수 없다.", "세 model은 입력 semantic과 class namespace, 최종 policy 연결이 모두 다르다.", "AI는 pattern을 제공하고 rule과 policy는 명시적 안전 요구와 상태 전이를 결정한다."]),
    ))

    pages.append(page(
        "P5", "PART V · 위험 융합과 상태기계", "센서 evidence를 하나의 숫자로 합치되 emergency·health·reason을 잃지 않는다.",
        "PART V · 위험과 상태", "PART OPENER", "risk/risk_rules.py; risk/risk_config.json; integrated_node/safenest_risk_engine.py",
        ("body", "위험 융합은 여러 sensor와 model의 서로 다른 evidence를 공통 부분 점수로 바꾸고 정책 가중치로 결합하는 과정이다. 그러나 모든 중요한 사건을 평균으로 표현할 수는 없다. 낙상과 무호흡처럼 즉시 또는 지속시간 확인 후 강제로 DANGER가 되어야 하는 사건은 emergency override로 분리한다."),
        ("diagram", "layers", {"labels": [("부분 evidence", "호흡·환경·심박·자세·움직임"), ("부분 점수", "S_i∈[0,1]"), ("가중 융합", "R_raw"), ("Emergency", "override"), ("시간 필터", "mean·IIR"), ("상태기계", "risk·health·reason") ]}, 76 * mm, "융합은 하나의 pipeline이지만 emergency와 health는 별도 경로로 유지된다."),
        ("h2", "이 장의 중심 질문"),
        ("body", "부분 점수와 weight가 어떤 의미를 가지는가, missing sensor를 0점으로 볼 수 있는가, emergency가 filter를 어떻게 우회하는가, 상태 전이가 왜 이전 상태에 의존하는가를 수식과 도식으로 설명한다."),
        ("summary", ["위험 융합은 서로 다른 evidence를 공통 정책 척도로 바꾸는 과정이다.", "Emergency와 health는 weighted average에 흡수시키지 않고 별도 의미를 보존해야 한다."]),
    ))

    pages.append(page(
        "40", "부분 점수는 sensor evidence를 정책 척도로 바꾼다", "물리값을 곧바로 더하지 않고 modality별 의미를 0…1로 해석한다.",
        "PART V · 위험과 상태", "UNIT 17 · 1/2", "risk/risk_rules.py:87-256; risk/risk_config.json",
        ("body", "ppm, RPM, BPM, class score처럼 단위가 다른 값을 직접 합할 수는 없다. 각 rule은 sensor evidence를 정상·주의·위험·fault 의미에 따라 dimensionless partial score S_i로 mapping한다. 이 mapping은 물리 법칙이 아니라 안전 정책이므로 threshold와 slope의 근거가 별도로 필요하다."),
        ("equation", r"S_i=\psi_i\!\left(x_i,\,m_i,\,q_i,\,s_i\right),\qquad 0\leq S_i\leq1", "5.1", "ψ_i는 sensor value x_i, model output m_i, 품질 q_i, 내부 상태 s_i를 부분 위험도로 바꾸는 modality별 rule이다."),
        ("diagram", "flow", {"labels": [("물리값", "rpm·ppm·bpm"), ("유효성", "type·range·time"), ("의미 rule", "threshold·timer"), ("부분 점수", "S_i"), ("reason", "원인 code") ]}, 39 * mm, "부분 점수와 reason은 같은 판단에서 나오지만 숫자와 설명이라는 다른 역할을 한다."),
        ("body", "Fault를 0점으로 두면 ‘위험 증거가 없음’과 ‘관측할 수 없음’이 같아진다. 현재 호흡 fault는 0.5, CO₂ fault는 0.2처럼 일부 score를 부여하고 health를 낮춘다. 이런 값은 보수성의 정도를 정하는 policy parameter이지 sensor 고장의 확률이 아니다."),
    ))

    pages.append(page(
        "41", "가중합은 부분 위험도를 하나의 척도로 결합한다", "Weight는 센서의 정확도만이 아니라 정책상 상대적 영향력을 나타낸다.",
        "PART V · 위험과 상태", "UNIT 17 · 2/2", "risk/risk_rules.py:303-323; risk/risk_config.json:18-28",
        ("body", "각 partial score가 0…1 범위에 있으면 weight로 결합해 0…100 위험도로 바꿀 수 있다. Weight 합을 1로 두면 모든 S_i=1일 때 R_raw=100이 된다. 하지만 이 정규화만으로 통계적 확률이나 최적 융합이 되는 것은 아니다."),
        ("equation", r"R_{\mathrm{raw}}=100\sum_{i=1}^{M}w_iS_i,\qquad \sum_i w_i=1", "5.2", "현재 weight는 respiration 0.30, environment 0.25, HR 0.20, posture 0.15, motion 0.10이다."),
        ("diagram", "fan_in", {"sources": ["0.30·S_resp", "0.25·S_env", "0.20·S_HR", "0.15·S_posture", "0.10·S_motion"], "center": "Σ weighted evidence", "output": "R_raw"}, 58 * mm, "Weight는 각 modality의 최대 기여도를 제한한다."),
        ("body", "예를 들어 environment score가 1이어도 단독 기여는 25점이다. 따라서 CO₂ component가 HIGH_CO2_DANGER reason을 내도 전역 필터 상태는 NORMAL일 수 있다. Component severity와 global state를 같은 단어로 부르면 이러한 policy 구조를 오해하게 된다."),
        ("summary", ["부분 점수는 단위가 다른 sensor evidence를 공통 정책 척도로 mapping한다.", "가중합의 weight는 각 modality가 전역 위험도에 미치는 최대 영향력을 정한다.", "부분 component의 DANGER reason과 최종 global DANGER state는 서로 다른 계층이다."]),
    ))

    pages.append(page(
        "42", "품질은 위험도와 별도의 정보다", "Q_i가 계산된다는 사실과 위험 가중치에 반영된다는 사실을 구분한다.",
        "PART V · 위험과 상태", "UNIT 18 · 1/2", "integrated_node/safenest_risk_engine.py:98-125,360-377; risk/risk_rules.py:312-319",
        ("body", "Sensor quality Q_i는 해당 경로의 입력이 기본 조건을 만족하는지를 나타낸다. 현재 Quality Gate는 thermal shape, CO₂ 범위, mmWave key 존재, PIR motion 존재를 coarse하게 본다. 그러나 Q_i는 telemetry와 실행 gate에 주로 쓰이며 current risk weight를 동적으로 재정규화하지 않는다."),
        ("equation", r"R_{\mathrm{current}}=100\sum_i w_iS_i", "5.3", "현재 구현의 위험식에는 Q_i가 곱해지지 않는다."),
        ("equation", r"R_{Q}=100\frac{\sum_i Q_iw_iS_i}{\sum_i Q_iw_i}", "5.4", "이 식은 가능한 설계 대안일 뿐 현재 구현이 아니다. 분모가 작을 때 과신할 수 있어 hazard review가 필요하다."),
        ("body", "가용 sensor만 재정규화하면 일부 고장에도 판단을 계속하는 fail-operational 장점이 있다. 반면 강한 위험 evidence를 내는 sensor가 고장나면 남은 약한 evidence의 weight가 커져 결과가 과신될 수 있다. Quality-aware fusion은 수식 하나가 아니라 missingness semantics와 안전 철학의 선택이다."),
        ("callout", "Health propagation", "Q_i, component status, fallback reason을 최종 output까지 보존해야 한다. Risk 숫자에 품질을 섞어버리면 낮은 위험인지 낮은 신뢰인지 구분하기 어려워진다.", "amber"),
    ))

    pages.append(page(
        "43", "Emergency timer는 지속시간을 정책으로 바꾼다", "순간적인 candidate와 일정 시간 지속된 사건을 구분한다.",
        "PART V · 위험과 상태", "UNIT 18 · 2/2", "risk/risk_rules.py:87-146; integrated_node/safenest_risk_engine.py:242-331",
        ("body", "무호흡 candidate는 breath_rpm≤0.5 또는 mmWave class2에서 시작된다. 유효한 numeric timestamp 경로에서는 첫 candidate 시각 t₀를 저장하고 이후 elapsed를 계산한다. Candidate가 사라지면 timer를 reset하며 2초 이상 지속되면 emergency override가 된다."),
        ("equation", r"t_{\mathrm{elapsed}}=t_k-t_0,\qquad \mathrm{emergency}=[t_{\mathrm{elapsed}}\geq2.0\ \mathrm{s}]", "5.5", "2초는 현재 provisional policy의 confirmation interval이다."),
        ("diagram", "timeline", {"markers": [(0.0, "candidate 시작", "t₀"), (0.48, "1.0 s", "계속 확인"), (0.94, "1.9 s", "비응급"), (1.0, "2.0 s", "EMERGENCY") ]}, 36 * mm, "지속시간 판단은 sample count가 아니라 검증된 timestamp 차이로 정의하는 것이 원칙이다."),
        ("body", "현재 timestamp 선택은 key가 없을 때 wall clock으로 fallback하지만 timestamp_s=None이 명시되면 None을 유지한다. RiskRules에 time과 dt_s가 모두 없으면 candidate를 즉시 2초로 처리하는 fallback이 있다. 또한 resp_phase와 non-numeric time이 함께 오면 Adapter에서 먼저 TypeError가 날 수 있다. 정상 2초 의미는 valid-time 경로에 한정된다."),
        ("summary", ["Quality는 판단 경로의 가용성을 나타내며 현재 risk weight를 재정규화하지 않는다.", "Emergency timer는 candidate 시작 시각과 현재 시각의 차이로 지속시간을 확인한다.", "유효하지 않은 time 경로는 정상 2초 의미와 분리해 fault로 설계해야 한다."]),
    ))

    pages.append(page(
        "44", "Emergency override는 평균 경로를 우회한다", "즉시 보호가 필요한 사건을 smoothing delay에 맡기지 않는다.",
        "PART V · 위험과 상태", "UNIT 19 · 1/2", "integrated_node/safenest_risk_engine.py:326-344; risk/risk_rules.py:87-210",
        ("body", "Moving mean과 IIR은 noise를 줄이지만 빠른 위험에서도 출력이 천천히 상승한다. 낙상 또는 확인된 무호흡처럼 policy가 즉시 경보를 요구하는 사건은 일반 filter 경로를 우회해 위험도와 IIR state를 100으로 설정한다."),
        ("equation", r"R[k]=e_k\,100+(1-e_k)\left[(1-\alpha)R[k-1]+\alpha M[k]\right]", "5.6", "e_k∈{0,1}은 emergency override 여부다. Emergency일 때 moving mean과 IIR update를 우회한다."),
        ("diagram", "flow", {"labels": [("부분 판단", "rules"), ("Emergency?", "e_k"), ("override", "R=100"), ("DANGER", "alarm"), ("다음 tick", "recovery") ]}, 38 * mm, "Emergency는 빠른 진입을 보장하지만 해제와 recovery는 별도 상태 정책이 필요하다."),
        ("body", "현재 emergency branch는 raw risk history를 clear하지 않는다. 다음 non-emergency packet에서는 IIR state 100과 기존 raw history가 함께 작용해 천천히 감소한다. 따라서 emergency를 한 번 겪은 시스템은 같은 현재 입력에서도 이전 history가 없는 시스템과 다른 출력을 낸다."),
        ("callout", "Safety latch의 별도성", "GUI risk가 낮아졌다고 실제 alarm을 자동 해제할지는 별도 policy다. Operator acknowledgement, actuator latch, 재경보 조건은 현재 filter state와 다른 safety lifecycle로 설계해야 한다.", "red"),
    ))

    pages.append(page(
        "45", "위험 상태기계는 이전 상태를 기억한다", "같은 R이라도 어디에서 왔는지에 따라 다음 상태가 달라질 수 있다.",
        "PART V · 위험과 상태", "UNIT 19 · 2/2", "integrated_node/safenest_risk_engine.py:338-344; risk/risk_config.json",
        ("body", "최종 status는 filtered risk의 단순 구간 분류가 아니라 이전 상태를 포함한 state transition이다. NORMAL/CAUTION에서 DANGER로 들어갈 때는 75를 사용하고, 이미 DANGER이면 65보다 클 때 유지한다. 이 차이가 hysteresis다."),
        ("diagram", "state", {}, 47 * mm, "상태 전이의 guard에는 risk threshold뿐 아니라 fault와 emergency event도 포함된다."),
        ("equation", r"z_{k+1}=F\!\left(z_k,R[k],e_k,h_k\right)", "5.7", "z_k는 risk state, e_k는 emergency, h_k는 system health다. 현재 구현은 일부 축을 별도 field로 출력한다."),
        ("body", "상태 이름은 사용자 행동과 연결되므로 진입·해제 조건, 최소 유지시간, acknowledgement, fault 우선순위를 명시해야 한다. Config와 code에 threshold가 중복되면 같은 상태 이름이 서로 다른 정책을 가리킬 수 있어 single source of truth가 필요하다."),
        ("summary", ["Emergency override는 smoothing을 우회해 R과 filter state를 즉시 100으로 만든다.", "Emergency 이후 recovery는 남아 있는 raw history와 IIR state의 영향을 받는다.", "최종 상태는 현재 risk뿐 아니라 이전 상태·emergency·health에 의존하는 상태기계다."]),
    ))

    pages.append(page(
        "46", "Fault는 원인과 영향 경로를 따라 전파되어야 한다", "하위 component의 실패가 최종 출력에서 사라지면 낮은 위험을 신뢰할 수 없다.",
        "PART V · 위험과 상태", "UNIT 20", "risk/risk_rules.py:39-61,258-335; integrated_node/safenest_risk_engine.py:120-415",
        ("body", "Fault propagation은 한 component의 오류를 downstream이 해석할 수 있는 status와 reason으로 바꾸는 과정이다. 예외를 catch하는 것만으로 충분하지 않다. 어떤 입력이 거부됐는지, 해당 modality가 판단에서 제외됐는지, fallback이 사용됐는지, 전역 health가 어떻게 바뀌었는지를 보존해야 한다."),
        ("diagram", "layers", {"labels": [("입력 fault", "type·range·time"), ("Component status", "OK·DEGRADED·FAULT"), ("Reason taxonomy", "구조화된 원인"), ("Fusion policy", "부분 점수·제외"), ("System health", "최종 가용성"), ("Output", "risk+health+reason") ]}, 74 * mm, "Reason은 log용 문자열이 아니라 상태 전이와 소비자 정책을 연결하는 기계 계약이다."),
        ("h2", "Fault containment의 경계"),
        ("body", "Sensor adapter, model wrapper, rule evaluator 각각이 예상 가능한 오류를 자신이 이해하는 reason으로 변환해야 한다. Non-numeric value가 np.isnan 또는 np.isfinite에서 TypeError로 전체 tick을 중단하면 containment 경계가 없는 것이다. 반대로 모든 예외를 같은 MODEL_ERROR로 바꾸면 원인을 재현할 수 없다."),
        ("callout", "출력 원칙", "Risk가 0이어도 system_status=FAULT일 수 있고, DANGER와 DEGRADED가 동시에 존재할 수 있다. 안전 UI와 API는 risk·health·reason을 항상 함께 전달해야 한다.", "red"),
        ("summary", ["Fault propagation은 입력 오류를 component status와 reason으로 바꾸어 전역 health까지 전달하는 과정이다.", "Exception containment와 원인 분류를 함께 설계해야 crash와 의미 손실을 모두 막을 수 있다.", "Risk 숫자는 health를 대체하지 않으며 두 축은 최종 출력에서 함께 보존되어야 한다."]),
    ))

    pages.append(page(
        "P6", "PART VI · 확장 가능한 시스템 구조", "센서가 늘어나도 identity·time·meaning·state의 경계가 흐려지지 않게 한다.",
        "PART VI · 확장 구조", "PART OPENER", "integrated_node/virtual_sensor_streamer.py; integrated_node/safenest_risk_engine.py; models/model_manifest.json",
        ("body", "시스템 확장의 어려움은 component 개수가 아니라 <b>관계의 가지 수</b>가 빠르게 늘어난다는 데 있다. Sensor마다 다른 payload·clock·fault 규칙을 fusion code가 직접 알면 새 sensor 하나가 기존 모든 경로와 결합된다. 확장 가능성은 이 결합을 명시적 계약과 state owner로 바꾸는 성질이다."),
        ("diagram", "layers", {"labels": [("Identity", "device·sensor·session"), ("Schema", "value·unit·semantic"), ("Clock", "measured·received·age"), ("State store", "keyed history·timer"), ("Fault model", "status·reason"), ("Observability", "trace·version·metric")]}, 76 * mm, "확장 지점은 코드 파일이 아니라 의미와 상태의 소유권을 보존하는 계약 경계여야 한다."),
        ("h2", "이 장의 중심 질문"),
        ("body", "Canonical schema가 어떤 차이를 흡수해야 하는가, 서로 다른 clock의 data를 언제 같은 사건으로 보는가, history와 timer는 무엇을 key로 분리해야 하는가, 동시성과 재시작에서 어떻게 결정성을 유지하는가를 다룬다."),
        ("summary", ["확장성은 component 수보다 component 사이 의미 결합도와 관련된다.", "Identity·schema·clock·state·fault를 명시하면 새 경로가 기존 의미를 깨는 범위를 제한할 수 있다."]),
    ))

    pages.append(page(
        "47", "Canonical schema는 정보의 공통 평면을 만든다", "생산자별 payload 차이를 흡수하되 의미 차이를 지우지 않는다.",
        "PART VI · 확장 구조", "UNIT 21 · 1/2", "integrated_node/virtual_sensor_streamer.py:171-199; integrated_node/safenest_risk_engine.py:120-229",
        ("body", "<b>Canonical schema</b>는 여러 생산자가 공통으로 전달해야 하는 정보의 최소 구조다. 모든 sensor를 같은 숫자 형식으로 압축하는 것이 아니라, 값에 붙어 있는 identity·time·unit·semantic·health를 동일한 위치에 두는 방식이다."),
        ("diagram", "schema", {}, 56 * mm, "Envelope은 payload 자체와 그 payload를 해석하는 metadata를 하나의 versioned contract로 묶는다."),
        ("body", "Identity는 이 값이 누구의 어떤 session에서 나왔는지를, Time은 언제 측정되고 언제 도착했는지를, Meaning은 단위와 보정·preprocessing semantic을, Health는 그 값을 어느 정도 신뢰할 수 있는지를 설명한다. Optional field는 없음의 의미와 fallback 규칙까지 정의해야 한다."),
        ("callout", "설계 대상", "SensorEnvelopeV1은 현재 코드에 이 이름으로 구현된 class가 아니라 확장을 위한 canonical contract 모형이다. 현재 dict payload와 adapter를 이 구조에 대입하면 누락된 semantic을 찾을 수 있다.", "amber"),
        ("summary", ["Canonical schema는 서로 다른 생산자를 같은 해석 규칙으로 연결한다.", "Value만 표준화하지 말고 identity·time·unit·semantic·health를 함께 versioning해야 한다."]),
    ))

    pages.append(page(
        "48", "시간 계약은 서로 다른 clock을 비교 가능하게 한다", "측정 시각·수신 시각·순서·age budget을 별도 field로 다룬다.",
        "PART VI · 확장 구조", "UNIT 21 · 2/2", "integrated_node/safenest_risk_engine.py:136-163; risk/risk_rules.py:87-146",
        ("body", "분산된 sensor는 각자의 clock으로 값을 측정하고 network·queue·inference 지연을 거쳐 fusion에 도착한다. <b>Event time</b>은 현상이 관측된 시각이고 <b>processing time</b>은 시스템이 처리한 시각이다. 두 시각을 하나로 두면 느게 도착한 과거 data를 현재 사건처럼 섞을 수 있다."),
        ("equation", r"a_i(t)=t_{\mathrm{fuse}}-t_{\mathrm{measured},i},\qquad \mathrm{valid}_i=[0\leq a_i(t)\leq B_i]", "6.1", "a_i는 fusion 시점의 data age, B_i는 modality별 freshness budget이다. Clock domain이 비교 가능해야 식이 성립한다."),
        ("diagram", "flow", {"labels": [("Sensor clock", "measured_at"), ("Transport", "delay·reorder"), ("Receiver", "received_at"), ("Adapter", "clock normalize"), ("Fusion", "age·alignment")]}, 39 * mm, "Sequence는 중복·순서 역전을, timestamp는 age와 sensor 간 정렬을 판별한다."),
        ("body", "Clock offset과 drift가 있으면 timestamp 비교 자체가 틀릴 수 있다. 따라서 clock source·unit·monotonicity·synchronization 방식을 contract에 넣고, 비교할 수 없으면 임의로 wall clock을 대입하기보다 time quality를 degraded로 전파해야 한다."),
        ("callout", "None과 누락의 차이", "Field가 없는 경우, null이 명시된 경우, 형식이 잘못된 경우는 서로 다른 상태다. Fallback을 적용할 조건과 fault로 보낼 조건을 schema에서 분리한다.", "purple"),
    ))

    pages.append(page(
        "49", "State는 key와 lifecycle을 가져야 한다", "History·filter·timer의 소유자를 명시해 장치와 session 사이의 오염을 막는다.",
        "PART VI · 확장 구조", "UNIT 22 · 1/2", "integrated_node/safenest_risk_engine.py:73-96; risk/risk_rules.py:82-146",
        ("body", "Stateful component를 여러 장치가 공유하면 한 장치의 window, IIR, apnea start time, previous status가 다른 장치의 판단에 들어갈 수 있다. 따라서 상태는 전역 변수가 아니라 <b>identity로 key된 시간적 문맥</b>으로 보아야 한다."),
        ("equation", r"s_{k+1}^{(d,\ell)}=f\!\left(s_k^{(d,\ell)},x_k^{(d,\ell)}\right)", "6.2", "d는 device, ℓ은 session·location·occupant 등 정책이 정한 context key다. 각 key는 독립된 state trajectory를 갖는다."),
        ("diagram", "layers", {"labels": [("State key", "device_id+session_id"), ("Signal history", "window·raw risk"), ("Filter state", "IIR·previous state"), ("Timer state", "candidate start"), ("Lifecycle", "create·reset·expire") ]}, 66 * mm, "State key는 오염을 막고 lifecycle은 종료된 context의 history가 새 context에 재사용되는 것을 막는다."),
        ("body", "Lifecycle은 state 생성, warm-up, 정상 update, reset, timeout, session 종료를 포함한다. Model 교체나 preprocessing version 변경은 과거 window와 새 semantic을 섞을 수 있으므로 state migration 또는 명시적 reset이 필요하다."),
        ("summary", ["State는 device·session·context key에 소유되어야 서로 다른 판단 경로가 섞이지 않는다.", "Lifecycle은 생성·warm-up·reset·expire를 포함하며 semantic version 변경을 state 변경과 연결한다."]),
    ))

    pages.append(page(
        "50", "동시성은 update order와 결정성의 문제다", "병렬 입력이 공유 state를 바꾸면 순서 규칙이 시스템 의미가 된다.",
        "PART VI · 확장 구조", "UNIT 22 · 2/2", "integrated_node/safenest_risk_engine.py:237-415; integrated_node/virtual_sensor_streamer.py",
        ("body", "여러 sensor packet이 동시에 도착하면 scheduler에 따라 update 순서가 바뀐다. 공유 buffer와 timer가 있는 코드에서 순서가 명시되지 않으면 동일한 input set으로도 서로 다른 output이 나올 수 있다. 이를 <b>race condition</b>이라 하며, 결과가 timing에 의존해 재현이 어렵다."),
        ("diagram", "fan_in", {"sources": ["Device A packet", "Device B packet", "Model callback", "Timer event"], "center": "Keyed event loop", "output": "Ordered state"}, 51 * mm, "Key별 serial update는 같은 state의 순서를 고정하고 다른 key는 병렬로 유지할 수 있다."),
        ("equation", r"F\!\left(F(s,m),m\right)=F(s,m)", "6.3", "중복 message m이 재전송되어도 상태가 한 번만 반영되는 idempotency의 이상적 조건이다."),
        ("body", "Sequence number로 중복·역순 packet을 판별하고, key별 queue 또는 lock으로 update를 직렬화하며, state snapshot과 output에 동일한 event identity를 남겨야 한다. 재시작 후에는 persisted state와 새 session을 구분해 이전 timer가 유령 경보를 만들지 않게 한다."),
        ("callout", "Lock만으로는 부족하다", "Lock은 동시 쓰기를 막지만 어떤 event를 먼저 처리해야 하는지는 정의하지 않는다. Ordering·deduplication·late-data policy와 함께 설계해야 한다.", "red"),
    ))

    pages.append(page(
        "51", "전체 시스템을 하나의 인과 서사로 연결한다", "물리 현상에서 위험·health·reason까지 중간 의미를 빠뜨리지 않는다.",
        "PART VI · 확장 구조", "SYNTHESIS", "README.md; models/model_manifest.json; risk/risk_rules.py; integrated_node/safenest_risk_engine.py",
        ("diagram", "layers", {"labels": [("물리 현상", "열·가스·전자기파·움직임"), ("센서 관측", "frame·ppm·I/Q·binary"), ("시간 정렬", "sample rate·window·freshness"), ("표현 변환", "filter·feature·tensor"), ("추론", "model score·class"), ("정책", "rule·timer·weight·override"), ("상태기계", "risk·health·reason"), ("소비자", "UI·alarm·telemetry") ]}, 92 * mm, "각 화살표는 단순한 data 전달이 아니라 새 가정과 semantic이 추가되는 계약 경계다."),
        ("h2", "설명의 출발점"),
        ("body", "Thermal은 표면 복사 온도의 공간 pattern, CO₂는 점유·호흡과 환기가 합쳐진 느린 동역학, mmWave는 거리 bin에 들어 있는 흉부 변위의 phase 변조, PIR은 열 패턴 변화 사건을 관측한다. 서로 다른 물리량은 각자의 sampling·filter·window를 거쳐 시간적 증거가 된다."),
        ("h2", "판단의 완성"),
        ("body", "Tensor contract와 preprocessing을 통과한 신호에서 model이 class score를 만들고, rule이 물리 threshold·지속시간·fault 의미를 부여한다. 부분 점수는 weight로 결합되지만 emergency는 평균을 우회하고, moving mean·IIR·hysteresis가 시간에 대한 정책을 최종 상태로 바꾼다. 이때 risk는 관측된 위험, health는 판단 경로의 가용성, reason은 상태변화의 원인을 나타낸다."),
        ("callout", "이론적 이해의 완성", "전체를 이해했다는 말은 코드 파일을 외웠다는 뜻이 아니다. 각 단계의 입력·출력·가정·시간·상태·고장 경계를 붙여 인과 서사를 재구성할 수 있다는 뜻이다.", "green"),
        ("summary", ["각 sensor는 서로 다른 물리 현상을 관측하며 값의 의미는 preprocessing 경로에서 결정된다.", "Model score는 policy 입력이지 최종 안전 결론이 아니며 rule·timer·fusion·state machine이 최종 판단을 만든다.", "Risk·health·reason은 서로 대체할 수 없는 출력 축이며 함께 보존되어야 한다.", "Stateful component는 identity와 lifecycle에 따라 분리하고 event ordering을 명시해야 재현 가능하다.", "확장은 새 component를 붙이는 일이 아니라 기존 semantic contract를 유지한 채 새 관측·판단 경로를 추가하는 일이다."]),
    ))

    return pages


def build_guide(output: Path = OUTPUT) -> Path:
    pages = build_pages()
    if len(pages) != 58:
        raise RuntimeError(f"expected 58 content pages, got {len(pages)}")
    writer = PageWriter(output)
    writer.cover()
    for theory_page in pages:
        render_page(writer, theory_page)
    if writer.page_no != 59:
        raise RuntimeError(f"expected 59 physical pages, got {writer.page_no}")
    writer.save()
    return output


if __name__ == "__main__":
    result = build_guide()
    print(result)
