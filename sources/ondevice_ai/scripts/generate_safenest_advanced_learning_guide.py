#!/usr/bin/env python3
"""Generate the page-contained SafeNest advanced system learning guide.

The document is intentionally laid out as one self-contained learning sheet per
PDF page.  PageWriter raises on overflow, so a section cannot silently split
across pages.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Flowable, Paragraph, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "SafeNest_시스템이해_심화학습서_20260727.pdf"
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


pdfmetrics.registerFont(TTFont("Korean", str(FONT_PATH)))
pdfmetrics.registerFontFamily(
    "Korean", normal="Korean", bold="Korean", italic="Korean", boldItalic="Korean"
)


STYLES = {
    "body": ParagraphStyle(
        "body",
        fontName="Korean",
        fontSize=8.45,
        leading=12.4,
        textColor=SLATE,
        wordWrap="CJK",
        allowWidows=0,
        allowOrphans=0,
    ),
    "small": ParagraphStyle(
        "small",
        fontName="Korean",
        fontSize=7.25,
        leading=10.2,
        textColor=MUTED,
        wordWrap="CJK",
    ),
    "table": ParagraphStyle(
        "table",
        fontName="Korean",
        fontSize=7.15,
        leading=9.7,
        textColor=SLATE,
        wordWrap="CJK",
    ),
    "table_head": ParagraphStyle(
        "table_head",
        fontName="Korean",
        fontSize=7.25,
        leading=9.4,
        textColor=WHITE,
        alignment=TA_CENTER,
        wordWrap="CJK",
    ),
    "h2": ParagraphStyle(
        "h2",
        fontName="Korean",
        fontSize=10.6,
        leading=14,
        textColor=INK,
        spaceAfter=2.0 * mm,
        wordWrap="CJK",
    ),
    "callout": ParagraphStyle(
        "callout",
        fontName="Korean",
        fontSize=8.1,
        leading=11.8,
        textColor=INK,
        wordWrap="CJK",
    ),
    "code": ParagraphStyle(
        "code",
        # AppleGothic is not monospaced, but it covers both Korean and ASCII.
        # A Korean-capable font is mandatory because several code-review and
        # fault-injection examples intentionally contain Korean annotations.
        fontName="Korean",
        fontSize=6.8,
        leading=9.2,
        textColor=colors.HexColor("#E2E8F0"),
    ),
    "toc": ParagraphStyle(
        "toc",
        fontName="Korean",
        fontSize=8.1,
        leading=11.6,
        textColor=SLATE,
        wordWrap="CJK",
    ),
}


def plain_cell(value: object, style: str = "table") -> Paragraph:
    return Paragraph(escape(str(value)).replace("\n", "<br/>"), STYLES[style])


def rich(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


class LayoutError(RuntimeError):
    pass


class MiniPlot(Flowable):
    """Small explanatory plot used for filters and transfer functions."""

    def __init__(self, kind: str, width: float = BODY_W, height: float = 42 * mm):
        super().__init__()
        self.kind = kind
        self.width = width
        self.height = height

    def draw_axes(self, x0, y0, w, h, xlabel, ylabel):
        c = self.canv
        c.setStrokeColor(MUTED)
        c.setLineWidth(0.7)
        c.line(x0, y0, x0 + w, y0)
        c.line(x0, y0, x0, y0 + h)
        c.setFont("Korean", 6.5)
        c.setFillColor(MUTED)
        c.drawRightString(x0 + w, y0 - 9, xlabel)
        c.drawString(x0 + 3, y0 + h + 3, ylabel)

    def draw(self):
        c = self.canv
        x0, y0 = 16 * mm, 9 * mm
        w, h = self.width - 28 * mm, self.height - 16 * mm
        if self.kind == "risk":
            self.draw_axes(x0, y0, w, h, "입력값", "부분 점수")
            c.setStrokeColor(BLUE)
            c.setLineWidth(2)
            c.line(x0, y0, x0 + 0.32 * w, y0)
            c.line(x0 + 0.32 * w, y0, x0 + 0.72 * w, y0 + 0.72 * h)
            c.line(x0 + 0.72 * w, y0 + 0.72 * h, x0 + w, y0 + h)
            for frac, label in ((0.32, "정상"), (0.72, "주의"), (1.0, "포화")):
                x = x0 + frac * w
                c.setStrokeColor(LINE)
                c.line(x, y0, x, y0 + h)
                c.setFillColor(MUTED)
                c.setFont("Korean", 6.2)
                c.drawCentredString(x, y0 - 9, label)
        elif self.kind == "filter":
            self.draw_axes(x0, y0, w, h, "packet index", "risk")
            vals = [0, 0, 4, 9, 18, 31, 44, 55, 64, 71, 76, 81, 84]
            for idx in range(len(vals) - 1):
                xa = x0 + idx / (len(vals) - 1) * w
                xb = x0 + (idx + 1) / (len(vals) - 1) * w
                ya = y0 + vals[idx] / 100 * h
                yb = y0 + vals[idx + 1] / 100 * h
                c.setStrokeColor(CYAN)
                c.setLineWidth(2)
                c.line(xa, ya, xb, yb)
            c.setStrokeColor(RED)
            c.setDash(3, 2)
            c.line(x0, y0 + 0.75 * h, x0 + w, y0 + 0.75 * h)
            c.setDash()
            c.setFont("Korean", 6.3)
            c.setFillColor(RED)
            c.drawString(x0 + 2, y0 + 0.75 * h + 3, "DANGER 진입 75")
        elif self.kind == "spectrum":
            self.draw_axes(x0, y0, w, h, "frequency (Hz)", "energy")
            for f, a, col in ((0.1, 0.20, LINE), (0.333, 0.90, BLUE), (0.5, 0.20, LINE)):
                x = x0 + f / 1.0 * w
                c.setStrokeColor(col)
                c.setLineWidth(3 if f == 0.333 else 1)
                c.line(x, y0, x, y0 + a * h)
                c.setFillColor(MUTED)
                c.setFont("Korean", 6.2)
                c.drawCentredString(x, y0 - 9, f"{f:g}")
            c.setFillColor(PALE_BLUE)
            c.rect(x0 + 0.1 * w, y0, 0.4 * w, h, fill=1, stroke=0)
            c.setStrokeColor(BLUE)
            c.setLineWidth(3)
            x = x0 + 0.333 * w
            c.line(x, y0, x, y0 + 0.9 * h)


class PageWriter:
    def __init__(self, output: Path):
        output.parent.mkdir(parents=True, exist_ok=True)
        self.c = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
        self.c.setTitle("SafeNest 시스템 이해 심화 학습서")
        self.c.setSubject("전자공학 고학년 팀원을 위한 구현 기반 시스템 학습서")
        self.c.setAuthor("SafeNest System Understanding")
        self.page_no = 0
        self.y = TOP_Y
        self.source = ""
        self.part = ""
        self.title = ""

    def _draw_flowable(self, item: Flowable, gap_after: float = 2.2 * mm):
        width, height = item.wrap(BODY_W, self.y - BOTTOM_Y)
        if height > self.y - BOTTOM_Y + 0.01:
            raise LayoutError(
                f"page {self.page_no} overflow in '{self.title}': "
                f"need {height/mm:.1f} mm, have {(self.y-BOTTOM_Y)/mm:.1f} mm"
            )
        item.drawOn(self.c, LEFT, self.y - height)
        self.y -= height + gap_after
        return height

    def cover(self):
        self.page_no += 1
        c = self.c
        c.setFillColor(INK)
        c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        c.setFillColor(BLUE)
        c.rect(0, PAGE_H - 13 * mm, PAGE_W, 13 * mm, fill=1, stroke=0)
        c.setFillColor(CYAN)
        c.circle(PAGE_W - 31 * mm, PAGE_H - 45 * mm, 17 * mm, fill=1, stroke=0)
        c.setFillColor(PURPLE)
        c.circle(PAGE_W - 16 * mm, PAGE_H - 60 * mm, 10 * mm, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Korean", 8)
        c.drawString(20 * mm, PAGE_H - 31 * mm, "SafeNest ON-DEVICE AI SYSTEM")
        c.setFont("Korean", 29)
        c.drawString(20 * mm, PAGE_H - 62 * mm, "시스템 이해")
        c.drawString(20 * mm, PAGE_H - 78 * mm, "심화 학습서")
        c.setFillColor(colors.HexColor("#CBD5E1"))
        c.setFont("Korean", 11)
        c.drawString(20 * mm, PAGE_H - 94 * mm, "전자공학 고학년 팀원을 위한 구현 추적형 교재")
        c.setFillColor(colors.HexColor("#1E3A5F"))
        c.roundRect(20 * mm, 72 * mm, PAGE_W - 40 * mm, 68 * mm, 7, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Korean", 9.5)
        cover_lines = [
            "센서 물리량 -> 샘플링·전처리 -> TFLite 텐서 계약",
            "-> 규칙·상태기계·융합 -> 고장·검증·확장 설계",
        ]
        for idx, line in enumerate(cover_lines):
            c.drawString(29 * mm, 119 * mm - idx * 9 * mm, line)
        c.setFont("Korean", 8)
        c.setFillColor(colors.HexColor("#BFDBFE"))
        c.drawString(29 * mm, 88 * mm, "기준: 2026-07-26 프로젝트 루트의 실제 코드와 로컬 검증 결과")
        c.drawString(29 * mm, 80 * mm, "문서 상태: 내부 학습·설계 검토용 / 의료 성능 보증 문서 아님")
        c.setFillColor(colors.HexColor("#94A3B8"))
        c.setFont("Korean", 7.5)
        c.drawString(20 * mm, 18 * mm, "v2.1 · 68-page page-contained edition")
        c.drawRightString(PAGE_W - 20 * mm, 18 * mm, "SafeNest System Understanding")
        c.bookmarkPage("cover")
        c.addOutlineEntry("표지", "cover", level=0, closed=False)
        c.showPage()

    def start(self, number: str, title: str, subtitle: str, part: str, source: str = ""):
        self.page_no += 1
        self.y = TOP_Y
        self.source = source
        self.part = part
        self.title = title
        c = self.c
        c.setFillColor(WHITE)
        c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Korean", 6.8)
        c.drawString(LEFT, PAGE_H - 11 * mm, "SafeNest 시스템 이해 심화 학습서")
        c.setFillColor(MUTED)
        c.drawRightString(PAGE_W - RIGHT, PAGE_H - 11 * mm, part)
        c.setStrokeColor(LINE)
        c.line(LEFT, PAGE_H - 14 * mm, PAGE_W - RIGHT, PAGE_H - 14 * mm)

        c.setFillColor(BLUE)
        c.setFont("Korean", 8)
        c.drawString(LEFT, self.y, f"LEARNING SHEET {number}")
        self.y -= 9 * mm
        c.setFillColor(INK)
        c.setFont("Korean", 18)
        c.drawString(LEFT, self.y, title)
        self.y -= 7 * mm
        c.setFillColor(MUTED)
        c.setFont("Korean", 8)
        c.drawString(LEFT, self.y, subtitle)
        self.y -= 6 * mm
        c.setStrokeColor(BLUE)
        c.setLineWidth(1.5)
        c.line(LEFT, self.y, LEFT + 24 * mm, self.y)
        c.setStrokeColor(LINE)
        c.setLineWidth(0.5)
        c.line(LEFT + 24 * mm, self.y, PAGE_W - RIGHT, self.y)
        self.y -= 5 * mm

        key = f"sheet-{number}"
        c.bookmarkPage(key)
        c.addOutlineEntry(f"{number}. {title}", key, level=0, closed=False)

    def h2(self, text: str):
        self._draw_flowable(rich(f"<b>{text}</b>", "h2"), gap_after=0.5 * mm)

    def body(self, text: str, gap: float = 2.2 * mm):
        self._draw_flowable(rich(text, "body"), gap_after=gap)

    def small(self, text: str, gap: float = 1.8 * mm):
        self._draw_flowable(rich(text, "small"), gap_after=gap)

    def bullets(self, items: list[str], color=BLUE, gap: float = 2.2 * mm):
        rows = []
        for item in items:
            rows.append(
                [
                    rich(f'<font color="{color.hexval()}">•</font>', "body"),
                    rich(item, "body"),
                ]
            )
        table = Table(rows, colWidths=[5 * mm, BODY_W - 5 * mm], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                    ("TOPPADDING", (0, 0), (-1, -1), 0.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ]
            )
        )
        self._draw_flowable(table, gap_after=gap)

    def numbered(self, items: list[str], gap: float = 2.2 * mm):
        rows = []
        for idx, item in enumerate(items, 1):
            rows.append([plain_cell(idx), rich(item, "body")])
        table = Table(rows, colWidths=[8 * mm, BODY_W - 8 * mm], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), PALE_BLUE),
                    ("TEXTCOLOR", (0, 0), (0, -1), BLUE),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.3, LINE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        self._draw_flowable(table, gap_after=gap)

    def table(self, headers: list[str], rows: list[list[object]], widths: list[float] | None = None, gap: float = 2.4 * mm):
        if widths is None:
            widths = [1 / len(headers)] * len(headers)
        total = sum(widths)
        col_widths = [BODY_W * w / total for w in widths]
        data = [[plain_cell(v, "table_head") for v in headers]]
        data.extend([[plain_cell(v) for v in row] for row in rows])
        table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), INK),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PAPER]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3.6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
                ]
            )
        )
        self._draw_flowable(table, gap_after=gap)

    def callout(self, label: str, text: str, tone: str = "blue", gap: float = 2.4 * mm):
        palette = {
            "blue": (PALE_BLUE, BLUE),
            "cyan": (PALE_CYAN, CYAN),
            "green": (PALE_GREEN, GREEN),
            "amber": (PALE_AMBER, AMBER),
            "red": (PALE_RED, RED),
            "purple": (PALE_PURPLE, PURPLE),
        }
        bg, accent = palette[tone]
        inner = Table(
            [[rich(f'<font color="{accent.hexval()}"><b>{label}</b></font>', "callout")], [rich(text, "callout")]],
            colWidths=[BODY_W - 10 * mm],
        )
        inner.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        outer = Table([[inner]], colWidths=[BODY_W], hAlign="LEFT")
        outer.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), bg),
                    ("BOX", (0, 0), (-1, -1), 0.6, accent),
                    ("LINEBEFORE", (0, 0), (0, 0), 4, accent),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        self._draw_flowable(outer, gap_after=gap)

    def code(self, lines: list[str], gap: float = 2.4 * mm):
        data = [[Paragraph(escape(line) if line else "&nbsp;", STYLES["code"])] for line in lines]
        table = Table(data, colWidths=[BODY_W], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), INK),
                    ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#475569")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ]
            )
        )
        self._draw_flowable(table, gap_after=gap)

    def equation(self, lines: list[str], note: str | None = None, gap: float = 2.4 * mm):
        rows = [[plain_cell(line, "table") ] for line in lines]
        if note:
            rows.append([rich(note, "small")])
        table = Table(rows, colWidths=[BODY_W], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE_PURPLE),
                    ("BOX", (0, 0), (-1, -1), 0.6, PURPLE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        self._draw_flowable(table, gap_after=gap)

    def pipeline(self, labels: list[tuple[str, str]], gap: float = 2.5 * mm):
        n = len(labels)
        arrow_w = 7 * mm
        box_w = (BODY_W - arrow_w * (n - 1)) / n
        cells: list[Flowable] = []
        for idx, (title, sub) in enumerate(labels):
            box = Table(
                [[rich(f"<b>{escape(title)}</b>", "callout")], [plain_cell(sub)]],
                colWidths=[box_w],
                rowHeights=[7 * mm, 13 * mm],
            )
            box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE if idx % 2 == 0 else PALE_CYAN),
                        ("BOX", (0, 0), (-1, -1), 0.6, BLUE if idx % 2 == 0 else CYAN),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            cells.append(box)
            if idx < n - 1:
                cells.append(rich('<font color="#64748B"><b>→</b></font>', "h2"))
        widths = []
        for idx in range(len(cells)):
            widths.append(box_w if idx % 2 == 0 else arrow_w)
        row = Table([cells], colWidths=widths, hAlign="LEFT")
        row.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        self._draw_flowable(row, gap_after=gap)

    def plot(self, kind: str, gap: float = 2.4 * mm):
        self._draw_flowable(MiniPlot(kind), gap_after=gap)

    def finish(self):
        c = self.c
        if self.source:
            c.setFillColor(PAPER)
            c.rect(LEFT, 18 * mm, BODY_W, 6.5 * mm, fill=1, stroke=0)
            c.setFillColor(MUTED)
            c.setFont("Korean", 5.9)
            source = self.source
            if len(source) > 145:
                source = source[:142] + "..."
            c.drawString(LEFT + 2 * mm, 20.3 * mm, f"코드 근거 · {source}")
        c.setStrokeColor(LINE)
        c.line(LEFT, 14 * mm, PAGE_W - RIGHT, 14 * mm)
        c.setFillColor(MUTED)
        c.setFont("Korean", 6.5)
        c.drawString(LEFT, 9.5 * mm, "2026-07-27 · 코드 기준 2026-07-26")
        c.drawRightString(PAGE_W - RIGHT, 9.5 * mm, str(self.page_no))
        c.showPage()

    def save(self):
        self.c.save()


def build_guide(output: Path = OUTPUT) -> None:
    w = PageWriter(output)
    w.cover()

    # Content pages are appended below.  Every call to start()/finish() is one
    # complete learning sheet and therefore one physical PDF page.

    w.start(
        "00",
        "이 문서의 사용법",
        "목표는 파일 암기가 아니라, 확장 전에 계약·상태·검증 경계를 설명할 수 있게 되는 것이다.",
        "학습 안내",
        "README.md:1-48; reports/TEST_RESULTS_20260726.md:1-45",
    )
    w.h2("권장 독자와 선수 지식")
    w.body(
        "대상은 전자공학 학부 고학년 수준의 프로젝트 팀원이다. 복소수, 푸리에 변환, 샘플링 정리, "
        "확률·통계, Python 배열, 기본적인 임베디드 I/O를 배웠다고 가정한다. 다만 특정 센서와 "
        "TFLite를 이미 안다고 가정하지 않으므로, 새 용어는 <b>정의 -> 수식 -> 현재 코드 -> 한계</b> 순서로 설명한다."
    )
    w.h2("읽고 나면 할 수 있어야 하는 일")
    w.numbered(
        [
            "한 sensor packet이 어느 함수와 상태를 거쳐 최종 경보가 되는지 추적한다.",
            "센서 물리량, 전처리 값, tensor, class, rule score를 서로 다른 계층으로 구분한다.",
            "10 Hz·300 sample, Z-score, INT8 scale/zero point, LPF·히스테리시스를 계산한다.",
            "<b>구현됨</b>과 <b>실센서·현장 검증됨</b>을 구분하고 과장된 주장을 걸러낸다.",
            "새 센서나 모델을 붙일 때 schema, timestamp, state ownership, fault propagation을 먼저 설계한다.",
        ]
    )
    w.callout(
        "문서의 비목표",
        "이 교재는 의료기기 안전성, 현장 정확도, Raspberry Pi 5 성능을 보증하지 않는다. 현재 저장소의 "
        "프로토타입을 정확히 이해하고 다음 설계를 검토하기 위한 내부 학습 자료다.",
        "amber",
    )
    w.h2("권장 학습 순서")
    w.pipeline(
        [
            ("I", "구조·계약"),
            ("II", "센서·신호"),
            ("III", "Edge AI"),
            ("IV", "융합·상태"),
            ("V", "검증·확장"),
        ]
    )
    w.finish()

    w.start(
        "01",
        "사실의 등급을 먼저 고정한다",
        "코드가 존재한다는 사실과 실제 환경에서 성능이 입증됐다는 사실은 서로 다르다.",
        "학습 안내",
        "models/model_manifest.json:6-108; reports/TEST_RESULTS_20260726.md:13-45",
    )
    w.table(
        ["표기", "정의", "이 문서에서의 예"],
        [
            ["IMPLEMENTED", "호출 가능한 코드 경로가 현재 저장소에 존재", "mmWave Stream Adapter, risk rules"],
            ["LOCAL-VERIFIED", "로컬 자동 테스트 또는 저장 benchmark로 제한적 확인", "48개 테스트, Thermal macOS latency"],
            ["PARTIAL", "호출은 되지만 결과가 최종 판단에 일부만 사용", "CO2 AI class는 model_meta에만 기록"],
            ["DISCONNECTED", "구성요소는 있으나 생산자와 소비자가 연결되지 않음", "MR60 receiver -> RiskEngine"],
            ["PLANNED", "문서나 roadmap의 목표이며 현재 runtime에 없음", "CO2 time-to-danger, ROS2 배포"],
            ["LIMITATION", "현재 설계·검증의 공백 또는 위험한 가정", "Class 1 provenance, per-sensor freshness 부재"],
        ],
        [0.18, 0.38, 0.44],
    )
    w.h2("주장을 읽는 네 단계")
    w.numbered(
        [
            "<b>Artifact(배포 파일)</b>: 파일이 있고 hash(바이트 지문)·크기가 일치하는가?",
            "<b>Mechanical contract(기계 계약)</b>: shape, dtype, quantization, invoke(graph 1회 실행)가 맞는가?",
            "<b>Semantic contract(의미 계약)</b>: 숫자의 단위·필터·시간축이 학습 입력과 같은가?",
            "<b>Operational evidence(운영 증거)</b>: 실센서, 독립 label, Pi, 장시간·고장 상황에서 요구 성능을 만족하는가?",
        ]
    )
    w.callout(
        "예시: mmWave local_runtime_validated",
        "현재 등급은 모델 파일·SHA-256·tensor·로컬 invoke가 확인됐다는 뜻이다. 실센서 호흡 정확도, "
        "RAPID_OR_ABNORMAL class의 provenance(그 class가 어떤 data·label·artifact에서 왔는지의 계보), "
        "Pi 지연시간까지 확인됐다는 뜻은 아니다.",
        "red",
    )
    w.finish()

    w.start(
        "P",
        "공통 선수개념과 표기 프라이머",
        "같은 기호가 다른 계층을 뜻하지 않도록 물리량·텐서·품질·상태 표기를 먼저 고정한다.",
        "학습 안내",
        "inference/*_interpreter.py; integrated_node/safenest_risk_engine.py:98-118,326-344",
    )
    w.pipeline(
        [
            ("물리", "unit·clock"),
            ("표준값", "canonical"),
            ("Tensor", "norm·INT8"),
            ("판단", "class·score"),
            ("상태", "filter·health"),
        ]
    )
    w.table(
        ["고정 표기", "정의", "혼동하면 안 되는 것"],
        [
            ["q_int8", "모델에 저장되는 -128..127 정수", "Q_quality,i(센서 품질)"],
            ["s_quant", "INT8 정수 한 칸의 실수 간격", "sigma_feature(Z-score 분모)"],
            ["f_ADC / f_frame", "chirp 내 ADC rate / frame 간 10 Hz", "둘을 모두 f_s로 쓰지 않음"],
            ["r_range / R_risk", "radar 거리 [m] / 전역 위험도 [0,100]", "대문자 R 하나로 합치지 않음"],
            ["thermal.class=2", "FALL", "mmwave.class=2(APNEA)"],
            ["class score", "정규화된 출력 성분", "현장확률·accuracy 보장"],
        ],
        [0.24, 0.39, 0.37],
    )
    w.h2("최소 정의")
    w.table(
        ["용어 묶음", "이 문서에서의 뜻"],
        [
            ["Artifact·hash·invoke", "배포 파일·그 바이트의 지문·runtime graph 1회 실행"],
            ["Tensor·Z-score·INT8", "다차원 배열·(x-mu)/sigma_feature·q_int8=round(z/s_quant)+zero"],
            ["Freshness·coverage", "마지막 값의 age 제한·window 첫/끝 sample 사이 실제 시간"],
            ["IIR·hysteresis", "과거 출력이 남는 재귀 필터·진입/해제 문턱을 달리하는 상태 기억"],
            ["State owner·health·reason", "상태를 보관/초기화하는 객체·경로 건강·판정 원인 코드"],
            ["fan-out/in·idempotency·oracle", "한 입력의 분기/재결합·중복 실행 안전성·시험의 기대 판정 규칙"],
            ["Lineage·provenance", "data·feature·artifact의 변환/version 계보·특정 결과가 어디서 왔는지의 근거"],
            ["Debounce·HIL", "짧은 변동을 시간/횟수 확인해 오검출 억제·실제 target hardware와 I/O를 포함한 통합시험"],
        ],
        [0.30, 0.70],
        gap=1.2 * mm,
    )
    w.h2("검증·운영 용어도 먼저 고정")
    w.table(
        ["묶음", "선행 정의"],
        [
            ["데이터", "holdout=학습에 쓰지 않은 평가 자료; open-set=학습 class 밖 입력; representative set=배포 분포 대표 표본"],
            ["Calibration", "ECE=confidence-accuracy 차이의 구간 가중 평균; reliability curve=그 차이를 구간별로 그린 그래프"],
            ["실행", "RSS=실제 상주 RAM; CPU governor=주파수 정책; deadline miss=시간 budget 초과; watchdog=heartbeat 감시"],
            ["Stream", "telemetry=운영 관측값; watermark=완료 입력 시각 경계; replay=기록 재생; jitter buffer=도착 변동 흡수"],
        ],
        [0.19, 0.81],
        gap=1.2 * mm,
    )
    w.callout(
        "센서 장을 읽는 반복 순서",
        "물리량·단위 -> canonical signal·시간축 -> 전처리 -> tensor/class -> rule score -> state/fault 순으로 읽는다. "
        "뒤의 수식에서는 이 페이지의 namespace를 사용한다.",
        "blue",
    )
    w.finish()

    w.start("TOC-A", "목차 I", "구조와 센서·신호 처리", "학습 안내")
    w.table(
        ["쪽", "학습 시트", "핵심 질문"],
        [
            [4, "P 공통 선수개념", "tensor·INT8·상태 표기를 어떻게 구분하는가?"],
            [8, "02 임무와 안전 경계", "SafeNest가 판단하는 것과 판단하지 못하는 것은?"],
            [9, "03 AS-IS 전체 구조", "현재 실제 실행 경로는 어디까지 연결됐는가?"],
            [10, "04 코드 계층 지도", "runtime·simulator·legacy를 어떻게 구분하는가?"],
            [11, "05 한 packet 추적", "생산자부터 결과까지 호출 순서는?"],
            [12, "06 공통 packet 계약", "키·shape·단위·필수성이 어디에 정의되는가?"],
            [13, "07 시간축 계약", "sampling, timestamp, freshness가 왜 별도 문제인가?"],
            [14, "08 상태와 생명주기", "buffer와 timer는 누구의 상태인가?"],
            [15, "09 Thermal 측정 체인", "온도와 grayscale intensity는 같은가?"],
            [16, "10 Thermal 데이터·학습", "label과 split이 일반화 주장에 충분한가?"],
            [17, "11 Thermal runtime", "정규화·양자화·낙상 override는 어떻게 연결되는가?"],
            [18, "12 Thermal 검증 경계", "현재 테스트가 보장하지 않는 것은?"],
            [19, "13 CO2 동역학", "농도와 변화율은 어떤 시간척도를 갖는가?"],
            [20, "14 CO2 AI 계약", "세 feature와 occupancy class는 무엇을 뜻하는가?"],
            [21, "15 CO2 위험 연결", "HIGH_CO2_DANGER가 왜 최종 NORMAL일 수 있는가?"],
            [22, "16 PIR·presence", "무움직임과 부재를 왜 구분해야 하는가?"],
            [23, "17 FMCW 개요", "chirp가 어떻게 거리 정보를 만드는가?"],
            [24, "18 range bin", "코드의 64-bin 거리축은 실제 radar resolution인가?"],
            [25, "19 I/Q·phase", "미세 변위가 phase 변화로 나타나는 이유는?"],
            [26, "20 clutter calibration", "무인 background 조건이 왜 필요한가?"],
            [27, "21 chest-bin FFT energy", "30 frame으로 0.1 Hz를 분해할 수 있는가?"],
            [28, "22 MR60 연결성", "실제 UART 값과 AI 입력 사이의 빈 곳은?"],
            [29, "23 mmWave window", "10 Hz·300 sample의 정확한 의미는?"],
            [30, "24 Stream Adapter", "gap·stale·presence state machine은 어떻게 동작하는가?"],
            [31, "25 CSV Adapter", "resampling과 session leakage를 어떻게 막는가?"],
        ],
        [0.09, 0.31, 0.60],
        gap=0,
    )
    w.finish()

    w.start("TOC-B", "목차 II", "Edge AI와 위험 융합", "학습 안내")
    w.table(
        ["쪽", "학습 시트", "핵심 질문"],
        [
            [32, "26 계약 4종", "Model·Manifest·Metadata·Provenance의 역할은?"],
            [33, "27 INT8 수학", "실수와 int8 사이를 어떻게 왕복하는가?"],
            [34, "28 CO2 포화 계산", "표현 범위 밖 입력은 어떤 정보를 잃는가?"],
            [35, "29 TFLite·confidence", "정규화 class score가 신뢰도와 같은가?"],
            [36, "30 Wrapper 감사", "세 wrapper의 검증 강도는 왜 다른가?"],
            [37, "31 모델 카드", "각 모델의 현재 증거 등급은?"],
            [38, "32 부분 점수", "센서 값이 0-1 score로 어떻게 변환되는가?"],
            [39, "33 가중 융합", "규칙 점수의 실제 기여도는 얼마인가?"],
            [40, "34 Quality Gate", "현재 Q_quality가 위험 가중치를 재정규화하는가?"],
            [41, "35 Emergency", "즉시·2초 확인 경로를 어떻게 구분하는가?"],
            [42, "36 LPF·히스테리시스", "필터와 상태 기억이 응답시간에 미치는 영향은?"],
            [43, "37 상태의 세 축", "risk·health·reason을 왜 동시에 봐야 하는가?"],
            [44, "38 출력·관측성", "확장 가능한 결과 envelope에 무엇이 빠졌는가?"],
            [45, "39 정상 startup 추적", "첫 30초의 NORMAL+DEGRADED는 왜 결정적인가?"],
            [46, "40 낙상·무호흡 추적", "두 emergency 경로의 지연 차이는?"],
            [47, "41 CO2·결측 추적", "R_risk=0, NORMAL, FAULT를 왜 구분하는가?"],
        ],
        [0.09, 0.31, 0.60],
        gap=0,
    )
    w.finish()

    w.start("TOC-C", "목차 III", "검증·확장과 구현 부록", "학습 안내")
    w.table(
        ["쪽", "학습 시트", "핵심 질문"],
        [
            [48, "42 현재 불일치", "threshold·health·type·CLI의 알려진 공백은?"],
            [49, "43 48개 테스트", "통과가 증명하는 범위와 빈 곳은?"],
            [50, "44 Benchmark·HIL", "Pi 성능 목표가 왜 아직 충돌하는가?"],
            [51, "45 데이터 검증", "subject/session/source leakage를 어떻게 막는가?"],
            [52, "46 확장 아키텍처", "복잡도 증가를 흡수할 경계는 무엇인가?"],
            [53, "47 추가 체크리스트", "새 센서·모델을 붙이기 전 승인 항목은?"],
            [54, "48 코드 읽기", "팀원이 어떤 순서로 코드를 읽어야 하는가?"],
            [55, "49 실습", "수식과 fault를 직접 재현하는 방법은?"],
            [56, "50 점검 문제", "핵심 개념을 설명할 수 있는가?"],
            [57, "51 해설", "답의 논리와 흔한 오해는?"],
            [58, "부록 A", "용어 A-M"],
            [59, "부록 B", "용어 N-Z"],
            [60, "부록 C", "복제 상수와 동기화 위험"],
            [61, "부록 D", "코드 근거 지도"],
            [62, "부록 E", "확장 전 팀 준비도 gate"],
            [63, "부록 F", "구현 상태기계·update order"],
            [64, "부록 G", "제안 canonical schema·호환성"],
            [65, "부록 H", "실행 가능한 수치 실습 카드"],
            [66, "부록 I", "raw→filter→상태 누적 계산"],
            [67, "부록 J", "요구·hazard·oracle·coverage 추적성"],
            [68, "부록 K", "운영·ML 고급 용어"],
        ],
        [0.09, 0.31, 0.60],
        gap=0,
    )
    w.finish()

    w.start(
        "02",
        "임무와 안전 경계",
        "여러 센서의 불완전한 관측으로 위험 가능성과 시스템 건강을 함께 추정하는 프로토타입이다.",
        "PART I · 구조와 계약",
        "README.md:1-7; risk/risk_rules.py:258-335",
    )
    w.h2("문제의 구조")
    w.pipeline(
        [
            ("현실", "사람·공간"),
            ("센서", "불완전 관측"),
            ("추론", "AI·규칙"),
            ("상태", "risk·health"),
            ("행동", "현재는 GUI"),
        ]
    )
    w.body(
        "SafeNest의 입력은 사람 자체가 아니라 열 격자, CO2 농도, radar 요약값·phase, PIR motion이다. "
        "이 관측에서 호흡 이상·낙상·환경 악화의 <b>가능성</b>을 계산한다. 따라서 센서가 보지 못한 사실을 "
        "정답처럼 말할 수 없고, 입력 또는 모델이 고장난 상태를 낮은 위험과 구분해야 한다."
    )
    w.h2("현재 경계 안과 밖")
    w.table(
        ["경계 안: 현재 코드", "경계 밖: 아직 없음"],
        [
            ["가상 sensor packet 생성", "실제 Thermal·SCD40 장치 driver"],
            ["세 TFLite wrapper와 규칙 호출", "현장 독립 정확도·의료적 유효성"],
            ["in-process 위험도·상태 dict", "relay·buzzer·원격 경보·watchdog"],
            ["matplotlib 통합 GUI", "ROS2/MQTT 서비스·fleet 운영"],
            ["macOS 로컬 테스트", "Pi 장시간·열 throttling·전력 검증"],
        ],
        [0.5, 0.5],
    )
    w.callout(
        "안전 해석 원칙",
        "낮은 risk_score는 '관측된 위험 증거가 낮다'는 뜻이다. 센서가 끊겨 판단할 수 없다는 뜻의 "
        "FAULT 또는 DEGRADED를 지우지 않는다. 숫자와 건강 상태를 함께 소비해야 한다.",
        "red",
    )
    w.finish()

    w.start(
        "03",
        "AS-IS 전체 구조",
        "현재 주 실행 경로는 하나의 Python process 안에서 순차 호출되는 동기 구조다.",
        "PART I · 구조와 계약",
        "integrated_node/virtual_sensor_streamer.py:71-200; integrated_node/safenest_integrated_plotter.py:119-274",
    )
    w.pipeline(
        [
            ("Virtual", "10 Hz packet"),
            ("Engine", "stateful gateway"),
            ("Model", "3 wrappers"),
            ("Rules", "score·override"),
            ("GUI", "4 panels"),
        ]
    )
    w.h2("엔진 내부 fan-out / fan-in")
    w.code(
        [
            "packet",
            "  +-> ThermalInterpreter ------+",
            "  +-> CO2 history -> CO2 AI ---+-> RiskRulesEvaluator",
            "  +-> mmWave buffer -> AI ------+        |",
            "  +-> RPM / HR / apnea / PIR ---+        v",
            "                                   moving mean -> IIR -> hysteresis",
            "                                                     |",
            "                                                     v",
            "                                   result dict -> matplotlib GUI",
        ]
    )
    w.table(
        ["특성", "현재 의미", "확장 시 위험"],
        [
            ["동기 dict", "한 tick의 값이 한 packet에 함께 있다고 가정", "비동기 센서의 서로 다른 age가 숨겨짐"],
            ["단일 instance", "buffer·timer가 한 사용자/세션에 귀속", "device/session이 섞이면 상태 오염"],
            ["in-process", "network serialization과 retry가 없음", "노드 분리 시 schema·idempotency 필요"],
            ["GUI output", "관측용 화면까지 연결", "actuator·persistent log·watchdog 없음"],
        ],
        [0.18, 0.35, 0.47],
    )
    w.callout(
        "현재 연결성",
        "가상 streamer는 resp_phase를 포함한 packet을 만들고, RiskEngine이 최신 model_meta를 결과에 붙여 GUI에 "
        "반환한다. 반면 실제 MR60 receiver는 이 packet을 만들지 않으며 통합엔진으로 publish하지 않는다.",
        "amber",
    )
    w.finish()

    w.start(
        "04",
        "코드 계층 지도",
        "파일이 있다는 이유만으로 공용 runtime 경로에 연결됐다고 가정하지 않는다.",
        "PART I · 구조와 계약",
        "README.md:18-48; walkthrough/SAFENEST_SYSTEM_LEARNING_GUIDE_20260725.md:45-96",
    )
    w.table(
        ["계층", "대표 경로", "판별 질문"],
        [
            ["계약", "models/model_manifest.json, metadata", "모델 입력 의미·버전·hash의 기준인가?"],
            ["Adapter", "adapters/", "생산자 형식을 canonical window로 바꾸는가?"],
            ["Inference", "inference/", "tensor 검사·양자화·invoke·decode를 담당하는가?"],
            ["Policy", "risk/risk_config.json, risk_rules.py", "센서값을 score와 reason으로 바꾸는가?"],
            ["Orchestration", "integrated_node/safenest_risk_engine.py", "상태를 소유하고 경로를 결합하는가?"],
            ["Simulation/UI", "virtual_sensor_streamer.py, plotter.py", "실센서가 아닌 시연·관측인가?"],
            ["Standalone", "mr60/, co2_data/, scripts/", "공용 packet을 실제로 생산·소비하는가?"],
            ["Evidence", "tests/, reports/, benchmarks/", "주장의 범위를 재현 가능한가?"],
        ],
        [0.18, 0.42, 0.40],
    )
    w.h2("가장 빠른 코드 읽기 규칙")
    w.bullets(
        [
            "<b>모델을 먼저 열지 말고</b> manifest의 role·input semantic·class map부터 읽는다.",
            "생산자와 소비자 양쪽에서 같은 key, shape, unit, timestamp를 찾는다.",
            "AI 결과가 model_meta에만 남는지 실제 rule score에 들어가는지 끝까지 추적한다.",
            "exception이 reason, sensor_status, system_status로 전파되는지 확인한다.",
            "테스트 함수가 class 정답을 검사하는지, 단지 invoke 성공만 검사하는지 구분한다.",
        ]
    )
    w.callout(
        "단일 진실원천의 범위",
        "manifest는 artifact 계약의 기준이지만 현재 training split, label provenance, limitation은 충분히 담지 않는다. "
        "'manifest에 있다'와 '현장 의미가 입증됐다'는 별개다.",
        "purple",
    )
    w.finish()

    w.start(
        "05",
        "한 packet을 끝까지 추적한다",
        "시스템 이해의 기본 단위는 class 하나가 아니라 생산자·변환·상태·소비자의 연결이다.",
        "PART I · 구조와 계약",
        "integrated_node/safenest_risk_engine.py:120-415",
    )
    w.numbered(
        [
            "<b>시간 선택</b>: packet.timestamp_s를 읽고 없으면 timestamp, 그것도 없으면 wall clock을 쓴다.",
            "<b>Quality Gate</b>: shape·키·일부 범위로 thermal/co2/mmWave/PIR의 Q_quality를 만든다.",
            "<b>Thermal</b>: 유효 frame이면 wrapper predict 후 FALL rule을 계산한다.",
            "<b>CO2</b>: 최대 30개 history로 endpoint slope를 만들고 AI와 ppm rule을 각각 호출한다.",
            "<b>mmWave</b>: RPM·apnea·HR rule을 즉시 계산하고 resp_phase는 300-sample buffer를 채운다.",
            "<b>PIR</b>: mmWave presence가 확인된 경우에만 no-motion timer를 누적한다.",
            "<b>Fusion</b>: emergency이면 즉시 100, 아니면 weighted raw -> 6-sample mean -> IIR을 적용한다.",
            "<b>Output</b>: risk/status/reasons/quality/derived/model_meta를 하나의 dict로 반환한다.",
        ]
    )
    w.h2("추적할 때 반드시 기록할 여섯 항목")
    w.table(
        ["항목", "예"],
        [
            ["생산자", "VirtualSensorStreamer.generate_packet"],
            ["값·단위", "co2_ppm [ppm], breath_rpm [rpm]"],
            ["형상·주기", "thermal (62,80), resp_phase nominal 10 Hz"],
            ["상태 소유자", "engine의 deque, rules의 timer"],
            ["최종 소비", "rule score인지 model_meta telemetry인지"],
            ["실패 표현", "reason + system_status + fallback provenance"],
        ],
        [0.28, 0.72],
    )
    w.callout(
        "논리적 비약 방지",
        "'모델이 호출됐다' 다음에는 반드시 '그 출력이 어느 분기에서 risk에 사용됐다'를 찾는다. CO2는 이 두 문장 "
        "사이에 연결이 끊겨 있다.",
        "red",
    )
    w.finish()

    w.start(
        "06",
        "공통 sensor packet 계약",
        "현재 계약은 schema 파일이 아니라 생산자와 evaluate_risk 내부 접근 코드에 암묵적으로 존재한다.",
        "PART I · 구조와 계약",
        "integrated_node/virtual_sensor_streamer.py:177-199; integrated_node/safenest_risk_engine.py:98-118,195-315",
    )
    w.table(
        ["경로", "의미·단위", "현재 사용·제약"],
        [
            ["timestamp_s", "공통 sample time [s]", "CO2 slope, timers, mmWave gap/stale"],
            ["thermal_80x62", "normalized intensity grid", "엔진은 ndarray (62,80)만 Q_quality=1"],
            ["co2_scd40.co2_ppm", "농도 [ppm]", "300-10000만 Q_quality=1; 환경 score"],
            ["co2_scd40.humidity", "상대습도 [%]", "기본 45; quality/finite 검사는 없음"],
            ["mmwave_mr60.presence", "0/1 존재 추정", "없으면 default 1; PIR gate와 buffer clear"],
            ["breath_rpm", "호흡 요약 [rpm]", "필수 key; 규칙 입력"],
            ["heart_bpm", "심박 요약 [bpm]", "20-240 범위에서 rule 사용"],
            ["apnea", "upstream 무호흡 flag", "1이면 즉시 emergency"],
            ["resp_phase", "연속 phase sample", "300개 TFLite window의 원소"],
            ["rfft_frame", "선택 64-bin complex", "chest-bin telemetry; AI 입력 미연결"],
            ["pir.motion", "0/1 움직임", "presence가 있을 때 no-motion timer"],
        ],
        [0.30, 0.28, 0.42],
    )
    w.callout(
        "핵심 결손",
        "schema_version, device_id, session_id, per-sensor timestamp·sequence·age, unit, calibration_id가 없다. "
        "다중 장치나 비동기 node로 확장하기 전에 명시적 envelope가 필요하다.",
        "amber",
    )
    w.finish()

    w.start(
        "07",
        "시간축 계약: sampling과 freshness",
        "sample 개수, 측정 시간, 도착 시간은 같은 개념이 아니다.",
        "PART I · 구조와 계약",
        "adapters/mmwave_stream_adapter.py:31-110; integrated_node/safenest_risk_engine.py:203-214",
    )
    w.table(
        ["용어", "정의", "현재 구현"],
        [
            ["measurement time", "센서가 물리량을 측정한 시각", "packet timestamp 하나로 대표"],
            ["arrival time", "process가 값을 받은 시각", "별도 보존 안 함"],
            ["sample rate", "단위시간당 유효 sample 수", "mmWave 10 Hz 선언, 강제 안 함"],
            ["jitter", "이상적 grid에서 시각 편차", "Stream 경로 정량 제한 없음"],
            ["gap", "연속 sample 사이 큰 공백", "mmWave 0.5 s 초과 거부"],
            ["stale", "마지막 값이 현재 판단에 너무 오래됨", "mmWave 2 s; 다른 센서는 없음"],
            ["coverage", "window 첫 sample부터 마지막까지 시간", "Stream은 29.9 s 확인 안 함"],
        ],
        [0.20, 0.39, 0.41],
    )
    w.equation(
        [
            "ideal grid: t[k] = t[0] + k / fs",
            "N = 300, fs = 10 Hz -> last-first coverage = (N-1)/fs = 29.9 s",
            "frequency spacing: delta_f = fs/N = 0.0333 Hz",
        ],
        "300개라는 shape만 맞아도 실제 coverage가 30초라는 보장은 없다.",
    )
    w.callout(
        "확장 시 요구",
        "각 sensor envelope에 measured_at, received_at, sequence, valid_for_ms를 넣고, fusion tick에서 age budget을 "
        "검사한다. 같은 packet에 담겼다는 이유만으로 동시에 측정됐다고 간주하지 않는다.",
        "blue",
    )
    w.finish()

    w.start(
        "08",
        "상태 소유권과 생명주기",
        "현재 엔진은 순수 함수가 아니라 과거 입력을 기억하는 stateful component다.",
        "PART I · 구조와 계약",
        "integrated_node/safenest_risk_engine.py:73-96; risk/risk_rules.py:64-85",
    )
    w.table(
        ["상태", "크기·역할", "reset 조건 / 현재 공백"],
        [
            ["mmwave buffer", "resp_phase 300개", "presence=0 push 또는 gap 시 clear"],
            ["rFFT history", "complex frame 30개", "명시 reset API 없음"],
            ["CO2 history", "timestamp·ppm 30개", "세션 변경·결측 시 clear 안 함"],
            ["risk history", "raw score 6개", "emergency 때도 과거 값 유지"],
            ["IIR state", "curr_smoothed_r", "emergency에서 100으로 설정"],
            ["status memory", "prev_status", "DANGER recovery threshold 선택"],
            ["apnea timer", "candidate 시작 시각", "정상/invalid에서 reset"],
            ["no-motion timer", "motion=0 시작 시각", "motion=1·presence false에서 reset"],
        ],
        [0.24, 0.31, 0.45],
    )
    w.h2("왜 소유권이 중요한가")
    w.bullets(
        [
            "두 device의 packet을 같은 instance에 넣으면 CO2 slope와 filter history가 섞인다.",
            "사람이 바뀌어도 session reset이 없으면 이전 사람의 30초 호흡 window가 남을 수 있다.",
            "빈 packet이 FAULT를 반환해도 모든 history를 초기화하지는 않는다.",
            "process restart는 우연히 reset이지만, 안전한 lifecycle 정책은 아니다.",
        ]
    )
    w.callout(
        "권장 키",
        "state는 최소 (device_id, sensor_id, session_id)에 귀속하고, start_session/end_session, calibration reset, "
        "fault recovery의 전이를 명시한다.",
        "red",
    )
    w.finish()

    w.start(
        "09",
        "Thermal 측정 체인",
        "실제 온도 센서 체인과 현재 저장소의 grayscale intensity 체인을 구분한다.",
        "PART II · 센서와 신호",
        "thermal_prep.py:19-130; inference/thermal_interpreter.py:106-148",
    )
    w.pipeline(
        [
            ("복사", "IR radiation"),
            ("센서", "pixel response"),
            ("보정", "NUC·emissivity"),
            ("frame", "temperature grid"),
            ("model", "posture class"),
        ]
    )
    w.body(
        "열화상 센서는 물체가 방출·반사하는 적외선 복사를 pixel array로 측정한다. 실장 시스템이라면 방사율, "
        "주변 반사온도, 렌즈 FOV, bad pixel, NUC(non-uniformity correction), 온도 단위와 설치 방향이 "
        "입력 계약에 포함돼야 한다. 이 값들이 달라지면 같은 자세도 다른 분포로 보일 수 있다."
    )
    w.h2("현재 repository의 실제 체인")
    w.pipeline(
        [
            ("PNG/JPEG", "display image"),
            ("grayscale", "8-bit intensity"),
            ("resize", "80 x 62"),
            ("divide", "value / 255"),
            ("2D-CNN", "3 classes"),
        ]
    )
    w.table(
        ["구분", "현재 의미", "아직 계약되지 않은 것"],
        [
            ["pixel", "0-1 영상 intensity", "°C, emissivity, NUC 상태"],
            ["geometry", "resize된 62x80 배열", "실제 sensor orientation·FOV"],
            ["time", "단일 frame", "frame rate·temporal fall motion"],
            ["label", "형상 규칙 기반 posture", "사람·장면·사고 ground truth"],
        ],
        [0.18, 0.35, 0.47],
    )
    w.callout(
        "용어 주의",
        "현재 thermal_80x62를 '온도 4,960개'라고 단정하면 안 된다. runtime semantic은 normalized frame이며, "
        "학습 원천은 grayscale 이미지다.",
        "amber",
    )
    w.finish()

    w.start(
        "10",
        "Thermal 데이터와 학습 설계",
        "높은 내부 재현값보다 label provenance와 독립 split이 먼저다.",
        "PART II · 센서와 신호",
        "thermal_prep.py:19-50,61-130; thermal_train.py:18-57",
    )
    w.table(
        ["class", "현재 frame 수", "비율", "해석"],
        [
            ["0 NOT_HUMAN", 4, "0.06%", "일반화 주장을 할 수 없을 정도로 희소"],
            ["1 NORMAL", 3042, "44.91%", "사람 frame의 형상 규칙 label"],
            ["2 FALL", 3728, "55.03%", "aspect ratio 또는 centroid 규칙 label"],
            ["합계", 6774, "100%", "NPZ 직접 재확인"],
        ],
        [0.24, 0.18, 0.16, 0.42],
    )
    w.h2("현재 label 생성")
    w.equation(
        [
            "mask = image > 0.35",
            "pixel_count < 20 -> NOT_HUMAN",
            "aspect_ratio >= 1.20 OR y_center >= 34 -> FALL",
            "otherwise -> NORMAL",
        ],
        "모델은 독립된 자세 정답이 아니라 동일 이미지에서 만든 기하 규칙을 재학습할 수 있다.",
    )
    w.h2("누수 위험")
    w.bullets(
        [
            "train/test는 frame 단위 무작위 80/20 분할이며 subject·session·scene grouping이 없다.",
            "연속 frame 또는 같은 사람의 유사 장면이 양쪽에 들어가면 독립 일반화가 과대평가된다.",
            "NOT_HUMAN 4장은 class imbalance뿐 아니라 open-set 열원 오인 문제를 평가할 수 없다.",
            "공식 artifact를 만드는 완전한 full-INT8 재현 pipeline과 현재 thermal_train.py가 일치하지 않는다.",
        ]
    )
    w.callout(
        "안전한 표현",
        "자동 테스트는 모델이 로드되고 입력 계약을 받으며 출력이 한 class로 완전히 붕괴하지 않는지 확인한다. "
        "실제 낙상 recall이나 현장 정확도를 증명하지 않는다.",
        "red",
    )
    w.finish()

    w.start(
        "11",
        "Thermal runtime과 위험 연결",
        "frame 정규화부터 emergency override까지 실제 계산을 따라간다.",
        "PART II · 센서와 신호",
        "inference/thermal_interpreter.py:80-199; risk/risk_rules.py:193-210",
    )
    w.numbered(
        [
            "엔진 Quality Gate는 정확히 ndarray shape (62,80)일 때만 thermal Q_quality=1로 둔다.",
            "wrapper는 (62,80), (62,80,1), (1,62,80,1)을 허용하고 NaN/Inf를 거부한다.",
            "값이 0-1 밖이면 한 frame의 min/max로 정규화한다.",
            "manifest input scale·zero point로 int8 변환하고 TFLite invoke를 실행한다.",
            "출력을 역양자화하고 음수를 clip한 뒤 합이 1이 되도록 다시 정규화한다.",
            "thermal.class=2(FALL)이고 confidence >= 0.8이면 R_risk=100 emergency다. 나머지 자세는 score 0이다.",
        ]
    )
    w.equation(
        [
            "x_norm = (x - min(frame)) / (max(frame) - min(frame))",
            "q_int8 = clip(rint(x_norm / 0.0038139177 - 128), -128, 127)",
            "score_raw = 0.00390625*(q_out+128); score = score_raw/sum(score_raw)",
        ],
        "input의 양자화 상한은 약 0.97255이므로 normalized 1.0은 int8 127에서 포화된다.",
    )
    w.callout(
        "프레임별 Min-Max의 효과",
        "절대 온도 offset을 제거해 contrast에는 강할 수 있지만, 장면마다 gain이 달라지고 uniform raw temperature frame은 "
        "의도하지 않은 값으로 clip될 수 있다. 센서 driver 계약과 함께 결정해야 한다.",
        "amber",
    )
    w.finish()

    w.start(
        "12",
        "Thermal 검증 경계와 보강점",
        "artifact 무결성, tensor 실행, 인식 성능, 안전 성능은 서로 다른 시험이다.",
        "PART II · 센서와 신호",
        "tests/test_thermal_interpreter.py:30-115; benchmarks/thermal_latest.json:1-15",
    )
    w.table(
        ["층", "현재 증거", "추가로 필요한 증거"],
        [
            ["Artifact", "manifest hash·size와 실제 파일 일치", "학습 run·converter·representative set lineage"],
            ["Tensor", "shape/dtype, finite, supported shape", "wrapper 시작 시 hash·scale·zero point 검증"],
            ["Smoke", "NPZ 일부에서 유효 class, 2종 이상 출력", "label 일치 confusion matrix"],
            ["Generalization", "없음", "subject/scene/session holdout, NOT_HUMAN 확대"],
            ["Temporal safety", "FALL 단일 frame 즉시", "debounce·tracking·false alarm/hour"],
            ["Edge performance", "macOS zero-frame 1000회 p95 0.1606 ms", "Pi sensor-to-alarm, RSS, temperature"],
        ],
        [0.18, 0.39, 0.43],
    )
    w.h2("설계 검토 질문")
    w.bullets(
        [
            "confidence 0.8은 validation set에서 calibration된 threshold인가, 단순 정책값인가?",
            "단일 false FALL이 즉시 경보여야 하는가, 2-of-N frame 확인이 필요한가?",
            "센서 설치 방향이 바뀌면 y_center>=34 규칙과 학습 분포를 어떻게 versioning할 것인가?",
            "temperature calibration이 들어오면 기존 intensity 모델과 별도 Adapter를 둘 것인가?",
        ]
    )
    w.callout(
        "현재 status",
        "manifest의 thermal status는 candidate다. '작동함'은 말할 수 있지만 '현장 낙상을 검증함'은 말할 수 없다.",
        "red",
    )
    w.finish()

    w.start(
        "13",
        "CO2 동역학과 slope",
        "느린 환경 신호는 값 자체뿐 아니라 환기·공간 부피·시간척도의 영향을 받는다.",
        "PART II · 센서와 신호",
        "integrated_node/safenest_risk_engine.py:92-94,195-230; risk/risk_rules.py:148-174",
    )
    w.body(
        "완전 혼합 공간의 개념 모델에서는 체적분율 C의 변화가 발생량 G, 공간 부피 V, 환기 유량 Q_v, 외기 농도 "
        "C_out에 좌우된다. 같은 사람이 호흡해도 V와 Q_v가 다르면 C(t)가 다르므로 CO2 하나를 사람 상태의 직접 "
        "ground truth로 취급할 수 없다. SafeNest는 이를 환경 위험 규칙과 occupancy AI의 보조 입력으로 사용한다."
    )
    w.equation(
        [
            "dC/dt = G/V - (Q_v/V)*(C-C_out)   [C: m3_CO2/m3_air, G: m3_CO2/s]",
            "Q_v [m3_air/s], V [m3_air] -> every term has unit 1/s; ppm = 10^6*C",
            "runtime slope = (C_ppm,last - C_ppm,first) / ((t_last - t_first) / 60)   [ppm/min]",
            "environment score = clip((C_ppm - 500) / 2000, 0, 1)",
        ]
    )
    w.table(
        ["현재 parameter", "값", "정확한 의미"],
        [
            ["history length", "30 samples", "시간 길이가 아니라 개수 제한"],
            ["warning", "1000 ppm", "reason HIGH_CO2_WARNING"],
            ["danger", "2500 ppm", "component reason/status; global override 아님"],
            ["slope warning", "15 ppm/min", "FAST_CO2_RISE reason; score에는 미반영"],
            ["score saturation", "2500 ppm", "environment partial score가 1이 됨"],
        ],
        [0.24, 0.18, 0.58],
    )
    w.callout(
        "가정과 시간척도 mismatch",
        "위 식은 well-mixed, 일정 V·Q_v·G, 추가 sink 없음이라는 개념 가정이다. 가상 10 Hz 호출에서는 30 sample이 "
        "약 2.9초지만 독립 CO2 경로의 설계는 10초 polling·5분 window였다. "
        "feature 이름이 같아도 시간축이 다르면 학습 입력 의미가 달라진다.",
        "red",
    )
    w.finish()

    w.start(
        "14",
        "CO2 AI의 입력과 출력",
        "현재 모델은 위험 도달시간이 아니라 UCI 기반 VACANT/OCCUPIED 2분류다.",
        "PART II · 센서와 신호",
        "models/model_manifest.json:37-65; inference/co2_interpreter.py:38-131",
    )
    w.table(
        ["축", "계약", "runtime 처리"],
        [
            ["input 0", "CO2_slope [ppm/min]", "history endpoint로 계산"],
            ["input 1", "Humidity [%]", "packet 값, 없으면 45"],
            ["input 2", "CO2 [ppm]", "현재 절대 농도"],
            ["normalization", "metadata mean/scale", "z=(x-mean)/scale"],
            ["tensor", "[1,3] int8", "scale 0.00582845, zero 57"],
            ["output", "[VACANT,OCCUPIED]", "역양자화 후 합 1 정규화"],
        ],
        [0.20, 0.35, 0.45],
    )
    w.h2("Wrapper의 현재 강도")
    w.bullets(
        [
            "metadata 파일이 없으면 hard-coded 통계를 조용히 사용한다.",
            "input finite, scale>0, model hash, expected shape/dtype/quantization을 시작 시 검증하지 않는다.",
            "humidity의 물리 범위·freshness는 Quality Gate가 검사하지 않는다.",
            "output 합이 0이면 [0.5,0.5]를 반환하지만 그 사실을 fallback reason으로 남기지 않는다.",
        ]
    )
    w.callout(
        "용어 정정",
        "CO2 AI가 '위험시간을 예측한다'고 설명하면 현재 artifact와 다르다. time-to-danger 회귀는 P1 계획이며, "
        "현재 출력은 VACANT/OCCUPIED 두 class의 정규화 score다. calibration 증거가 없어 재실 확률로 단정하지 않는다.",
        "amber",
    )
    w.finish()

    w.start(
        "15",
        "CO2 AI와 위험 규칙의 실제 연결",
        "AI 호출 성공과 최종 위험도 사용을 분리해서 읽는다.",
        "PART II · 센서와 신호",
        "integrated_node/safenest_risk_engine.py:195-232,391-402; risk/risk_rules.py:148-174,312-319",
    )
    w.code(
        [
            "co2_ppm + humidity + derived slope",
            "          |",
            "          +-> CO2Interpreter -> VACANT/OCCUPIED -> model_meta + GUI",
            "          |                                  (risk input 아님)",
            "          +-> evaluate_environment(ppm, slope)",
            "                         |",
            "                         +-> score + reasons -> weighted risk",
        ]
    )
    w.h2("2500 ppm 단독 계산")
    w.equation(
        [
            "C = 2500 -> environment score = (2500-500)/2000 = 1.0",
            "global contribution = 100 * 0.25 * 1.0 = 25 points",
            "25 < engine CAUTION entry 40 -> final status can remain NORMAL",
        ],
        "동시에 reason=HIGH_CO2_DANGER와 component status=DANGER는 만들어진다.",
    )
    w.table(
        ["축", "결과", "소비자가 해야 할 해석"],
        [
            ["component", "HIGH_CO2_DANGER", "환경 규칙 임계값 초과"],
            ["risk_score", "최대 약 25 단독", "다른 modality가 없으면 전역 점수 낮음"],
            ["status_str", "NORMAL 가능", "현 global policy 결과"],
            ["system_status", "OK 가능", "장비 가용성과 위험은 별도"],
        ],
        [0.20, 0.30, 0.50],
    )
    w.callout(
        "팀 hazard decision 필요",
        "CO2 danger를 reason만 남길지 global override로 올릴지는 코드 버그 여부가 아니라 안전 요구사항 결정이다. "
        "다만 현재 동작을 '즉시 DANGER'라고 문서화하면 틀리다.",
        "red",
    )
    w.finish()

    w.start(
        "16",
        "PIR과 presence gating",
        "PIR motion=0은 사람이 없다는 뜻도, 쓰러졌다는 뜻도 아니다.",
        "PART II · 센서와 신호",
        "risk/risk_rules.py:212-256; integrated_node/safenest_risk_engine.py:240-249,306-315",
    )
    w.table(
        ["입력 조합", "timer 동작", "RuleResult"],
        [
            ["PIR invalid/missing", "reset", "score 0, PIR_SENSOR_MISSING, DEGRADED"],
            ["presence false", "reset", "score 0, PRESENCE_NOT_CONFIRMED"],
            ["presence true + motion 1", "reset", "score 0, NORMAL"],
            ["presence true + motion 0, <15 s", "누적", "score 0.5, NO_MOTION_DETECTED"],
            ["presence true + motion 0, >=15 s", "유지", "score 1.0, LONG_NO_MOTION"],
        ],
        [0.36, 0.20, 0.44],
    )
    w.h2("현재 presence_confirmed의 정확한 의미")
    w.body(
        "엔진 주석은 fusion presence를 암시하지만 실제 코드는 <b>mmwave_mr60.presence == 1</b>만 사용한다. "
        "presence key가 없으면 default가 1이다. Thermal NOT_HUMAN, CO2 OCCUPIED, PIR 자체 정보는 presence "
        "결정에 융합되지 않으며 risk_config의 presence_required 값도 분기에서 읽지 않는다."
    )
    w.callout(
        "확장 위험",
        "presence=0 packet에 resp_phase가 없으면 엔진은 Stream Adapter push를 호출하지 않아 과거 buffer가 남을 수 있다. "
        "입실·퇴실 event와 session reset을 별도 계약으로 만드는 편이 안전하다.",
        "amber",
    )
    w.h2("전자공학 관점")
    w.body(
        "PIR은 열복사 변화에 반응하는 event-like sensor이고, radar presence는 미세 움직임·vendor algorithm의 "
        "상태 추정이다. 두 신호의 bandwidth와 failure mode가 다르므로 단순 동의어로 취급하지 않는다."
    )
    w.finish()

    w.start(
        "17",
        "FMCW radar 신호 체인",
        "거리와 미세 움직임을 얻는 이론적 체인과 현재 코드의 시작점을 구분한다.",
        "PART II · 센서와 신호",
        "integrated_node/safenest_risk_engine.py:30-71; models/mmwave/sensor_stats_metadata_v0.1.0.json:2-19",
    )
    w.pipeline(
        [
            ("Chirp", "f(t) sweep"),
            ("Echo", "delay tau"),
            ("Mixer", "beat f_b"),
            ("Range FFT", "complex bins"),
            ("Phase", "micro-motion"),
        ]
    )
    w.body(
        "FMCW(frequency-modulated continuous wave)는 시간에 따라 주파수를 선형 변화시키는 chirp를 송신하고, "
        "지연된 echo와 현재 송신파를 혼합해 beat frequency를 만든다. 선형 chirp slope를 S [Hz/s], "
        "왕복 지연을 tau라고 하면 정지 표적의 이상적 beat는 f_b ≈ S tau이고 tau=2R/c다."
    )
    w.equation(
        [
            "S = B / T_chirp",
            "f_b = S * tau = S * (2R/c)",
            "R = c * f_b / (2S)",
            "ideal range resolution: delta_R = c / (2B)",
        ],
        "실제 시스템은 sampling, window, leakage, antenna, calibration, multi-target의 영향을 추가로 받는다.",
    )
    w.h2("현재 코드가 하지 않는 것")
    w.bullets(
        [
            "chirp bandwidth·slope·ADC rate로 range axis를 계산하지 않는다.",
            "raw ADC -> mixer/FFT를 구현하지 않고 선택적 complex rfft_frame이 이미 있다고 가정한다.",
            "MR60 UART receiver는 실제 complex rFFT를 파싱하지 않는다.",
        ]
    )
    w.callout(
        "학습 포인트",
        "FMCW 이론이 코드에 설명돼 있다는 사실과, 실제 MR60 raw 신호 체인이 연결됐다는 주장을 혼동하지 않는다.",
        "amber",
    )
    w.finish()

    w.start(
        "18",
        "Range bin과 거리축의 물리성",
        "FFT index를 meter로 바꾸려면 sensor configuration이 필요하다.",
        "PART II · 센서와 신호",
        "integrated_node/safenest_risk_engine.py:47-71,86-90",
    )
    w.h2("Range bin")
    w.body(
        "Range FFT의 k번째 complex coefficient는 특정 beat-frequency 구간을 대표한다. 이를 거리로 바꾸는 mapping은 "
        "chirp slope, ADC sampling, FFT length, zero padding, 보정 offset에 의존한다. 따라서 '64개 bin'만으로 실제 "
        "거리 분해능을 결정할 수 없다."
    )
    w.equation(
        [
            "f_b[k] = k * f_ADC / N_FFT    [one common mapping]",
            "r_range[k] = c * f_b[k] / (2S_chirp)",
            "display grid in current code = linspace(0, 5 m, 64)",
            "display spacing = 5/63 = 0.07937 m",
        ],
        "마지막 두 줄의 display spacing은 실제 radar delta_R=c/(2B)를 증명하지 않는다.",
    )
    w.table(
        ["현재 상수", "값", "주의"],
        [
            ["num_bins", 64, "입력 shape 가정"],
            ["distance grid", "0-5 m linear", "chirp 설정에서 유도되지 않음"],
            ["search range", "0.5-3.0 m", "valid_idx 선택용 정책"],
            ["default bin", 5, "grid상 약 0.397 m"],
        ],
        [0.24, 0.22, 0.54],
    )
    w.callout(
        "확장 계약",
        "rfft_frame과 함께 radar_config_id, bandwidth, chirp slope, ADC rate, FFT length, range_axis_m를 versioning한다. "
        "서로 다른 firmware의 bin index를 그대로 합치지 않는다.",
        "blue",
    )
    w.finish()

    w.start(
        "19",
        "I/Q, complex phase, 미세 변위",
        "크기만 남기면 호흡으로 인한 sub-wavelength 움직임 정보를 잃을 수 있다.",
        "PART II · 센서와 신호",
        "integrated_node/safenest_risk_engine.py:37-65; models/mmwave/sensor_stats_metadata_v0.1.0.json:19",
    )
    w.body(
        "한 range bin의 complex 값 X=I+jQ는 반사 amplitude와 phase를 함께 보존한다. I와 Q는 직교 성분이며 "
        "phase는 atan2(Q,I)다. 표적이 radar 방향으로 delta_R만큼 움직이면 monostatic 왕복 경로가 2 delta_R "
        "변하므로 phase 변화는 대략 4 pi delta_R/lambda다."
    )
    w.equation(
        [
            "X(t,r) = I(t,r) + j Q(t,r) = A(t,r) exp(j phi(t,r))",
            "phi(t,r) = atan2(Q, I)",
            "delta_phi ≈ 4 pi * delta_R / lambda",
            "at 60 GHz: lambda ≈ 5 mm; delta_R=1 mm -> delta_phi≈2.51 rad",
        ],
        "phase는 미세 변위에 민감하지만 wrap, motion artifact, multipath에도 민감하다.",
    )
    w.h2("Phase unwrap")
    w.body(
        "atan2 출력은 보통 -pi와 pi 사이에서 불연속적으로 접힌다. np.unwrap은 인접 차이가 pi를 넘으면 "
        "2 pi 배수를 보정해 연속 궤적을 만든다. sampling gap이나 큰 움직임이 있으면 올바른 배수를 놓칠 수 있으므로 "
        "unwrap 이전의 quality·gap 검사가 중요하다."
    )
    w.callout(
        "현재 semantic",
        "mmWave metadata는 입력을 resp_phase_unwrapped_clutter_removed로 선언한다. 실제 sensor adapter가 이 두 "
        "전처리를 재현해야만 tensor shape 이상의 의미 계약이 성립한다.",
        "red",
    )
    w.finish()

    w.start(
        "20",
        "Clutter map과 calibration 조건",
        "평균 복소 배경을 빼려면 '배경만 관측했다'는 조건이 먼저 입증돼야 한다.",
        "PART II · 센서와 신호",
        "integrated_node/safenest_risk_engine.py:30-45,283-292",
    )
    w.equation(
        [
            "clutter estimate: C_hat[r] = (1/N) * sum_t X[t,r]",
            "filtered frame: X_tilde[t,r] = X[t,r] - C_hat[r]",
        ],
        "복소 감산은 I와 Q를 각각 빼 amplitude와 phase를 함께 보존하는 vector subtraction이다.",
    )
    w.table(
        ["전제", "위반 시 결과", "현재 구현"],
        [
            ["무인 배경", "사람의 흉부 반사가 clutter에 흡수", "presence=0 gate 없음"],
            ["환경 정지", "문·팬·좌석 변화가 잔차로 남음", "재보정 trigger 없음"],
            ["동일 radar 설정", "bin·phase mapping 불일치", "config id 없음"],
            ["충분한 N", "noise variance가 크게 남음", "엔진은 history 15개 후 평균"],
            ["raw history", "필터 전후 frame 혼합 가능", "calibration 뒤 filtered frame을 같은 deque에 append"],
        ],
        [0.22, 0.38, 0.40],
    )
    w.h2("현재 순서")
    w.code(
        [
            "if not calibrated and len(history) >= 15:",
            "    clutter = mean(history)",
            "filtered = raw - clutter   # calibrated 이후",
            "history.append(filtered)",
        ]
    )
    w.callout(
        "권장 state machine",
        "UNCALIBRATED -> EMPTY_CONFIRMED -> COLLECTING_BACKGROUND -> CALIBRATED -> DRIFT_DETECTED -> RECALIBRATE를 "
        "명시하고, calibration artifact에 환경·firmware·시간·sample 수를 기록한다.",
        "blue",
    )
    w.finish()

    w.start(
        "21",
        "Adaptive chest-bin FFT band-energy의 해상도",
        "수식이 맞아도 관측 길이가 짧으면 선택 가능한 주파수 bin이 부족하다.",
        "PART II · 센서와 신호",
        "integrated_node/safenest_risk_engine.py:47-71,86-90,283-292",
    )
    w.equation(
        [
            "r* = argmax_r E_r,   E_r=sum_{f in [0.1,0.5]} |FFT(x_r)[f]|^2",
            "x_r = unwrap(angle(X_tilde[:,r])) - mean(unwrap(angle(X_tilde[:,r])))",
            "frequency spacing delta_f = fs/N",
            "N=30, fs=10 Hz -> delta_f=0.333 Hz",
            "N=10, fs=10 Hz -> delta_f=1 Hz; [0.1,0.5] mask has no FFT bin",
        ]
    )
    w.plot("spectrum")
    w.body(
        "코드 대응은 angle -> unwrap -> mean detrend -> rfft/rfftfreq -> band mask -> 제곱합이다. 정규화, window 길이, "
        "Hz당 스케일을 갖춘 PSD estimator가 아니라 <b>periodogram-like FFT band-energy proxy</b>다. history maxlen은 "
        "30이므로 3초 관측에서 호흡 대역 안의 양의 FFT bin은 사실상 0.333 Hz 하나뿐이다. "
        "N=10부터 함수를 실행하지만 이때 energy 합은 모든 range에서 0이어서 첫 valid bin이 선택될 수 있다. "
        "0.1 Hz 수준의 분해능에는 nominal 10초 이상이 필요하다."
    )
    w.table(
        ["추가 한계", "현재 상태"],
        [
            ["detrend", "평균만 제거; 선형 drift 제거 없음"],
            ["window function", "없음; spectral leakage 가능"],
            ["quality/SNR", "energy threshold·peak ratio 없음"],
            ["multi-target", "가장 큰 한 bin만 선택"],
            ["downstream", "active_chest_bin telemetry만 갱신; resp_phase 생성 안 함"],
        ],
        [0.28, 0.72],
    )
    w.finish()

    w.start(
        "22",
        "MR60BHA2의 현재 연결성",
        "UART parser가 있다는 사실과 AI 입력 pipeline이 연결됐다는 사실을 구분한다.",
        "PART II · 센서와 신호",
        "mr60/sensor_receiver.py:83-157,191-237; mr60/sensor_simulator.py:93-175",
    )
    w.table(
        ["값", "Simulator/Receiver", "RiskEngine 요구", "연결 상태"],
        [
            ["presence", "parse", "presence", "개념상 매핑 가능, 미연결"],
            ["motion/amplitude", "parse", "PIR 별도 motion", "schema 결정 필요"],
            ["breath_rate", "parse", "breath_rpm", "이름 변환 필요, 미연결"],
            ["heart_rate", "parse", "heart_bpm", "이름 변환 필요, 미연결"],
            ["distance", "parse", "telemetry 후보", "RiskEngine packet에 없음"],
            ["apnea", "생산 안 함", "즉시 safety flag", "결손"],
            ["resp_phase", "생산 안 함", "300-sample AI input", "결손"],
            ["real rfft", "parse 안 함", "optional complex[64]", "합성값만 별도 생성"],
        ],
        [0.18, 0.27, 0.29, 0.26],
    )
    w.code(
        [
            "MR60 simulator -> TCP bytes -> sensor_receiver -> terminal dashboard",
            "                                                    X",
            "VirtualSensorStreamer -> Python packet -> SafeNestRiskEngine -> GUI",
        ]
    )
    w.callout(
        "통합 Adapter의 책임",
        "protocol checksum·sequence·source timestamp를 검증하고, vendor output이 sensor-derived인지 직접 계산인지 "
        "provenance를 붙인 뒤 canonical packet을 publish해야 한다.",
        "amber",
    )
    w.finish()

    w.start(
        "23",
        "mmWave 10 Hz·300 sample window",
        "tensor의 세 축과 시간·주파수 해상도를 계산한다.",
        "PART II · 센서와 신호",
        "models/model_manifest.json:67-95; models/mmwave/sensor_stats_metadata_v0.1.0.json:2-19",
    )
    w.table(
        ["계약", "값", "의미"],
        [
            ["shape", "[1,300,1]", "batch 1, time 300, feature 1"],
            ["sample rate", "10 Hz", "이상적 sample interval 0.1 s"],
            ["coverage", "29.9 s", "첫 sample과 300번째 sample 사이"],
            ["class map", "NORMAL / RAPID_OR_ABNORMAL\nAPNEA", "300-sample window 분류; label duration·작성 규칙은 미확인"],
            ["Nyquist", "5 Hz", "10 Hz sampling의 이론적 상한"],
            ["delta_f", "0.0333 Hz", "300-point DFT bin 간격, 약 2 rpm"],
            ["respiration band", "0.1-0.5 Hz", "약 6-30 rpm"],
        ],
        [0.20, 0.29, 0.51],
    )
    w.h2("RPM과 phase의 차이")
    w.body(
        "breath_rpm은 일정 구간의 파형에서 peak 주기 등을 추정한 <b>요약 통계</b>이고, resp_phase는 매 시점의 "
        "연속 sample이다. RPM 16을 300번 복제하면 16 rpm 호흡파가 아니라 DC sequence가 되며, phase waveform의 "
        "모양·주기·motion artifact 정보를 모두 잃는다."
    )
    w.callout(
        "첫 추론 시각",
        "정확한 10 Hz 생산과 t=0 첫 sample을 가정하면 300번째 sample은 t=29.9 s에 온다. 그러나 upstream "
        "apnea flag와 RPM 규칙은 AI window를 기다리지 않고 동작한다.",
        "blue",
    )
    w.finish()

    w.start(
        "24",
        "Stream Adapter state machine",
        "ring buffer는 단순 저장소가 아니라 입실·시간·신선도 조건을 가진 gate다.",
        "PART II · 센서와 신호",
        "adapters/mmwave_stream_adapter.py:24-110; integrated_node/safenest_risk_engine.py:242-281",
    )
    w.table(
        ["event", "동작", "reason / 상태"],
        [
            ["presence != 1", "buffer·timestamp clear", "MMWAVE_PRESENCE_NOT_DETECTED"],
            ["None/NaN/Inf value", "reject, 기존 buffer 유지", "MMWAVE_VALUE_NAN_OR_INF"],
            ["non-finite time", "reject", "MMWAVE_TIMESTAMP_NON_FINITE"],
            ["dt <= 0", "reject, 기존 buffer 유지", "MMWAVE_TIMESTAMP_NON_MONOTONIC"],
            ["dt > 0.5 s", "buffer clear 후 reject", "MMWAVE_STREAM_GAP_TOO_LARGE"],
            ["valid sample", "deque append, max 300", "accepted"],
            ["len < 300", "AI 미실행", "MMWAVE_WINDOW_NOT_READY"],
            ["age > 2 s", "get_window=None", "MMWAVE_WINDOW_STALE"],
        ],
        [0.26, 0.36, 0.38],
    )
    w.h2("중요한 비검사 항목")
    w.bullets(
        [
            "dt가 0.1초 근처인지 검사하지 않는다. 0&lt;dt&lt;=0.5이면 허용한다.",
            "300개 window의 실제 coverage가 29.9초인지 확인하거나 재샘플링하지 않는다.",
            "ready 이후 매 새 sample마다 299/300이 겹치는 window를 추론한다. nominal 10 inference/s다.",
            "get_status는 wall clock을 쓰므로 replay timestamp와 혼용할 때 해석을 주의해야 한다.",
        ]
    )
    w.callout(
        "설계 질문",
        "실시간 경로가 strict 10 Hz를 요구할지, jitter buffer와 resampler를 둘지, 또는 model을 timestamp-aware하게 "
        "바꿀지 선택해야 한다. 선언된 sample_rate만으로는 계약이 강제되지 않는다.",
        "red",
    )
    w.finish()

    w.start(
        "25",
        "CSV Adapter와 dataset windowing",
        "offline Adapter는 실시간 경로보다 강한 시간 grid와 session 격리를 구현한다.",
        "PART II · 센서와 신호",
        "adapters/mmwave_csv_adapter.py:23-126",
    )
    w.numbered(
        [
            "timestamp_s와 resp_phase column 존재를 확인한다.",
            "subject_id·session_id가 없으면 UNKNOWN을 채우고 group별로 분리한다.",
            "timestamp 중복·역행이 있는 session은 전체를 건너뛴다.",
            "raw 300개 단위 후보에서 0.5초 초과 gap과 NaN/Inf를 거부한다.",
            "0.1초 target grid를 만들고 np.interp로 선형 재샘플링한다.",
            "총 보간 공백 비율이 5%를 넘으면 window를 버린다.",
            "raw 300행 후보를 30행씩 이동하며 MMWaveWindow를 yield한다.",
        ]
    )
    w.table(
        ["잘한 경계", "남은 경계"],
        [
            ["subject/session 간 window 누수 방지", "label은 첫 sample 하나를 사용; window 내 일관성 미검사"],
            ["정확한 10 Hz target grid", "downsampling 전 anti-alias low-pass 없음"],
            ["gap·interpolation budget", "거부 이유와 count를 report로 반환하지 않음"],
            ["source field", "presence·device·firmware·ground truth provenance 부족"],
            ["row-based stride", "원본 cadence가 10 Hz가 아니면 실제 span/stride가 30초/3초가 아님"],
        ],
        [0.42, 0.58],
    )
    w.equation(
        [
            "implementation: candidate length=300 raw rows, stride=30 raw rows",
            "only if raw cadence=10 Hz: span≈30 s, stride≈3 s, overlap=90%",
        ],
        "각 후보 안의 target grid는 0.1초지만 시작 위치는 시간 기준이 아니다. gap_fraction도 보간 sample 수가 아니라 "
        "초과 gap 시간 합/30초의 proxy다. 겹치는 window의 무작위 split은 누수를 만든다.",
    )
    w.finish()

    w.start(
        "26",
        "Model·Manifest·Metadata·Provenance",
        "네 계약은 서로 대체할 수 없으며 모두 있어야 model output의 의미를 복원할 수 있다.",
        "PART III · Edge AI 계약",
        "models/model_manifest.json:1-110; inference/model_registry.py:23-93",
    )
    w.table(
        ["계약", "답하는 질문", "SafeNest 예"],
        [
            ["Model", "어떤 연산 graph와 weights를 실행하는가?", ".tflite binary"],
            ["Manifest", "어느 artifact·shape·dtype·scale·class인가?", "model_id, SHA-256, [1,300,1]"],
            ["Metadata", "학습 때 숫자를 어떻게 해석·변환했는가?", "mean/std, feature order, semantic"],
            ["Provenance", "어떤 데이터·코드·fallback이 이 결과를 만들었는가?", "version, source, fallback_reason"],
            ["Validation", "어느 환경·시험까지 통과했는가?", "local invoke true, Pi false"],
        ],
        [0.18, 0.45, 0.37],
    )
    w.code(
        [
            "sensor value --(metadata preprocessing)--> float tensor",
            "float tensor --(manifest quantization)--> int8 tensor",
            "int8 tensor --(model)--> int8 output",
            "output --(manifest class_map)--> named prediction",
            "prediction --(provenance)--> auditable system decision",
        ]
    )
    w.h2("현재 manifest의 한계")
    w.bullets(
        [
            "artifact path·hash·tensor 중심이며 thermal/CO2의 training data·split·limitation 필드가 부족하다.",
            "validation block은 mmWave에만 있고 모델별 evidence 형식이 균일하지 않다.",
            "manifest가 있어도 모든 wrapper가 hash·quantization을 동일하게 검증하는 것은 아니다.",
        ]
    )
    w.finish()

    w.start(
        "27",
        "Z-score와 INT8 양자화",
        "정규화는 분포를 맞추고, 양자화는 제한된 정수 격자에 표현한다.",
        "PART III · Edge AI 계약",
        "inference/mmwave_interpreter.py:159-199; models/model_manifest.json:78-94",
    )
    w.equation(
        [
            "normalization: z = (x - mu_feature) / sigma_feature",
            "quantization: q_int8 = clip(rint(z/s_quant + zero), -128, 127)",
            "dequantization: z_hat = s_quant * (q_int8 - zero)",
            "quantization error before clipping: |z_hat-z| <= s_quant/2",
        ]
    )
    w.h2("mmWave 한 sample 예")
    w.table(
        ["단계", "계산", "결과"],
        [
            ["raw", "x=2.0", "resp_phase unit은 metadata semantic에 의존"],
            ["Z-score", "(2.0-0.006092)/2.501384", "z≈0.7971"],
            ["quantize", "np.rint(0.7971/0.0325986 - 13)", "q_int8≈11"],
            ["dequantize", "0.0325986*(11+13)", "z_hat≈0.7824"],
            ["error", "z_hat-z", "약 -0.0147, clipping 없을 때"],
        ],
        [0.20, 0.50, 0.30],
    )
    w.h2("Clipping")
    w.body(
        "q_int8 계산값이 범위를 넘으면 서로 다른 큰 입력이 같은 끝값으로 합쳐진다. 현재 wrapper의 np.rint는 정확히 "
        "x.5인 tie를 가장 가까운 짝수로 보내는 round-to-nearest, ties-to-even이다. manifest의 scale은 tensor 전체에 "
        "하나인 per-tensor 방식이며, 축별 scale을 쓰는 per-axis와 다르다. saturation ratio는 clip된 원소 수/전체 원소 수다."
    )
    w.callout(
        "두 단계 혼동 금지",
        "mu_feature/sigma_feature를 바꾸는 것은 s_quant를 바꾸는 것과 다르다. 출력 score를 합 1로 다시 나누는 것도 "
        "표시상 정규화일 뿐 confidence calibration을 복구하지 않는다. 어느 것도 근거 없이 임의 수정하지 않는다.",
        "red",
    )
    w.finish()

    w.start(
        "28",
        "CO2 INT8 포화 범위 계산",
        "manifest quantization과 metadata를 합치면 배포 입력의 유효 표현 범위를 계산할 수 있다.",
        "PART III · Edge AI 계약",
        "models/model_manifest.json:48-60; models/co2/co2_scaling_metadata_v0.1.0.json:2-16",
    )
    w.equation(
        [
            "q_min=-128, q_max=127, s=0.00582845, zero=57",
            "z_min=s*(-128-57)≈-1.0783",
            "z_max=s*(127-57)≈0.4080",
            "raw range for feature j: x in [mean_j + scale_j*z_min, mean_j + scale_j*z_max]",
        ]
    )
    w.table(
        ["feature", "metadata mean/std", "비포화 raw 범위(약)", "일상 입력 예"],
        [
            ["slope", "0.011 / 4.373", "-4.70 to 1.80 ppm/min", "15는 상단 포화"],
            ["humidity", "25.731 / 5.532", "19.77 to 27.99 %", "45%는 상단 포화"],
            ["CO2", "606.48 / 314.39", "267.5 to 734.7 ppm", "1000 ppm은 상단 포화"],
        ],
        [0.18, 0.26, 0.31, 0.25],
    )
    w.body(
        "직접 실행하면 (slope=15, humidity=45, CO2=1000)과 더 큰 위험 입력이 모두 여러 축에서 q_int8=127로 "
        "포화될 수 있다. 모델은 두 입력의 크기 차이를 볼 수 없고, occupancy output이 정상적으로 반환돼도 입력 "
        "정보가 이미 소실됐을 수 있다."
    )
    w.callout(
        "P0 감사 항목",
        "converter에 이미 standardized 대표 데이터가 들어갔는지, runtime에서 Z-score를 다시 적용해 이중 정규화하는지, "
        "representative range가 왜 좁은지 학습 notebook과 converter를 복구해 확인해야 한다.",
        "red",
    )
    w.finish()

    w.start(
        "29",
        "TFLite 실행과 confidence 해석",
        "invoke 성공, softmax-like 출력, 실제 정확도는 세 개의 다른 주장이다.",
        "PART III · Edge AI 계약",
        "inference/thermal_interpreter.py:150-199; inference/mmwave_interpreter.py:189-245",
    )
    w.pipeline(
        [
            ("Validate", "shape·finite"),
            ("Encode", "norm·int8"),
            ("Invoke", "TFLite graph"),
            ("Decode", "dequantize"),
            ("Interpret", "class·meta"),
        ]
    )
    w.table(
        ["용어", "의미", "의미가 아닌 것"],
        [
            ["invoke success", "runtime이 graph를 오류 없이 실행", "입력 semantic·class 정답"],
            ["probability vector", "출력 값을 clip·sum-normalize한 벡터", "well-calibrated probability 보장"],
            ["confidence", "현재 argmax component", "현장 accuracy 또는 안전 확률"],
            ["latency_ms", "대부분 invoke 주변 시간", "sensor-to-alarm end-to-end latency"],
            ["fallback", "AI 대신 heuristic 또는 no-input 상태", "정상 AI와 동일한 증거 등급"],
        ],
        [0.22, 0.38, 0.40],
    )
    w.h2("왜 다시 합을 1로 만드는가")
    w.body(
        "INT8 softmax output은 scale 1/256의 격자 때문에 합이 정확히 1이 아닐 수 있다. wrapper는 음수를 0으로 "
        "clip하고 합으로 나눈다. 이는 표시를 편하게 하지만, model calibration을 새로 증명하지는 않는다."
    )
    w.callout(
        "검증 순서",
        "runtime test -> labeled holdout confusion matrix -> calibration/ECE -> operating threshold -> false alarm/hour와 "
        "miss cost의 순서로 증거를 쌓는다.",
        "blue",
    )
    w.finish()

    w.start(
        "30",
        "세 Wrapper의 검증 강도 감사",
        "공통 interface처럼 보여도 initialization과 fault 처리의 깊이가 다르다.",
        "PART III · Edge AI 계약",
        "inference/thermal_interpreter.py:57-104; inference/co2_interpreter.py:49-109; inference/mmwave_interpreter.py:58-157",
    )
    w.table(
        ["검사", "Thermal", "CO2", "mmWave"],
        [
            ["model file required", "예, 없으면 init 실패", "예, 없으면 init 실패", "아니오, heuristic fallback object 유지"],
            ["metadata required", "별도 없음", "없으면 hard-coded fallback", "예, 없으면 init 실패"],
            ["SHA-256 at init", "아니오", "아니오", "예"],
            ["shape/dtype", "예", "아니오", "예"],
            ["scale/zero point", "아니오", "아니오", "예"],
            ["finite input", "예", "직접 검사 없음", "예"],
            ["output finite", "예", "직접 검사 없음", "직접 검사 없음"],
            ["fallback provenance", "engine exception reason", "engine exception reason", "model_id·reason 명시"],
        ],
        [0.28, 0.24, 0.24, 0.24],
    )
    w.h2("Registry health")
    w.body(
        "ModelRegistry.health는 Thermal·CO2에 대해 file+interpreter만 보고, mmWave는 SHA까지 확인한다. 엔진 초기화는 "
        "health 결과를 gate로 사용하지 않는다. wrapper 생성 자체가 실패해 runner=None이면 일부 경로에서 structured "
        "reason 없이 model_meta NOT_RUN만 남고 system_status가 OK일 수 있다."
    )
    w.callout(
        "권장 공통 계약",
        "모든 wrapper가 load() -> validate_artifact() -> validate_tensor() -> health() -> predict()를 같은 reason taxonomy로 "
        "구현하고, registry health를 엔진 system_status에 강제 전파한다.",
        "red",
    )
    w.finish()

    w.start(
        "31",
        "현재 세 모델 카드",
        "모델의 역할보다 증거 등급과 연결 상태를 먼저 읽는다.",
        "PART III · Edge AI 계약",
        "models/model_manifest.json:6-108; reports/TEST_RESULTS_20260726.md:13-45",
    )
    w.table(
        ["항목", "Thermal", "CO2", "mmWave"],
        [
            ["status", "candidate", "candidate", "local_runtime_validated"],
            ["role", "posture 3-class", "occupancy 2-class", "respiration 3-class"],
            ["input", "[1,62,80,1] int8", "[1,3] int8", "[1,300,1] int8"],
            ["artifact size", "318,184 B", "4,464 B", "466,616 B"],
            ["risk link", "FALL>=0.8 emergency", "class 미반영; ppm rule", "class1 score, class2 timer"],
            ["strong evidence", "hash test·tensor·smoke", "predict smoke", "hash·tensor·real invoke"],
            ["critical gap", "independent labels·NOT_HUMAN", "INT8 saturation·time scale", "real sensor·Class 1 provenance"],
            ["Pi evidence", "없음", "없음", "없음"],
        ],
        [0.20, 0.27, 0.26, 0.27],
    )
    w.h2("모델 크기보다 중요한 것")
    w.bullets(
        [
            "동일 class name이더라도 label 정의와 minimum duration이 다르면 다른 task다.",
            "INT8라는 사실만으로 속도·전력·정확도 이득을 보장하지 않는다. target hardware에서 측정한다.",
            "confidence threshold는 validation evidence와 함께 versioning해야 한다.",
            "model output이 실제 global risk에 사용되는지 별도의 연결성 표가 필요하다.",
        ]
    )
    w.callout(
        "금지할 요약",
        "'세 모델 통합 완료' 대신 '세 wrapper 로컬 호출 경로가 있고, Thermal/mmWave 일부 출력만 위험 규칙에 연결되며 "
        "실센서 통합은 미완료'라고 표현해야 정확하다.",
        "amber",
    )
    w.finish()

    w.start(
        "32",
        "센서별 부분 점수 함수",
        "각 modality는 먼저 0-1 score와 reason으로 변환된 뒤 가중합에 들어간다.",
        "PART IV · 위험 융합과 상태",
        "risk/risk_rules.py:87-256; risk/risk_config.json:4-27",
    )
    w.table(
        ["modality", "조건", "부분 score / 효과"],
        [
            ["Respiration", "None / valid=False / NaN", "0.5 + RESP_SENSOR_FAULT"],
            ["Respiration", "문자열 또는 +Inf/-Inf", "일부 경로에서 tick 전체 예외 가능; structured fault 아님"],
            ["Respiration", "12-20 rpm, AI class 0", "0"],
            ["Respiration", "범위 밖 또는 AI class 1", "0.75"],
            ["Respiration", "apnea flag 또는 확인된 candidate", "1 + emergency"],
            ["Environment", "invalid", "0.2 + CO2_SENSOR_FAULT"],
            ["Environment", "valid ppm", "clip((ppm-500)/2000,0,1)"],
            ["Heart rate", "HR>110", "min(1,(HR-105)/25)"],
            ["Heart rate", "HR<55", "min(1,(60-HR)/25)"],
            ["Posture", "FALL and confidence>=0.8", "1 + emergency"],
            ["Posture", "그 외 valid class", "0"],
            ["Motion", "presence+motion0 <15 s", "0.5"],
            ["Motion", "presence+motion0 >=15 s", "1.0"],
        ],
        [0.22, 0.45, 0.33],
    )
    w.plot("risk")
    w.callout(
        "부분 status와 전역 status",
        "CO2 component의 status가 DANGER여도 전역 score는 가중치 때문에 NORMAL일 수 있다. 부분 RuleResult의 "
        "status 문자열은 전역 level을 직접 강제하지 않는다.",
        "red",
    )
    w.finish()

    w.start(
        "33",
        "가중 융합 계산 예제",
        "점수 기여도를 직접 계산하면 reason과 최종 level의 차이가 보인다.",
        "PART IV · 위험 융합과 상태",
        "risk/risk_rules.py:312-334; risk/risk_config.json:18-27",
    )
    w.equation(
        [
            "R_raw = 100 * (0.30*S_resp + 0.25*S_env + 0.20*S_HR",
            "                 + 0.15*S_fall + 0.10*S_motion)",
        ]
    )
    w.h2("비응급 조합")
    w.table(
        ["관측", "score", "weight", "기여점수"],
        [
            ["호흡 8 rpm", 0.75, 0.30, 22.50],
            ["CO2 2000 ppm", 0.75, 0.25, 18.75],
            ["HR 130 bpm", 1.00, 0.20, 20.00],
            ["Thermal NORMAL", 0.00, 0.15, 0.00],
            ["15초 무움직임", 1.00, 0.10, 10.00],
            ["합계", "", "", 71.25],
        ],
        [0.38, 0.18, 0.18, 0.26],
    )
    w.body(
        "순수 RiskRulesEvaluator에서는 35 이상 75 미만이므로 CAUTION이다. 통합엔진에서는 이 raw 값이 최근 6개 "
        "평균과 alpha=0.25 IIR을 거치므로, 이전 상태가 낮았다면 표시 risk는 71.25보다 작고 DANGER 진입은 더 늦다."
    )
    w.callout(
        "Emergency가 하나라도 있으면",
        "같은 조합에서 Thermal FALL confidence 0.8 이상 또는 apnea override가 발생하면 가중합을 사용하지 않고 "
        "즉시 R=100, DANGER로 간다.",
        "red",
    )
    w.finish()

    w.start(
        "34",
        "Quality Gate의 현재 의미",
        "Q_quality,i는 계산되지만 현재 위험 가중치를 동적으로 재정규화하지 않는다.",
        "PART IV · 위험 융합과 상태",
        "integrated_node/safenest_risk_engine.py:98-125,360-377; risk/risk_rules.py:312-319",
    )
    w.table(
        ["sensor", "현재 Q_quality=1 조건", "비정상 Q_quality", "미검사"],
        [
            ["Thermal", "ndarray (62,80)", 0.0, "finite·range·age"],
            ["CO2", "300<=ppm<=10000", 0.2, "finite humidity·age"],
            ["mmWave", "breath_rpm와 apnea key", 0.0, "value range·presence·phase readiness"],
            ["PIR", "motion key and not None", 0.5, "0/1 범위는 뒤에서 별도"],
        ],
        [0.20, 0.32, 0.16, 0.32],
    )
    w.h2("문서에서 혼동하기 쉬운 두 식")
    w.equation(
        [
            "CURRENT: R_risk = 100 * sum_i (w_i * S_i)",
            "NOT IMPLEMENTED: R_risk = 100*sum_i(Q_quality,i*w_i*S_i)/sum_i(Q_quality,i*w_i)",
        ],
        "현재 q_gate dict는 실행 여부·telemetry에 쓰이지만 risk 식의 weight를 바꾸지 않는다.",
    )
    w.table(
        ["결측 modality", "현재 점수 영향"],
        [
            ["호흡 fault", "부분 score 0.5 -> 15점 + system degradation"],
            ["CO2 fault", "부분 score 0.2 -> 5점 + system degradation"],
            ["HR/Thermal/PIR missing", "대부분 0점 + system degradation"],
            ["전체 missing", "특수 분기 R_risk=0, status=FAULT"],
        ],
        [0.36, 0.64],
    )
    w.callout(
        "설계 선택",
        "동적 재정규화는 가용 센서로 계속 판단하는 fail-operational 장점이 있지만, 고장난 강한 증거를 제거해 "
        "과신할 수 있다. hazard 분석 없이 수식만 추가하지 않는다.",
        "amber",
    )
    w.finish()

    w.start(
        "35",
        "Emergency Override와 timer",
        "즉시 경로와 지속시간 확인 경로를 분리해 latency budget을 읽는다.",
        "PART IV · 위험 융합과 상태",
        "risk/risk_rules.py:87-146,193-210; integrated_node/safenest_risk_engine.py:326-331",
    )
    w.table(
        ["trigger", "확인 시간", "동작"],
        [
            ["upstream apnea==1", "0 s*", "RiskRules의 time 검증 뒤 EMERGENCY_APNEA"],
            ["breath_rpm<=0.5", "2.0 s*", "유효 numeric timestamp 경로에서 확인"],
            ["mmWave AI class 2", "2.0 s*", "유효 time 경로; 같은 apnea timer 공유"],
            ["Thermal class 2, conf>=0.8", "0 s", "즉시 EMERGENCY_FALL"],
        ],
        [0.42, 0.20, 0.38],
    )
    w.code(
        [
            "sample_ts = packet.get('timestamp_s', packet.get('timestamp', time.time()))",
            "timestamp key absent -> wall clock; explicit timestamp_s=None -> None 유지",
            "RiskRules reached + non-None numeric time -> apnea flag보다 먼저 validate",
            "resp_phase + non-numeric time -> Adapter np.isfinite에서 TypeError 가능",
            "candidate true at t0 -> elapsed = 0",
            "candidate true at t0+2.0 -> emergency",
            "RiskRules time=None and dt_s=None -> elapsed=2.0, 첫 candidate 즉시 확정",
        ]
    )
    w.h2("Timestamp 주의")
    w.bullets(
        [
            "정상 통합 경로의 timer는 sample count가 아니라 유효 timestamp 차이를 써 packet rate 변화에 덜 민감하다.",
            "validate_timestamp의 previous는 직전 sample이 아니라 candidate 시작 시각이어서 모든 단조성을 보장하지 않는다.",
            "RiskRules까지 도달한 non-numeric/non-finite time은 apnea==1보다 먼저 FAULT다. 단 resp_phase가 있으면 Adapter가 먼저 crash할 수 있다.",
            "명시적 None 또는 Evaluator 단독 무시간 호출은 RPM/class2 candidate를 첫 호출에 확정한다. 따라서 '2초'는 유효 time 경로에 한정한다.",
            "generic SENSOR_TIMESTAMP_NON_MONOTONIC reason이 최종 sensor_status mapping에서 누락되는 경우가 있다.",
            "upstream apnea flag의 생성 근거·freshness가 packet에 없으므로 즉시 경로의 신뢰 경계가 불명확하다.",
        ]
    )
    w.callout(
        "Latency 정의",
        "'2초 내 경보'는 AI 30초 startup, sensor update 주기, packet transport, debounce, actuator 시간을 분해해 "
        "정의해야 한다. 코드의 timer 하나만으로 end-to-end 요구를 만족했다고 말할 수 없다.",
        "red",
    )
    w.finish()

    w.start(
        "36",
        "Moving mean, IIR, Hysteresis",
        "노이즈 억제는 응답 지연과 상태 기억을 만든다.",
        "PART IV · 위험 융합과 상태",
        "integrated_node/safenest_risk_engine.py:326-344; risk/risk_config.json:18-28",
    )
    w.equation(
        [
            "M[k] = mean(last up to 6 raw risk values)",
            "R[k] = R[k-1] + 0.25*(M[k]-R[k-1])",
            "     = 0.75*R[k-1] + 0.25*M[k]",
            "IIR-only 63% response ≈ -1/ln(0.75) = 3.48 packets",
        ],
        "앞단 6-point moving mean 때문에 전체 응답은 IIR 단독보다 더 느리다.",
    )
    w.plot("filter")
    w.table(
        ["이전 상태", "새 R", "다음 상태"],
        [
            ["NORMAL/CAUTION", "R<40", "NORMAL"],
            ["NORMAL/CAUTION", "40<=R<75", "CAUTION"],
            ["NORMAL/CAUTION", "R>=75", "DANGER"],
            ["DANGER", "R>65", "DANGER 유지"],
            ["DANGER", "35<R<=65", "CAUTION"],
            ["DANGER", "R<=35", "NORMAL"],
        ],
        [0.30, 0.28, 0.42],
    )
    w.callout(
        "정책 이중화",
        "risk_config의 normal_max=35와 통합엔진 일반 진입 40이 다르다. 엔진은 RiskRulesEvaluator.level을 버리고 "
        "40/75를 hard-code하므로 config의 35를 바꿔도 최종 CAUTION 진입은 변하지 않는다.",
        "red",
    )
    w.finish()

    w.start(
        "37",
        "상태의 세 축을 함께 읽는다",
        "위험 수준, 장비 건강, 원인 코드는 서로 다른 질문에 답한다.",
        "PART IV · 위험 융합과 상태",
        "risk/risk_rules.py:258-335; integrated_node/safenest_risk_engine.py:360-415",
    )
    w.table(
        ["축", "질문", "대표 값"],
        [
            ["status_str", "현재 관측 위험은 어느 수준인가?", "NORMAL/CAUTION/DANGER/FAULT"],
            ["system_status", "그 판단 경로는 얼마나 가용한가?", "OK/DEGRADED/FAULT"],
            ["reasons", "어떤 증거·고장 때문에 그렇게 됐는가?", "HIGH_CO2_DANGER, WINDOW_NOT_READY"],
            ["model_meta", "어느 artifact/fallback이 결과를 냈는가?", "source, version, class, latency"],
        ],
        [0.22, 0.43, 0.35],
    )
    w.table(
        ["조합", "올바른 해석"],
        [
            ["NORMAL + OK", "현재 낮은 위험, 주요 경로 가용"],
            ["NORMAL + DEGRADED", "위험 증거는 낮지만 판단 범위가 제한됨"],
            ["DANGER + OK", "가용한 경로가 응급 증거를 냄"],
            ["DANGER + DEGRADED", "응급 증거와 일부 고장이 동시에 존재"],
            ["R_risk=0 + FAULT", "안전이 아니라 판단 불가"],
        ],
        [0.32, 0.68],
    )
    w.h2("현재 propagation hole")
    w.body(
        "하위 RuleResult가 SENSOR_TIMESTAMP_NON_MONOTONIC을 내도 evaluate_system의 reason mapping은 일부 다른 이름만 "
        "인식해 최종 system_status=OK가 될 수 있다. 또한 계산한 sensor_status dict는 최종 engine output에 포함되지 않는다."
    )
    w.callout(
        "UI 원칙",
        "risk 색 하나로 health를 덮지 않는다. 모든 화면·로그·경보 API는 status_str, system_status, reasons를 "
        "동시에 표시해야 한다.",
        "red",
    )
    w.finish()

    w.start(
        "38",
        "현재 출력 계약과 관측성",
        "확장 가능한 시스템은 결과뿐 아니라 시간·출처·품질·상태 전이를 설명해야 한다.",
        "PART IV · 위험 융합과 상태",
        "integrated_node/safenest_risk_engine.py:360-415; integrated_node/safenest_integrated_plotter.py:119-258",
    )
    w.table(
        ["현재 field", "내용", "관측 가치"],
        [
            ["risk_score/status", "0-100, level, emergency", "최종 위험 표시"],
            ["reasons", "deduplicated reason list", "이상·fault 설명"],
            ["sensor_quality", "coarse q gate", "입력 기본 유효성"],
            ["system_status", "OK/DEGRADED/FAULT", "경로 건강"],
            ["derived_metrics", "CO2 slope, presence, window count", "중간 상태"],
            ["model_meta.*", "source/version/class/confidence/latency/fallback", "AI provenance"],
            ["active_chest_*", "bin·display distance", "rFFT telemetry"],
        ],
        [0.25, 0.37, 0.38],
    )
    w.h2("확장 전에 추가할 envelope")
    w.code(
        [
            "schema_version, decision_id, device_id, session_id",
            "evaluated_at, input_watermark, per_sensor{measured_at, age_ms, seq}",
            "policy_version, manifest_digest, calibration_ids",
            "component_scores, component_status, raw_risk, filtered_risk",
            "state_transition{from,to,reason}, processing_latency_ms",
        ]
    )
    w.callout(
        "왜 필요한가",
        "분산 node에서 재시도·순서 뒤바뀜·중복 packet이 생기면 decision_id와 timestamp 없이 원인을 재현하기 어렵다. "
        "안전 시스템의 observability는 디버그 편의가 아니라 검증 가능성의 일부다.",
        "blue",
    )
    w.finish()

    w.start(
        "39",
        "정상 startup 시나리오 추적",
        "AI window가 준비되지 않은 초기 상태를 센서 고장이나 안전 상태와 혼동하지 않는다.",
        "PART IV · 위험 융합과 상태",
        "integrated_node/safenest_risk_engine.py:242-281,303-377; tests/test_three_model_integration.py:43-52",
    )
    w.table(
        ["시각", "mmWave buffer", "AI 상태", "위험·건강 해석"],
        [
            ["첫 sample, tau=0", "1/300", "WINDOW_NOT_READY", "runner 존재 시 system DEGRADED"],
            ["t=10.0 s", "101/300", "WINDOW_NOT_READY", "정상 RPM이어도 AI 증거 없음"],
            ["t=29.8 s", "299/300", "WINDOW_NOT_READY", "마지막 startup sample"],
            ["t=29.9 s", "300/300", "TFLite invoke", "fallback 없으면 mmWave AI OK"],
            ["t=30.0 s 이후", "rolling 300", "매 tick invoke", "99.67% overlap window"],
        ],
        [0.18, 0.23, 0.28, 0.31],
    )
    w.h2("한 정상 packet의 병렬 경로")
    w.bullets(
        [
            "Thermal은 매 packet 단일 frame을 추론한다.",
            "CO2 AI는 첫 sample부터 slope=0으로 호출될 수 있지만 class는 risk에 반영되지 않는다.",
            "호흡·HR·PIR rule은 mmWave 30초 window와 독립적으로 즉시 동작한다.",
            "기본 mmWave runner와 resp_phase가 있는 동안 첫 299 sample은 WINDOW_NOT_READY가 붙으므로 "
            "NORMAL+DEGRADED가 가능 수준이 아니라 현재 경로에서 결정적이다.",
        ]
    )
    w.callout(
        "테스트 해석",
        "표의 tau는 첫 sample을 0으로 재기준화한 상대시간이다(현재 Virtual Streamer 내부 첫 t는 0.1 s). integration의 "
        "normal scenario는 주로 status_str을 확인하며 startup system_status==OK를 보장하지 않는다.",
        "amber",
    )
    w.finish()

    w.start(
        "40",
        "낙상과 무호흡 시나리오 추적",
        "두 경로 모두 R=100이지만 trigger와 신뢰 경계가 다르다.",
        "PART IV · 위험 융합과 상태",
        "risk/risk_rules.py:87-146,193-210; tests/test_fault_injection.py:58-78",
    )
    w.table(
        ["경로", "필요 입력", "지연", "주의"],
        [
            ["Thermal fall", "class2 + confidence>=0.8", "단일 frame 즉시", "temporal debounce 없음"],
            ["Upstream apnea flag", "apnea==1; Rule 도달+valid time", "검증 후 즉시", "bad time은 FAULT; Adapter가 먼저 crash 가능"],
            ["RPM apnea", "breath_rpm<=0.5", "valid time이면 2.0 s", "explicit None이면 첫 candidate 즉시 확정"],
            ["AI apnea", "mmWave class2", "valid time이면 window 후 2.0 s", "첫 AI nominal 29.9 s; None은 즉시 확정"],
        ],
        [0.22, 0.31, 0.20, 0.27],
    )
    w.code(
        [
            "if emergency:",
            "    R = 100",
            "    curr_smoothed_r = 100",
            "    status = DANGER",
            "    # moving mean and IIR bypassed",
            "# absent timestamp key -> wall clock; explicit None stays None",
            "# resp_phase + nonnumeric time may crash in StreamAdapter before rules",
            "# RiskRules with no timestamp/dt_s confirms candidate immediately",
        ]
    )
    w.h2("Recovery의 기억")
    w.body(
        "응급 다음 non-emergency packet에서 smoothed state는 100부터 감소한다. M=0이라고 단순화하면 75, 56.25, "
        "42.19 ... 순으로 떨어지며 이전 DANGER용 65/35 threshold가 적용된다. 실제 값은 남아 있는 6개 raw history에 "
        "영향받는다."
    )
    w.callout(
        "Fail-safe 검토",
        "응급 진입은 빠르지만 해제 조건, operator acknowledgement, actuator latch는 구현돼 있지 않다. GUI status가 "
        "낮아졌다고 실제 alarm을 자동 해제할지는 별도 safety policy다.",
        "red",
    )
    w.finish()

    w.start(
        "41",
        "CO2·결측·FAULT 시나리오",
        "reason, 숫자 risk, 상태를 따로 계산해야 직관과 다른 결과를 설명할 수 있다.",
        "PART IV · 위험 융합과 상태",
        "integrated_node/safenest_risk_engine.py:124-169; tests/test_fault_injection.py:28-57",
    )
    w.table(
        ["입력", "risk/status 가능 결과", "reason·health"],
        [
            ["CO2 2500 단독 + 나머지 정상", "raw 기여 25, 최종 NORMAL 가능", "HIGH_CO2_DANGER, health OK 가능"],
            ["CO2 missing", "fault score 기여 5", "CO2_SENSOR_FAULT, DEGRADED"],
            ["breath NaN", "호흡 fault 기여 15", "RESP_SENSOR_FAULT, DEGRADED/FAULT mapping"],
            ["HR missing", "HR 기여 0", "HR_SENSOR_MISSING, DEGRADED"],
            ["빈 dict", "risk=0, status=FAULT", "ALL_SENSORS_MISSING, system FAULT"],
            ["runner=None but raw valid", "NORMAL 가능", "일부 model NOT_RUN이 health로 전파 안 됨"],
        ],
        [0.36, 0.33, 0.31],
    )
    w.h2("빈 packet 특수 분기")
    w.body(
        "모든 q가 missing 기준을 만족하거나 dict가 비면 일반 weighted fusion을 건너뛴다. risk=0은 점수가 안전하다는 "
        "뜻이 아니라 계산 가능한 관측이 없다는 뜻이므로 status_str과 system_status를 모두 FAULT로 둔다."
    )
    w.callout(
        "소비자 금지 규칙",
        "if risk_score < threshold: safe 같은 단일 조건을 쓰지 않는다. 먼저 system_status==FAULT를 처리하고, "
        "DEGRADED 정책을 적용한 뒤 risk level을 해석한다.",
        "red",
    )
    w.finish()

    w.start(
        "42",
        "현재 확인된 계약·상태 불일치",
        "확장 전에 고쳐야 할 항목을 '현재 사실'과 '권장 결정'으로 분리한다.",
        "PART V · 검증과 확장",
        "risk/risk_rules.py:39-61,285-300; integrated_node/run_demo.py:31-37; integrated_node/safenest_risk_engine.py:338-415",
    )
    w.table(
        ["우선", "현재 사실", "영향 / 권장"],
        [
            ["P0", "config CAUTION 35, engine 진입 40", "정책 이중화; 한 진실원천으로 통일"],
            ["P0", "generic timestamp reason이 status mapping에서 누락", "하위 FAULT가 system OK로 숨을 수 있음"],
            ["P0", "runner=None·load error 일부가 health에 미전파", "model NOT_RUN인데 system OK 가능"],
            ["P0", "CO2 danger 단독 최종 NORMAL 가능", "의도된 policy인지 hazard review"],
            ["P0", "CO2 int8 실사용 범위 포화", "converter·이중 전처리 감사 후 재양자화"],
            ["P0", "문자열 CO2/RPM/phase/time의 type 검사 부재", "비교·np.isnan/isfinite에서 tick 전체 crash 가능; exception containment 필요"],
            ["P1", "CLI가 thermal_pred/co2_info legacy key 사용", "run_demo.py --cli KeyError"],
            ["P1", "sensor_status 계산 후 최종 dict에서 버림", "per-sensor health 관측성 손실"],
            ["P1", "presence 주석은 fusion, 실제는 mmWave 하나", "문서·이름·정책 일치 필요"],
            ["P1", "rFFT 경로는 AI input에 미연결", "telemetry와 inference 경로 명시"],
        ],
        [0.11, 0.47, 0.42],
    )
    w.callout(
        "검토 원칙",
        "이 표는 새 기능 목록이 아니라 현재 시스템을 잘못 이해하게 만드는 경계 목록이다. 복잡도를 늘리기 전에 "
        "reason taxonomy, policy source, health propagation부터 닫아야 한다.",
        "red",
    )
    w.finish()

    w.start(
        "43",
        "48개 자동 테스트가 증명하는 범위",
        "2026-07-27 재실행에서도 48/48 통과했지만, 시험 oracle의 범위를 넘겨 해석하지 않는다.",
        "PART V · 검증과 확장",
        "reports/TEST_RESULTS_20260726.md:13-45; tests/test_*.py",
    )
    w.table(
        ["묶음", "수", "검증", "주요 공백"],
        [
            ["Fault injection", 4, "empty·NaN·CO2 missing·override", "복합 fault·모든 reason 전파"],
            ["CSV Adapter", 2, "session 격리·large gap", "실측 CSV·label consistency"],
            ["mmWave interpreter", 8, "hash·tensor·metadata·invoke", "labeled class accuracy"],
            ["Stream Adapter", 7, "time edge·gap·NaN·presence·stale", "10 Hz jitter·장시간 loss"],
            ["Risk rules", 7, "weights·호흡·PIR·timestamp·fall", "CO2/HR/LPF/final threshold"],
            ["Thermal", 7, "hash·tensor·shape·finite·smoke", "independent fall recall"],
            ["Integration", 13, "synthetic scenarios·invoke fallback", "real sensor·concurrency·long-run"],
        ],
        [0.24, 0.08, 0.35, 0.33],
    )
    w.h2("이번 재실행")
    w.body(
        "macOS system Python 3.9 환경에서 48 tests가 모두 통과했다. LibreSSL/urllib3 경고와 tf.lite.Interpreter "
        "폐기 예정 경고는 있었지만 실패 원인은 아니었다. bundled document runtime에는 TensorFlow가 없어 전체 model "
        "test를 실행할 수 없었으므로, 실행 환경도 evidence의 일부로 기록해야 한다."
    )
    w.callout(
        "올바른 결론",
        "'48개 로컬 계약·예외·합성 시나리오 테스트 통과'는 맞다. '현장 안전성·정확도 검증 완료'는 틀리다.",
        "amber",
    )
    w.finish()

    w.start(
        "44",
        "Benchmark와 Raspberry Pi HIL 계획",
        "평균 invoke 시간 하나로 실시간·안전 성능을 판단하지 않는다.",
        "PART V · 검증과 확장",
        "benchmarks/thermal_latest.json:1-15; benchmarks/benchmark_thermal.py:23-59; models/model_manifest.json:96-103",
    )
    w.table(
        ["현재 측정", "값", "제한"],
        [
            ["Thermal macOS mean", "0.1397 ms", "zero frame, wrapper-level predict 전체"],
            ["Thermal p95/p99", "0.1606 / 0.2070 ms", "macOS arm64, Pi 아님"],
            ["runs", "warmup 50 + measure 1000", "sensor I/O·GUI·fusion 제외"],
            ["CO2/mmWave", "저장 수치 없음", "예시 latency를 실측으로 사용 금지"],
            ["memory/temperature", "없음", "RSS·throttling·전력 미측정"],
        ],
        [0.30, 0.24, 0.46],
    )
    w.h2("Pi 5 승인 측정 항목")
    w.numbered(
        [
            "모델 SHA, runtime/version, thread 수, CPU governor를 고정한다.",
            "warm-up 후 모델별 p50/p95/p99/max와 preprocessing을 분리 측정한다.",
            "sensor timestamp부터 decision·actuator까지 end-to-end latency를 측정한다.",
            "RSS, CPU%, 온도, throttling flag, 전력을 장시간 기록한다.",
            "정상·startup·fault·fallback·동시 GUI 부하를 별도 scenario로 반복한다.",
            "deadline miss율과 recovery time을 평균값과 함께 보고한다.",
        ]
    )
    w.callout(
        "성능 요구의 상태",
        "요구 문서가 충돌한다. roadmap W4는 mmWave Pi p95&lt;20 ms, IMPORT_PROVENANCE/P1-1은 p95&lt;100 ms를 "
        "적었다. 둘 다 결과가 아니며 승인된 요구사항 ID도 없다. 먼저 REQ-LAT-MM-001과 승인 기준을 하나로 고정한다. "
        "30초 startup과 invoke p95는 서로 다른 latency component다.",
        "red",
    )
    w.finish()

    w.start(
        "45",
        "데이터 split, leakage, 평가 지표",
        "안전 모델의 성능은 sample 수보다 독립성 단위와 error cost로 정의한다.",
        "PART V · 검증과 확장",
        "adapters/mmwave_csv_adapter.py:70-124; thermal_train.py:18-23; docs/roadmap_and_setup/safenest_mmwave_latest_development_direction_20260726.md",
    )
    w.table(
        ["누수 단위", "잘못된 split", "권장 split"],
        [
            ["Thermal sequence", "인접 frame random split", "subject+scene+session group holdout"],
            ["mmWave overlap", "90% 겹친 window random split", "원 session을 한 split에만 배치"],
            ["Subject", "같은 사람 train/test", "subject-independent holdout"],
            ["Source/device", "dataset 정체를 class 단서로 학습", "source-balanced + leave-one-source-out"],
            ["Calibration", "test 통계로 mean/std 결정", "train-only fit, artifact versioning"],
        ],
        [0.25, 0.36, 0.39],
    )
    w.h2("최소 metric set")
    w.bullets(
        [
            "class별 precision, recall, F1, confusion matrix를 source별로 보고한다.",
            "APNEA·FALL은 false negative와 false alarm/hour를 별도로 본다.",
            "confidence calibration은 ECE·reliability curve로 확인한다.",
            "INT8 전후 동일 holdout의 성능 차이와 saturation ratio를 보고한다.",
            "운영 시험은 sensor-to-alarm latency와 fault recovery time을 함께 측정한다.",
        ]
    )
    w.callout(
        "Class 1",
        "mmWave RAPID_OR_ABNORMAL의 양성 학습 sample provenance가 복구되기 전에는 확정 안전 근거로 단독 사용하지 "
        "않는다. 필요하면 NORMAL/APNEA 2-class baseline과 비교한다.",
        "amber",
    )
    w.finish()

    w.start(
        "46",
        "복잡도 증가를 흡수하는 확장 구조",
        "센서 수보다 계약·시간·상태·정책의 결합도를 먼저 낮춘다.",
        "PART V · 검증과 확장",
        "docs/roadmap_and_setup/safenest_mmwave_latest_development_direction_20260726.md; README.md:18-42",
    )
    w.pipeline(
        [
            ("Drivers", "raw + clock"),
            ("Adapters", "canonical data"),
            ("Feature", "window·quality"),
            ("Decision", "models·policy"),
            ("Outputs", "alarm·log"),
        ]
    )
    w.table(
        ["경계", "책임", "금지할 결합"],
        [
            ["Driver", "protocol·CRC·sensor timestamp", "risk threshold를 driver에 넣기"],
            ["Adapter", "unit·shape·resample·quality", "model class를 sensor raw로 위장"],
            ["Feature store", "window·session·calibration", "device 간 state 공유"],
            ["Model service", "artifact contract·predict·health", "UI key에 맞춘 임의 출력"],
            ["Policy engine", "score·override·state transition", "hard-code와 config 이중화"],
            ["Output gateway", "idempotent alert·audit log", "risk만 보고 health 무시"],
        ],
        [0.20, 0.42, 0.38],
    )
    w.h2("분산화는 두 번째 단계")
    w.body(
        "먼저 in-process interface를 typed schema와 explicit state로 정리한다. 그 다음 ROS2/MQTT 등 transport를 "
        "선택한다. transport를 먼저 도입하면 암묵적 key와 시간 불일치가 network boundary 뒤에 숨는다."
    )
    w.callout(
        "확장 원칙",
        "새 sensor를 붙일 때 기존 packet dict에 key만 추가하지 않는다. canonical envelope, freshness budget, quality, "
        "state owner, fallback, test oracle를 하나의 변경 단위로 승인한다.",
        "blue",
    )
    w.finish()

    w.start(
        "47",
        "새 센서·모델 추가 체크리스트",
        "인터페이스를 먼저 고정하면 시스템 확장 시 원인 추적 비용이 줄어든다.",
        "PART V · 검증과 확장",
        "models/model_manifest.json; adapters/; inference/; tests/",
    )
    w.table(
        ["단계", "승인 질문"],
        [
            ["1 물리", "측정 물리량, 단위, 범위, bandwidth, noise·failure mode가 정의됐는가?"],
            ["2 시간", "sensor clock, sample rate, jitter, age, gap, reset 조건이 정의됐는가?"],
            ["3 Schema", "version, device/session id, shape, optionality, null semantics가 있는가?"],
            ["4 Adapter", "calibration·resampling·quality와 rejection reason이 재현 가능한가?"],
            ["5 Dataset", "subject/session/source split과 label provenance가 고정됐는가?"],
            ["6 Artifact", "model hash, metadata, tensor, class map, converter가 manifest에 있는가?"],
            ["7 Runtime", "load/predict health와 fallback reason이 공통 taxonomy를 따르는가?"],
            ["8 Policy", "score·override·threshold·conflict rule이 hazard review를 통과했는가?"],
            ["9 Evidence", "unit, integration, fault, HIL, long-run, regression 시험이 있는가?"],
            ["10 Operations", "log, alert idempotency, watchdog, rollback, version telemetry가 있는가?"],
        ],
        [0.22, 0.78],
    )
    w.callout(
        "Definition of Done",
        "모델 파일이 실행되는 상태가 완료가 아니다. semantic contract, policy 연결, fault propagation, target hardware "
        "evidence까지 연결돼야 시스템 기능으로 승인한다.",
        "red",
    )
    w.finish()

    w.start(
        "48",
        "팀원을 위한 코드 읽기 경로",
        "위에서 아래가 아니라 진실원천에서 실행 경로와 증거로 이동한다.",
        "PART V · 검증과 확장",
        "README.md:43-48; walkthrough/README.md:1-35",
    )
    w.numbered(
        [
            "<b>README.md</b>: 현재 기준선·디렉터리 역할·테스트 범위를 파악한다.",
            "<b>models/model_manifest.json</b>: 세 artifact의 shape·dtype·class·status를 표로 옮긴다.",
            "<b>inference/*_interpreter.py</b>: validation -> preprocessing -> invoke -> decode 차이를 비교한다.",
            "<b>adapters/</b>: window의 시간·session·rejection 조건을 state diagram으로 그린다.",
            "<b>risk/risk_config.json + risk_rules.py</b>: 부분 score·override·reason을 수식으로 만든다.",
            "<b>safenest_risk_engine.py</b>: state ownership, 호출 순서, smoothing, output을 추적한다.",
            "<b>virtual streamer + plotter</b>: simulation input과 현재 UI 소비 계약을 확인한다.",
            "<b>mr60/</b>: 독립 receiver와 통합 경로 사이의 결손을 기록한다.",
            "<b>tests/ + reports/ + benchmarks/</b>: 주장마다 실제 oracle과 환경을 연결한다.",
            "<b>NEXT_STEPS·roadmap</b>: 구현 사실과 계획을 다시 분리한다.",
        ]
    )
    w.h2("코드 리뷰 기록 형식")
    w.code(
        [
            "CLAIM: 한 문장 주장",
            "EVIDENCE: file:line + 재현 명령",
            "STATUS: IMPLEMENTED / LOCAL-VERIFIED / PARTIAL / DISCONNECTED",
            "ASSUMPTION: unit, timebase, label, device",
            "RISK: 틀렸을 때 생기는 system effect",
            "NEXT TEST: claim을 반증할 최소 시험",
        ]
    )
    w.finish()

    w.start(
        "49",
        "필수 실습 4개",
        "설명을 읽는 데서 끝내지 않고 현재 artifact와 상태를 직접 재현한다.",
        "PART V · 검증과 확장",
        "tests/; models/model_manifest.json; risk/risk_rules.py; integrated_node/safenest_risk_engine.py",
    )
    w.table(
        ["실습", "수행", "완료 기준"],
        [
            ["A INT8", "세 모델 z 범위·raw 범위·clipping ratio 계산", "CO2 포화 시작점을 수식·코드로 일치시킴"],
            ["B 시간축", "10 Hz 300개, gap 0.5 s, stale 2 s timing diagram", "첫 invoke·reset·debounce 시각을 설명"],
            ["C 위험도", "CO2 2500, RR 8, HR 130, no-motion을 단독·조합", "raw·filtered·status·reason을 분리 계산"],
            ["D Fault", "Inf·문자열·역행 time·runner None·복합 invoke error 주입", "문자열은 현재 crash 재현; 미래 TYPE_MISMATCH oracle과 분리"],
        ],
        [0.16, 0.49, 0.35],
    )
    w.h2("권장 실행 뼈대")
    w.code(
        [
            "engine = SafeNestRiskEngine()",
            "for k in range(...):",
            "    packet = make_packet(timestamp_s=100.0 + 0.1*k)",
            "    result = engine.evaluate_risk(packet)",
            "    log(k, result['risk_score'], result['status_str'],",
            "        result['system_status'], result['reasons'])",
            "assert expected_transition_sequence == observed",
        ]
    )
    w.h2("보고서에 포함할 것")
    w.bullets(
        [
            "입력 packet과 model/policy version, 실행 Python·runtime 환경",
            "기대식, 실제 결과, 허용 오차, 실패 시 최소 재현 코드",
            "왜 그 assertion이 안전 요구를 대표하는지 한 문장",
            "현재 테스트가 놓친 branch 또는 reason propagation 하나 이상",
        ]
    )
    w.callout(
        "실습 금지",
        "현 NPZ에서 높은 일치율이 나와도 현장 accuracy로 이름 붙이지 않는다. 내부 regression과 독립 validation을 "
        "분리한다.",
        "amber",
    )
    w.finish()

    w.start(
        "50",
        "이해 점검 문제",
        "답은 용어 정의와 코드 근거, 한계까지 포함해야 한다.",
        "학습 점검",
        "이 문서 전체",
    )
    w.numbered(
        [
            "breath_rpm 16을 300번 복제한 배열이 왜 resp_phase 30초 window가 아닌가?",
            "10 Hz, N=300의 Nyquist와 delta_f를 계산하고 호흡 대역과 연결하라.",
            "rFFT history N=30일 때 0.1-0.5 Hz FFT band-energy proxy의 한계를 설명하라.",
            "Thermal intensity와 실제 temperature grid의 계약 차이를 네 가지 이상 쓰라.",
            "CO2 2500 ppm에서 HIGH_CO2_DANGER인데 최종 NORMAL이 가능한 계산을 보이라.",
            "Q_quality를 담은 q_gate와 동적 가중치 재정규화가 현재 어떻게 다른지 수식으로 설명하라.",
            "upstream apnea flag, RPM candidate, mmwave.class=2의 earliest alarm time을 비교하라.",
            "status_str=NORMAL, system_status=DEGRADED가 모순이 아닌 이유를 설명하라.",
            "Manifest hash가 맞아도 model semantic이 틀릴 수 있는 예를 두 개 들라.",
            "48개 테스트 통과로 말할 수 있는 것과 말할 수 없는 것을 각각 세 개 쓰라.",
            "다중 device를 같은 RiskEngine instance에 넣으면 오염되는 상태를 네 개 이상 찾으라.",
            "새 sensor node 추가 전 schema에 넣어야 할 시간·식별·품질 field를 설계하라.",
        ]
    )
    w.callout(
        "통과 기준",
        "정답 숫자만 맞는 것은 부족하다. 그 숫자가 어떤 파일의 어느 정책·artifact에서 왔고, 어떤 상황에는 "
        "적용되지 않는지 설명할 수 있어야 한다.",
        "blue",
    )
    w.finish()

    w.start(
        "51",
        "점검 문제 해설의 논리",
        "핵심은 shape·숫자·상태를 의미 계약과 연결하는 것이다.",
        "학습 점검",
        "이 문서 전체",
    )
    w.table(
        ["문항", "핵심 답"],
        [
            ["1", "RPM은 window 요약, phase는 시간 sample. 상수 배열은 DC이고 파형 주기·shape가 없다."],
            ["2", "Nyquist 5 Hz, delta_f 0.0333 Hz≈2 rpm, 0.1-0.5 Hz는 6-30 rpm."],
            ["3", "delta_f=0.333 Hz라 대역 내 사실상 한 bin; N=10에는 bin이 없다."],
            ["4", "°C·방사율·NUC·FOV·orientation·frame rate 대신 grayscale/resize/value/255다."],
            ["5", "env score=1, weight .25 -> 25점; engine CAUTION 40 미만."],
            ["6", "현재 R_risk=100*sum(wS); Q_quality는 telemetry/gate. 품질 가중 재정규화는 미구현."],
            ["7", "flag 즉시, RPM/AI candidate 2초; AI는 최초 window 29.9초 후 가능."],
            ["8", "관측 위험과 장비 건강은 직교 축. 일부 경로 unavailable이어도 낮은 위험 증거 가능."],
            ["9", "다른 phase semantic·filter 또는 잘못된 feature order/시간척도는 같은 shape/hash로도 오류."],
            ["10", "local contract/smoke는 증명; Pi·실센서·현장 정확도·장시간 안전은 미증명."],
            ["11", "CO2/risk/rFFT history, mmWave buffer, timers, IIR, prev_status가 instance 전역."],
            ["12", "schema/device/session/sequence/measured_at/received_at/age/unit/calibration/quality/reason."],
        ],
        [0.10, 0.90],
    )
    w.callout(
        "흔한 오답",
        "'AI가 세 센서를 융합한다', 'CO2 danger는 즉시 DANGER', '테스트 통과=정확도 검증', "
        "'INT8=무조건 4배 빠름'은 현재 코드·증거와 맞지 않는다.",
        "red",
    )
    w.finish()

    w.start("A", "용어집 A-M", "처음 등장한 전문 용어를 시스템 맥락으로 다시 정의한다.", "부록", "이 문서 전체")
    w.table(
        ["용어", "정의"],
        [
            ["Adapter", "생산자 형식을 canonical unit·shape·time grid로 변환하고 거부 이유를 내는 경계"],
            ["Aliasing", "Nyquist를 넘는 성분이 낮은 주파수로 접혀 보이는 현상"],
            ["Calibration", "sensor·signal·confidence가 기준과 맞도록 parameter를 추정·검증하는 과정"],
            ["Canonical", "출처가 달라도 downstream이 동일하게 소비하도록 고정한 표준 표현"],
            ["Clipping", "양자화 표현 범위를 넘은 값이 최소·최대 정수로 포화되는 것"],
            ["Clutter", "벽·좌석 등 지속적 radar reflection과 그 background estimate"],
            ["Confidence", "현재 wrapper가 선택 class에 부여한 최대 출력값; 정확도와 동일하지 않음"],
            ["Contract", "key·unit·shape·dtype·time·error semantics에 대한 생산자-소비자 약속"],
            ["Coverage", "window 첫 sample에서 마지막 sample까지 실제 측정 시간"],
            ["Dtype", "float32, int8처럼 tensor 원소의 저장·연산 형식"],
            ["Fallback", "주 model 경로를 쓸 수 없을 때 사용하는 대체 처리와 그 provenance"],
            ["FMCW", "주파수를 sweep하는 연속파 radar 방식; beat frequency로 거리를 추정"],
            ["Freshness", "현재 decision에서 sensor 값이 사용 가능한 최대 age를 만족하는 성질"],
            ["HIL", "hardware-in-the-loop. 실제 target hardware와 I/O를 포함한 통합 시험"],
            ["Hysteresis", "진입과 해제 threshold를 다르게 해 상태 flicker를 줄이는 방법"],
            ["I/Q", "complex radar signal의 in-phase와 quadrature 직교 성분"],
            ["IIR", "과거 출력 state를 재귀적으로 사용하는 infinite impulse response filter"],
            ["Manifest", "artifact id·path·hash·tensor·class·validation을 기록한 기계 계약"],
        ],
        [0.24, 0.76],
        gap=0,
    )
    w.finish()

    w.start("B", "용어집 N-Z", "상태·신호·검증 용어", "부록", "이 문서 전체")
    w.table(
        ["용어", "정의"],
        [
            ["Node", "입력을 생산·처리·표시하는 실행 단위. model과 동의어가 아님"],
            ["NUC", "thermal pixel 간 응답 편차를 보정하는 non-uniformity correction"],
            ["Nyquist", "sample rate fs로 구분 가능한 이상적 최고 주파수 fs/2"],
            ["Phase unwrap", "-pi/pi로 접힌 phase에 2pi 배수를 더해 연속 궤적을 복원"],
            ["Provenance", "data·code·artifact·fallback·version이 결과를 만든 계보"],
            ["PSD", "단위 주파수당 power 분포. 현재 chest-bin 코드는 정규화 PSD가 아니라 FFT 제곱합 proxy"],
            ["Quality Gate", "입력의 기본 유효성·범위·시간 조건을 검사하는 경계"],
            ["Quantization", "실수 범위를 scale·zero point를 사용해 제한된 정수 격자에 매핑"],
            ["Range bin", "range FFT의 주파수 index를 거리 mapping으로 해석한 구간"],
            ["Reason taxonomy", "fault·hazard 원인 코드의 이름·계층·전파 규칙"],
            ["Resampling", "불규칙 또는 다른 rate의 sample을 목표 시간 grid로 변환"],
            ["Schema", "message field·type·version·optional semantics의 구조적 정의"],
            ["Semantic contract", "tensor 숫자가 실제로 어떤 물리량·단위·전처리를 뜻하는지에 대한 약속"],
            ["Stateful", "현재 출력이 과거 buffer·timer·상태에 의존하는 성질"],
            ["Stale", "마지막 update가 freshness budget을 넘어 최신 판단에 부적합한 상태"],
            ["Tensor", "model이 입력·출력하는 다차원 숫자 배열"],
            ["TFLite", "edge runtime용 TensorFlow Lite model artifact 형식"],
            ["Wrapper", "model validation·전처리·invoke·후처리·health를 캡슐화한 코드"],
            ["Z-score", "x에서 학습 mean을 빼고 std로 나눠 좌표계를 맞추는 정규화"],
        ],
        [0.24, 0.76],
        gap=0,
    )
    w.finish()

    w.start("C", "복제 상수와 동기화 위험", "현재 숫자가 여러 파일에 복제돼 있으므로 변경 지점과 검증을 함께 추적한다.", "부록", "models/model_manifest.json; risk/risk_config.json; integrated_node/safenest_risk_engine.py")
    w.table(
        ["항목", "현재 값", "복제 위치 / 주의"],
        [
            ["mmWave window", "10 Hz, 300, nominal 30 s", "manifest·metadata·Stream/CSV Adapter·RiskEngine constructor에 복제; cadence 미강제"],
            ["mmWave gap/stale", ">0.5 s / >2 s", "Stream Adapter constructor·is_stale"],
            ["respiration normal", "12-20 rpm", "risk_config.json"],
            ["apnea confirm", "2.0 s", "risk_config.json"],
            ["PIR no-motion", "15.0 s", "risk_config.json"],
            ["CO2 warning/danger", "1000 / 2500 ppm", "risk_config.json; provisional"],
            ["CO2 slope reason", "15 ppm/min", "risk_config.json; score 직접 영향 없음"],
            ["risk weights", ".30/.25/.20/.15/.10", "risk_config.json; 합 1 검사"],
            ["raw evaluator level", "35 / 75", "risk_config.json"],
            ["engine entry", "40 / 75", "engine hard-code; config와 불일치"],
            ["DANGER recovery", "65 / 35", "engine hard-code"],
            ["smoothing", "history 6, alpha .25", "engine hard-code"],
            ["Thermal fall", "class2, confidence .8", "risk_rules.py hard-code"],
            ["rFFT", "64 bins, history 30", "engine hard-code; physical config 아님"],
        ],
        [0.28, 0.28, 0.44],
    )
    w.callout(
        "변경 절차",
        "현재 표는 single source 목록이 아니다. 정책·시간 상수는 요구사항 ID와 hazard rationale을 붙인 canonical config로 "
        "수렴시키고, 불가피한 복제는 startup assertion으로 동기화한다. boundary·recovery·GUI·문서를 한 regression으로 묶는다.",
        "red",
    )
    w.finish()

    w.start("D", "코드 근거 지도", "주요 주장별 다시 확인할 파일", "부록", "repository root as of 2026-07-26")
    w.table(
        ["주제", "우선 근거", "보조 증거"],
        [
            ["전체 packet·orchestration", "integrated_node/safenest_risk_engine.py", "virtual_sensor_streamer.py, plotter.py"],
            ["위험 score·timer·health", "risk/risk_rules.py, risk_config.json", "tests/test_risk_rules.py"],
            ["모델 artifact 계약", "models/model_manifest.json", "각 metadata·IMPORT_PROVENANCE"],
            ["Thermal runtime·data", "thermal_interpreter.py, thermal_prep.py", "thermal_train.py, thermal tests"],
            ["CO2 runtime", "co2_interpreter.py", "co2 metadata, co2_data/ guides"],
            ["mmWave runtime", "mmwave_interpreter.py", "stream/csv adapters, mmWave tests"],
            ["real MR60 gap", "mr60/sensor_receiver.py", "sensor_simulator.py"],
            ["자동 검증", "reports/TEST_RESULTS_20260726.md", "tests/test_*.py"],
            ["성능", "benchmarks/thermal_latest.json", "benchmark_thermal.py"],
            ["확장 방향", "docs/roadmap_and_setup/\nsafenest_mmwave_latest_\ndevelopment_direction_20260726.md", "NEXT_STEPS.md"],
        ],
        [0.26, 0.42, 0.32],
    )
    w.h2("재현 명령")
    w.code(
        [
            "python3 -m unittest discover -s tests -p 'test_*.py' -v",
            "python3 integrated_node/run_demo.py        # GUI",
            "python3 integrated_node/run_demo.py --cli  # 현재 legacy-key failure 확인 대상",
        ]
    )
    w.callout(
        "Benchmark 명령 주의",
        "python3 benchmarks/benchmark_thermal.py는 thermal_latest.json과 날짜가 고정된 thermal_mac_20260725.json을 "
        "덮어쓴다. 배포 근거를 보존하려면 임시 worktree/복사본에서 실행하거나 스크립트에 --output 옵션을 먼저 추가한다.",
        "amber",
    )
    w.callout(
        "날짜 기준",
        "문서 작성일은 2026-07-27이지만 기술 사실은 2026-07-26 정리된 프로젝트 루트와 2026-07-27 재실행 "
        "테스트를 기준으로 한다. 코드가 바뀌면 이 근거 지도를 다시 생성한다.",
        "blue",
    )
    w.finish()

    w.start("E", "확장 전 팀 준비도 Gate", "모든 항목을 설명·재현할 수 있을 때 다음 센서와 node를 추가한다.", "부록", "이 문서 전체")
    w.table(
        ["Gate", "통과 질문", "현재 상태"],
        [
            ["G1 Architecture", "AS-IS 연결·단절·state owner를 한 장에 그릴 수 있는가?", "미평가: 팀원 도식·구두설명 필요"],
            ["G2 Contract", "각 field의 unit·shape·time·optional·reason이 schema화됐는가?", "암묵적, 보강 필요"],
            ["G3 Model", "세 artifact의 semantic·quant range·validation level을 설명하는가?", "미평가: 계산 실습 필요"],
            ["G4 Policy", "score·override·LPF·hysteresis가 한 진실원천인가?", "threshold 이중화"],
            ["G5 Health", "모든 load/invoke/timestamp fault가 최종 health에 전파되는가?", "알려진 hole 있음"],
            ["G6 Evidence", "실센서·Pi·independent label·long-run evidence가 있는가?", "미완료"],
            ["G7 Operations", "watchdog·alert·audit·rollback이 설계됐는가?", "미구현"],
        ],
        [0.20, 0.52, 0.28],
    )
    w.h2("한 페이지 기억 지도")
    w.pipeline(
        [
            ("물리", "unit·noise"),
            ("시간", "rate·age"),
            ("계약", "shape·version"),
            ("판단", "score·state"),
            ("증거", "test·HIL"),
        ]
    )
    w.bullets(
        [
            "새 복잡도는 Adapter와 schema 경계에서 흡수하고, policy 안으로 원시 형식을 끌고 오지 않는다.",
            "모델 출력보다 semantic contract와 fault provenance를 먼저 검증한다.",
            "risk, system health, reasons를 항상 함께 전달한다.",
            "구현됨, 로컬 검증됨, 현장 검증됨을 같은 말로 쓰지 않는다.",
        ]
    )
    w.callout(
        "최종 목표",
        "팀원 모두가 '어디에서 어떤 가정이 들어가고, 그 가정이 깨지면 어떤 상태와 reason으로 보이는가'를 "
        "설명할 수 있어야 시스템 확장이 안전해진다.",
        "green",
    )
    w.finish()

    w.start(
        "F",
        "구현 상태기계와 update order",
        "상태는 값 목록이 아니라 owner·key·guard·reset·invariant로 설명해야 재사용과 동시성이 안전해진다.",
        "부록",
        "integrated_node/safenest_risk_engine.py:73-96,195-344; risk/risk_rules.py:82-146,213-256",
    )
    w.table(
        ["owner / state key", "생명주기·reset", "현재 invariant와 fault 경계"],
        [
            ["StreamAdapter\nbuffer,timestamps,last_timestamp,last_push_time,presence", "engine 생성: empty,presence=1; presence!=1은 presence=0+clear; gap은 storage만 clear", "presence=1은 value/time 검증 전에 대입; 최대 300개; stale read는 clear하지 않음"],
            ["RiskRules\napnea_started_at,apnea_timer_s", "candidate false·invalid에서 None/0", "timestamp 경로와 dt_s 누적 경로; RPM<=0.5와 mmwave.class=2가 timer 공유"],
            ["RiskRules\nno_motion_started_at,no_motion_timer_s", "motion=1·presence false·invalid에서 None/0", "timestamp 또는 dt_s 경로; presence true+motion0가 15 s 지속돼야 score=1"],
            ["Engine\nco2_history", "instance 생성 때만 reset; maxlen=30", "device/session key 없음; source 교체 시 history 오염 가능"],
            ["Engine\nrisk_history,curr_R,prev_status", "instance 생성 때 0/NORMAL; explicit session reset 없음", "non-emergency 순서 raw append→mean→IIR→hysteresis; emergency는 curr_R=100, raw history는 유지"],
            ["Engine\nrFFT history,clutter", "instance 생성 때 reset; 15 history 뒤 calibration", "absence로 calibration을 gate하지 않으며 raw/filtered history가 섞일 수 있음"],
        ],
        [0.27, 0.33, 0.40],
        gap=1.4 * mm,
    )
    w.h2("mmWave Stream Adapter 전이")
    w.table(
        ["설명용 상태", "event / guard", "action", "다음"],
        [
            ["ANY", "presence!=1", "buffer·timestamp clear", "EMPTY"],
            ["FIRST/EMPTY", "finite value,time; last_timestamp=None", "append", "WARMING"],
            ["WARMING/READY", "finite value,time; 0<dt<=0.5", "append", "WARMING 또는 READY"],
            ["WARMING/READY", "dt>0.5", "clear 후 현재 sample reject", "GAP_REJECT; storage EMPTY"],
            ["READY", "read age>2 s", "window 반환 거부", "STALE predicate; buffer retained"],
            ["ANY", "presence=1; numeric NaN/Inf 또는 dt<=0", "presence=1; sample reject; storage 유지", "storage 유지; presence는 1"],
            ["ANY", "presence=1; non-numeric value/time", "presence=1 뒤 TypeError escape 가능", "storage 유지; tick abort 가능"],
        ],
        [0.17, 0.35, 0.28, 0.20],
        gap=1.2 * mm,
    )
    w.callout(
        "동시성 결론",
        "GAP_REJECT와 STALE은 저장된 enum state가 아니라 PushResult와 age로부터 얻는 설명용 결과다. 또한 RiskEngine "
        "한 instance에는 device/session별 key와 lock이 없다. 여러 생산자가 같은 instance를 호출하면 deque, "
        "timer, IIR, prev_status가 섞인다. 확장 시 instance-per-stream 또는 keyed StateStore와 명시적 reset API가 필요하다.",
        "red",
    )
    w.finish()

    w.start(
        "G",
        "제안 canonical schema와 호환성 규칙",
        "현재 dict를 그대로 굳히지 말고 optional·null·version·error 의미를 기계 검증 가능한 계약으로 만든다.",
        "부록 · DESIGN TARGET / NOT IMPLEMENTED",
        "integrated_node/virtual_sensor_streamer.py:177-199; integrated_node/safenest_risk_engine.py:120-169,360-415",
    )
    w.code(
        [
            "SensorValue<T> {",
            "  measured_at_s: float, received_at_s: float, sequence: uint64,",
            "  age_budget_ms: uint32, value: T, unit: str|map<field,str>, status: str,",
            "  quality: 0..1, calibration_id: str|null, reason: [str]",
            "}",
            "SensorEnvelopeV1 {",
            "  schema_version: '1.0.0', message_id: UUID,",
            "  device_id: str, session_id: str,",
            "  sensors: {",
            "    co2: SensorValue{value: 1000.0, unit: 'ppm', ...},",
            "    mmwave: SensorValue{value: {breath_rpm: 16.0,",
            "      resp_phase: 0.12, presence: 1}, unit: {breath_rpm:'rpm',",
            "      resp_phase:'rad', presence:'boolean'},",
            "      phase_semantic: 'resp_phase_unwrapped_clutter_removed',",
            "      preprocess_version: '0.1.0', ...},",
            "    thermal: SensorValue{value: {shape: [62,80]},",
            "      unit: 'normalized_intensity', ...},",
            "    pir: SensorValue{value: {motion: 0}, unit: 'boolean', ...}",
            "  }",
            "}",
        ],
        gap=1.4 * mm,
    )
    w.table(
        ["조건", "validator 동작", "reason"],
        [
            ["필수 field 누락", "전체 message reject", "SCHEMA_REQUIRED_\nFIELD_MISSING"],
            ["nullable field=null", "schema가 허용하고 status/reason이 있을 때만 accept", "VALUE_UNAVAILABLE"],
            ["sensor time 역전·age 초과", "해당 sensor reject; decision health에 전파", "TIME_ORDER/STALE"],
            ["unit·shape·type·semantic 불일치", "해당 sensor reject; decision health에 전파", "UNIT/SHAPE/TYPE/SEMANTIC_MISMATCH"],
            ["같은 message_id 재전송", "중복 side effect 없이 같은 decision_id 반환", "IDEMPOTENT_REPLAY"],
            ["minor의 미지 field", "보존 또는 ignore; required 의미 변경 금지", "호환"],
            ["지원하지 않는 major", "fail closed", "SCHEMA_MAJOR_\nUNSUPPORTED"],
        ],
        [0.23, 0.45, 0.32],
        gap=1.3 * mm,
    )
    w.h2("최소 producer-consumer contract test")
    w.code(
        [
            "msg = valid_fixture(schema_version='1.0.0')",
            "assert consume(serialize(msg)).message_id == msg.message_id",
            "assert reject(delete(msg, 'device_id')) == 'SCHEMA_REQUIRED_FIELD_MISSING'",
            "assert reject(set_(msg, 'sensors.co2.unit', 'C')) == 'UNIT_MISMATCH'",
            "assert reject(expire(msg, 'sensors.mmwave')) == 'STALE'",
            "assert reject(change_phase_semantic(msg)) == 'SEMANTIC_MISMATCH'",
            "assert accept(add_unknown_minor_field(msg))",
            "assert consume(msg).decision_id == consume(msg).decision_id",
        ]
    )
    w.finish()

    w.start(
        "H",
        "실행 가능한 수치 실습 카드",
        "명령·fixture·기대값·채점 기준을 고정해 읽기 과제를 재현 가능한 학습 증거로 바꾼다.",
        "부록",
        "scripts/verify_safenest_learning_examples.py; tests/; models/model_manifest.json",
    )
    w.h2("환경과 명령")
    w.code(
        [
            "cd <repository-root>",
            "python3 scripts/verify_safenest_learning_examples.py",
            "python3 -m unittest discover -s tests -p 'test_*.py' -v",
            "# PASS asserts A-C arithmetic, D type, E gap reset, F time-boundary behavior",
        ],
        gap=1.3 * mm,
    )
    w.table(
        ["실험 / fixture", "기대 oracle", "현재 결론"],
        [
            ["A CO2 INT8\nmanifest+metadata 그대로", "z=-1.0783..0.4080; raw CO2≈267.49..734.75 ppm", "범위 밖은 clip; 현장 성능 증거는 아님"],
            ["B filter\nraw=[80]*10, R0=0", "tick2 R=46.25 CAUTION; tick9 R=75.4949 DANGER", "moving mean+IIR 지연 재현"],
            ["C recovery\ntick10 emergency, 이후 raw0*6", "tick14 R=60.3385 CAUTION; tick16 R=36.4404 NORMAL", "emergency가 raw history를 clear하지 않음"],
            ["D type fault\nco2_ppm='1000' 또는 RPM='16'", "현재 TypeError로 tick 중단 가능", "스크립트가 crash 경계를 assert; 미래는 TYPE_MISMATCH+DEGRADED"],
            ["E stream time\n0.0,0.1,0.7 s", "세 번째 push reject; MMWAVE_STREAM_\nGAP_TOO_LARGE; buffer clear", "스크립트가 reason과 buffer=0을 assert"],
            ["F time guard\nRule time=None; Adapter time='bad'", "candidate 즉시 emergency; Adapter TypeError", "스크립트가 두 현재 경계를 assert; 구조화 fault는 미구현"],
        ],
        [0.27, 0.39, 0.34],
        gap=1.4 * mm,
    )
    w.h2("제출물과 100점 rubric")
    w.table(
        ["배점", "완료 증거"],
        [
            [25, "명령·Python/runtime·artifact digest와 원본 stdout"],
            [25, "각 기대값을 수식으로 독립 계산하고 허용오차 명시"],
            [25, "현재 crash와 바람직한 structured fault를 구분한 최소 재현"],
            [25, "요구사항 ID→test oracle→관측 결과→남은 gap을 한 줄로 연결"],
        ],
        [0.14, 0.86],
        gap=1.1 * mm,
    )
    w.callout(
        "통과 기준",
        "80점 이상이면서 D/F type·time fault 항목을 반드시 통과한다. 스크립트 PASS는 산술·계약 회귀일 뿐 실센서 accuracy나 "
        "안전 인증을 의미하지 않는다.",
        "green",
    )
    w.finish()

    w.start(
        "I",
        "raw→moving mean→IIR→상태 누적 계산",
        "한 시퀀스를 끝까지 계산하면 필터 지연, 히스테리시스, emergency history 효과가 동시에 보인다.",
        "부록",
        "integrated_node/safenest_risk_engine.py:326-344; scripts/verify_safenest_learning_examples.py",
    )
    w.equation(
        [
            "fixture: history=[], R_risk[-1]=0, prev=NORMAL; raw[0..9]=80",
            "M[k]=mean(last up to 6 raw); R_risk[k]=0.75*R_risk[k-1]+0.25*M[k]",
            "tick10 emergency -> R_risk=100 and prev=DANGER; six raw 80 values remain",
            "tick11..16 raw=0; apply DANGER recovery, then ordinary entry thresholds",
        ],
        gap=1.5 * mm,
    )
    w.table(
        ["tick", "raw", "M[k]", "R_risk[k]", "status after tick"],
        [
            [0, 80, "80.0000", "20.0000", "NORMAL"],
            [1, 80, "80.0000", "35.0000", "NORMAL"],
            [2, 80, "80.0000", "46.2500", "CAUTION (>=40)"],
            [5, 80, "80.0000", "65.7617", "CAUTION"],
            [8, 80, "80.0000", "73.9932", "CAUTION"],
            [9, 80, "80.0000", "75.4949", "DANGER (>=75)"],
            [10, "EMG", "bypass", "100.0000", "DANGER; history retained"],
            [11, 0, "66.6667", "91.6667", "DANGER (>65)"],
            [12, 0, "53.3333", "82.0833", "DANGER"],
            [13, 0, "40.0000", "71.5625", "DANGER"],
            [14, 0, "26.6667", "60.3385", "CAUTION (DANGER recovery)"],
            [15, 0, "13.3333", "48.5872", "CAUTION"],
            [16, 0, "0.0000", "36.4404", "NORMAL (<40)"],
        ],
        [0.10, 0.14, 0.20, 0.22, 0.34],
        gap=1.4 * mm,
    )
    w.callout(
        "해석",
        "응급 직후 M=0으로 놓아 100→75→56.25로 계산하면 실제 구현과 다르다. risk_history의 80이 여섯 tick에 걸쳐 "
        "밀려나므로 해제가 더 늦다. 또한 tick14에서 DANGER를 벗어난 뒤 tick15부터 일반 40/75 문턱을 다시 쓴다.",
        "red",
    )
    w.finish()

    w.start(
        "J",
        "요구사항·hazard·oracle·coverage 추적성",
        "테스트 수가 아니라 어떤 실패를 어떤 판정 규칙으로 덮었는지와 남은 gap을 연결한다.",
        "부록",
        "tests/; risk/risk_config.json; models/model_manifest.json; docs/roadmap_and_setup/",
    )
    w.table(
        ["ID / 요구", "실패 hazard", "test oracle", "현재 coverage / gap"],
        [
            ["REQ-TIME-01\n10 Hz·300", "cadence drift가 model semantic을 변경", "dt≈0.1, coverage≈29.9, gap reset", "gap/stale만 일부 검증; strict cadence·coverage 미검사"],
            ["REQ-EMG-01\napnea 2.0 s", "조기 false alarm 또는 늦은 경보", "t0+1.9 non-emg; t0+2.0 emg", "rule timestamp test 있음; end-to-end sensor→alarm 없음"],
            ["REQ-TYPE-01\n입력 type", "한 malformed field가 tick 전체 중단", "문자열/Inf 주입→structured reason, no crash", "미충족; 현재 TypeError/Inf propagation hole"],
            ["REQ-POL-01\nthreshold source", "config와 runtime 판단 불일치", "35/40/65/75 exact-boundary regression", "rule 일부; engine hard-code와 config 충돌"],
            ["REQ-MOD-01\nartifact", "다른 모델·tensor를 조용히 실행", "hash/size/shape/dtype/quant mismatch reject", "Thermal/mmWave 강함; CO2 init validation 약함"],
            ["REQ-DATA-01\n독립 split", "인접·동일 subject 누수로 성능 과장", "subject+session+source group holdout", "CSV session 격리 일부; 독립 label evidence 없음"],
            ["REQ-LAT-01\nPi p95", "deadline miss로 경보 지연", "승인 target에서 p95·miss rate·thermal soak", "20 ms/100 ms 문서 충돌; Pi 결과 없음"],
            ["REQ-OBS-01\n추적 가능 출력", "재시도·혼합 stream 원인 재현 불가", "decision/device/session/id/age/version 필수", "현재 dict에 핵심 식별·age·transition 없음"],
        ],
        [0.18, 0.25, 0.27, 0.30],
        gap=1.0 * mm,
    )
    w.h2("미종결 항목 closure register")
    w.table(
        ["ID / owner", "승인", "next test", "차단할 claim"],
        [
            ["TIME / Platform 담당 미배정", "미승인", "10 Hz jitter·29.9 s coverage HIL", "시간 semantic 보장"],
            ["EMG / Safety 담당 미배정", "규칙만 부분", "sensor→actuator 지연 HIL", "end-to-end 2 s"],
            ["TYPE / Runtime 담당 미배정", "P0 미승인", "문자열·Inf fuzz + no-crash oracle", "malformed 입력 내성"],
            ["POL / Safety 담당 미배정", "P0 미승인", "config와 35/40/65/75 boundary 통합", "threshold 통제"],
            ["MOD / ML 담당 미배정", "부분", "CO2 strict-init mismatch", "세 wrapper 동일 계약"],
            ["DATA / Data·ML 담당 미배정", "미승인", "group holdout + 독립 label", "현장 accuracy"],
            ["LAT / Embedded 담당 미배정", "미승인", "20/100 ms 해소 + Pi soak", "Pi real-time"],
            ["OBS / Platform 담당 미배정", "설계만", "schema contract + replay", "분산 추적 가능"],
        ],
        [0.24, 0.14, 0.35, 0.27],
        gap=1.0 * mm,
    )
    w.h2("Gate 판정 규칙")
    w.numbered(
        [
            "요구사항은 ID·owner·승인 상태·정량 threshold를 가져야 한다.",
            "oracle은 pass/fail을 계산할 입력, 시간, 허용오차, 예상 reason까지 고정한다.",
            "coverage는 실행된 branch·환경·장치·데이터 독립성으로 기록하고 테스트 개수로 대신하지 않는다.",
            "gap에는 책임자·다음 시험·차단할 claim을 붙인다. 문서에 적었다는 사실만으로 CLOSED 처리하지 않는다.",
        ],
        gap=1.2 * mm,
    )
    w.finish()

    w.start("K", "운영·ML 고급 용어집", "첫 등장만으로 뜻을 추론하기 어려운 검증·운영 용어를 정의한다.", "부록", "이 문서 전체")
    w.table(
        ["용어", "정의"],
        [
            ["CPU governor", "CPU 주파수를 성능·전력 정책에 따라 조절하는 OS 제어 방식; benchmark 조건으로 고정"],
            ["Deadline miss", "정해진 시간 budget 안에 처리가 끝나지 않은 실행의 비율 또는 사건"],
            ["Debounce", "짧은 입력 변동을 일정 시간·횟수 확인해 단일 오검출을 억제하는 처리"],
            ["ECE", "confidence 구간별 정확도 차이를 가중 평균한 expected calibration error"],
            ["Fan-out / fan-in", "한 입력을 여러 경로로 분기 / 여러 결과를 한 판단으로 재결합하는 구조"],
            ["Holdout", "학습·parameter 선택에 사용하지 않고 마지막 성능 평가에만 남긴 독립 데이터"],
            ["Idempotency", "같은 message를 중복 처리해도 외부 효과가 한 번 처리한 것과 같아지는 성질"],
            ["Jitter buffer", "도착 간격 변동을 흡수해 일정 시간 grid로 내보내는 임시 buffer"],
            ["Lineage", "data·feature·artifact·결과가 어떤 변환과 version을 거쳤는지의 계보"],
            ["Open-set", "학습 class 밖의 입력이 운영 중 들어올 수 있음을 다루는 인식 조건"],
            ["Oracle", "주어진 입력에서 시험의 기대 출력·상태·reason을 결정하는 판정 규칙"],
            ["Reliability curve", "confidence 구간별 예측 confidence와 실제 정확도를 비교하는 calibration 그래프"],
            ["Representative dataset", "INT8 변환 시 activation 범위를 추정하도록 실제 배포 분포를 대표해야 하는 표본"],
            ["Replay", "기록된 timestamp/data를 다시 흘려 deterministic state·fault를 재현하는 시험"],
            ["RSS", "process가 실제 RAM에 점유한 resident set size; 파일의 model size와 다름"],
            ["Telemetry", "운영 상태·지연·품질·version을 관측용으로 지속 기록·전송하는 값"],
            ["Watermark", "이 시각 이전 event가 충분히 도착했다고 판단하는 stream 처리 기준"],
            ["Watchdog", "정해진 heartbeat·deadline이 끊기면 fault 처리나 restart를 일으키는 감시 장치"],
        ],
        [0.27, 0.73],
        gap=0,
    )
    w.finish()

    w.save()


if __name__ == "__main__":
    build_guide()
    print(OUTPUT)
