"""설계도(흐름도) 그리기용 작은 도구 — matplotlib 만 사용.

좌표는 그림 인치 단위(왼쪽 위 원점, 아래로 y 증가). 상자 높이는 글줄 수로 자동 결정.
  fig, ax = canvas(W, H, title, subtitle)
  node(ax, "key", x, y, w, "제목", ["본문 줄", ...], mod="모듈.py", kind="proc", color=C["ec46"])
  edge(ax, "a", "b", label="...", route="h")
  save(fig, "이름")  -> PNG(200 dpi) + SVG
"""
import re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
try:
    import koreanize_matplotlib  # noqa: F401  (나눔고딕)
except ImportError:
    pass
# 나눔고딕에 없는 기호(−, Σ, Φ, √ 등)는 DejaVu Sans로 대체
plt.rcParams["font.family"] = [plt.rcParams["font.family"][0] if plt.rcParams["font.family"] else "sans-serif", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent
INK, INK2, MUTED, LINE = "#1d2327", "#4b5560", "#7d8790", "#c9d0d5"
C = {
    "common": "#2f8a57",   # 예측대상·공통 자료
    "ec46": "#2a6fc4",     # EC46 주별 1개월 체계
    "seas5": "#c8661d",    # SEAS5 월별 3개월 체계
    "pub": "#6b55b0",      # 자동화·게시
    "dec": "#a77a00",      # 판단
    "ext": "#7d8790",      # 확장 후보·참고
}
TINT = {k: v + "14" for k, v in C.items()}           # 아주 옅은 바탕
FS_T, FS_B, FS_M = 12.0, 10.4, 9.0                      # 제목·본문·모듈 글자 크기(pt)
LH = FS_B / 72 * 1.42                                  # 본문 줄 간격(인치)
NODES = {}
WARN = []


def _tw(s, fs):
    """대략적인 글줄 폭(인치): 한글 1.0em, 그 밖 0.56em."""
    em = fs / 72
    return sum((1.0 if ord(ch) > 0x2E80 else 0.56) for ch in s) * em


def canvas(W, H, title, subtitle=None):
    NODES.clear()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.35, 0.42, title, fontsize=16, fontweight="bold", color=INK, va="center")
    if subtitle:
        ax.text(0.35, 0.78, subtitle, fontsize=9.5, color=INK2, va="center")
    return fig, ax


def node(ax, key, x, y, w, title, lines=(), mod=None, kind="proc", color=None, h=None, title_fs=FS_T):
    """kind: proc(처리, 둥근 상자) · data(자료, 왼쪽 띠) · dec(판단, 마름모) · note(설명, 점선) · ext(확장 후보, 점선)."""
    color = color or (C["dec"] if kind == "dec" else C["ext"] if kind in ("note", "ext") else INK2)
    lines = list(lines)
    if kind == "dec":
        h = h or 0.95
        cx, cy = x + w / 2, y + h / 2
        ax.add_patch(Polygon([(cx, y), (x + w, cy), (cx, y + h), (x, cy)], closed=True,
                             fc="#fff8e6", ec=color, lw=1.3, zorder=2))
        ax.text(cx, cy - (0.07 if lines else 0), title, ha="center", va="center", fontsize=FS_B + 0.3,
                fontweight="bold", color=INK, zorder=3)
        for i, s in enumerate(lines):
            ax.text(cx, cy + 0.12 + i * LH, s, ha="center", va="center", fontsize=FS_M, color=INK2, zorder=3)
        NODES[key] = (x, y, w, h)
        return NODES[key]
    head = FS_T / 72 * 2.3 if title else 0.08
    hh = head + len(lines) * LH + (FS_M / 72 * 2.7 if mod else 0) + 0.10
    h = max(h or 0, hh)
    if kind == "proc":
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.09", fc="white", ec=color, lw=1.5, zorder=2)
        ax.add_patch(p)
        ax.add_patch(FancyBboxPatch((x, y), w, head, boxstyle="round,pad=0,rounding_size=0.09", fc=color + "1f",
                                    ec="none", zorder=2.1))
    elif kind == "data":
        ax.add_patch(Rectangle((x, y), w, h, fc="#f6f7f8", ec=color, lw=1.0, zorder=2))
        ax.add_patch(Rectangle((x, y), 0.07, h, fc=color, ec="none", zorder=2.2))
        # 접힌 모서리(문서) 표시
        f = 0.13
        ax.add_patch(Polygon([(x + w - f, y), (x + w, y + f), (x + w - f, y + f)], closed=True, fc=color + "55",
                             ec=color, lw=0.8, zorder=2.3))
    else:  # note, ext
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=color + "0d",
                                    ec=color, lw=1.1, ls=(0, (4, 2.5)), zorder=2))
    tx = x + (0.16 if kind == "data" else 0.11)
    if title:
        ax.text(tx, y + head / 2 + 0.01, title, fontsize=title_fs, fontweight="bold", color=color if kind != "data" else INK,
                va="center", zorder=3)
        _chk(key, title, title_fs, w - (tx - x) - 0.08)
    for i, s in enumerate(lines):
        ax.text(tx, y + head + (i + 0.5) * LH, s, fontsize=FS_B, color=INK, va="center", zorder=3)
        _chk(key, s, FS_B, w - (tx - x) - 0.06)
    if mod:
        ax.text(tx, y + h - FS_M / 72 * 1.5 - 0.02, mod, fontsize=FS_M, color=color if kind != "data" else INK2, va="center",
                family=["DejaVu Sans Mono", "NanumGothic"], zorder=3)
        _chk(key, mod, FS_M * 1.08, w - (tx - x) - 0.06)
    NODES[key] = (x, y, w, h)
    return NODES[key]


def _chk(key, s, fs, avail):
    if _tw(re.sub(r"\$.*?\$", "xx", s), fs) > avail + 0.02:
        WARN.append(f"[넘침] {key}: {s}")


def anchor(key, side, t=0.5):
    x, y, w, h = NODES[key]
    return {"r": (x + w, y + t * h), "l": (x, y + t * h), "t": (x + t * w, y), "b": (x + t * w, y + h)}[side]


def edge(ax, a, b, label=None, sa=None, sb=None, route=None, mid=None, color=MUTED, ls="-", ta=0.5, tb=0.5,
         lw=1.3, lpos=0.5, loff=(0, 0)):
    """직교 경로 화살표. route: 'h'(가로-세로-가로), 'v'(세로-가로-세로), 's'(직선)."""
    ax_, ay_, aw, ah = NODES[a]; bx_, by_, bw, bh = NODES[b]
    if sa is None or sb is None:
        if bx_ >= ax_ + aw - 0.05:
            sa, sb = sa or "r", sb or "l"
        elif bx_ + bw <= ax_ + 0.05:
            sa, sb = sa or "l", sb or "r"
        elif by_ >= ay_ + ah:
            sa, sb = sa or "b", sb or "t"
        else:
            sa, sb = sa or "t", sb or "b"
    p0, p1 = anchor(a, sa, ta), anchor(b, sb, tb)
    if route is None:
        route = "h" if sa in "lr" and sb in "lr" else "v" if sa in "tb" and sb in "tb" else "l"
    if route == "s" or (route == "h" and abs(p0[1] - p1[1]) < 1e-3) or (route == "v" and abs(p0[0] - p1[0]) < 1e-3):
        pts = [p0, p1]
    elif route == "h":
        m = mid if mid is not None else (p0[0] + p1[0]) / 2
        pts = [p0, (m, p0[1]), (m, p1[1]), p1]
    elif route == "v":
        m = mid if mid is not None else (p0[1] + p1[1]) / 2
        pts = [p0, (p0[0], m), (p1[0], m), p1]
    else:  # 'l' : 한 번 꺾기 (가로→세로 또는 세로→가로)
        pts = [p0, (p1[0], p0[1]), p1] if sa in "lr" else [p0, (p0[0], p1[1]), p1]
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys, color=color, lw=lw, ls=ls, zorder=1.5, solid_capstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1.6,
                arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.28", color=color, lw=lw,
                                shrinkA=0, shrinkB=0.5, ls="-"))
    if label:
        k = max(range(len(pts) - 1), key=lambda i: abs(pts[i + 1][0] - pts[i][0]) + abs(pts[i + 1][1] - pts[i][1]))
        (x0, y0), (x1, y1) = pts[k], pts[k + 1]
        ax.text(x0 + (x1 - x0) * lpos + loff[0], y0 + (y1 - y0) * lpos + loff[1], label, fontsize=FS_M, color=INK2,
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))


def band(ax, x, y, w, h, label, color, sub=None):
    """체계(레인) 바탕 띠."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=color + "0b",
                                ec=color + "55", lw=1.0, zorder=0.5))
    ax.text(x + 0.16, y + 0.2, label, fontsize=11, fontweight="bold", color=color, va="center", zorder=1)
    if sub:
        ax.text(x + 0.16 + _tw(label, 11) + 0.15, y + 0.2, sub, fontsize=8.3, color=INK2, va="center", zorder=1)


def step_header(ax, x, y, w, n, text, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, 0.42, boxstyle="round,pad=0,rounding_size=0.21", fc=color, ec="none", zorder=1))
    ax.text(x + w / 2, y + 0.215, f"{n}  {text}", fontsize=10, fontweight="bold", color="white", ha="center",
            va="center", zorder=2)


def legend(ax, x, y, items=("proc", "data", "dec", "note")):
    lab = {"proc": "처리 단계", "data": "자료(파일)", "dec": "판단", "note": "규칙·설명", "ext": "확장 후보"}
    for k in items:
        if k == "proc":
            ax.add_patch(FancyBboxPatch((x, y), 0.36, 0.2, boxstyle="round,pad=0,rounding_size=0.05", fc="white", ec=INK2, lw=1.2))
        elif k == "data":
            ax.add_patch(Rectangle((x, y), 0.36, 0.2, fc="#f6f7f8", ec=INK2, lw=0.9))
            ax.add_patch(Rectangle((x, y), 0.05, 0.2, fc=INK2, ec="none"))
        elif k == "dec":
            ax.add_patch(Polygon([(x + 0.18, y - 0.02), (x + 0.38, y + 0.1), (x + 0.18, y + 0.22), (x - 0.02, y + 0.1)],
                                 fc="#fff8e6", ec=C["dec"], lw=1.1))
        else:
            ax.add_patch(FancyBboxPatch((x, y), 0.36, 0.2, boxstyle="round,pad=0,rounding_size=0.04", fc="#7d87900d",
                                        ec=C["ext"], lw=1, ls=(0, (3, 2))))
        ax.text(x + 0.46, y + 0.1, lab[k], fontsize=8, color=INK2, va="center")
        x += 0.62 + _tw(lab[k], 8)
    return x


def footer(ax, W, H, text):
    ax.text(W - 0.3, H - 0.22, text, fontsize=7.5, color=MUTED, ha="right", va="center")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    if WARN:
        print("\n".join(WARN)); WARN.clear()
    print("저장", name)
