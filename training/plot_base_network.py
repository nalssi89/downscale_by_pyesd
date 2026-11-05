"""기준망 지점 지도: ASOS 66지점과 추가 17지점(공항·레이더·기타 ASOS). python training/plot_base_network.py OUT.png"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mapbase  # noqa: E402

plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False, "font.size": 10})
st = pd.read_csv(HERE / "base_stations.csv")
add = st[st.asos66 == 0]
kind = add.name.str.contains(r"\(공\)").map({True: "공항"}).fillna(add.name.str.contains(r"\(레\)").map({True: "레이더(산정)"})).fillna("기타 ASOS·도서")
fig, ax = plt.subplots(figsize=(6.4, 7.4), dpi=150)
mapbase.draw(ax, box=(124.5, 33, 131.2, 38.7), color="#777", lw=0.4)
a = st[st.asos66 == 1]
ax.scatter(a.lon, a.lat, s=26, marker="^", c="#eb6834", edgecolors="white", linewidths=0.5, label=f"ASOS 66 ({len(a)})", zorder=3)
for k, c, m in (("기타 ASOS·도서", "#2a78d6", "o"), ("공항", "#1baf7a", "s"), ("레이더(산정)", "#6250d6", "D")):
    s = add[kind == k]
    ax.scatter(s.lon, s.lat, s=34, marker=m, c=c, edgecolors="white", linewidths=0.6, label=f"추가: {k} ({len(s)})", zorder=4)
    for _, r in s.iterrows():
        ax.annotate(r["name"], (r.lon, r.lat), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color="#333")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"기준망 {len(st)}지점 (1991~2026 월별 유효 20년 이상)", loc="left", fontsize=11)
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
plt.tight_layout(); plt.savefig(sys.argv[1] if len(sys.argv) > 1 else "base_network.png")
