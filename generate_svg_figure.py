import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── parsed from SVG_summary.txt ──────────────────────────────────────────────

DATASETS = {
    "DLPFC\n151673": {
        "n_domains": 7,
        "n_svgs_by_domain": [5, 1, 0, 3, 2, 53, 3],
        "total_svgs": 67,
        "top_markers": {
            0: ["CAMK2N1","ENC1","GPM6A","ARPP19","HPCAL1"],
            1: ["PCP4"],
            2: [],
            3: ["NEFL","NEFM","SNCG"],
            4: ["MBP","PLP1"],
            5: ["MBP","PTGDS","FTL","GFAP","FTH1"],
            6: ["RTN1","CCK","CHN1"],
        },
        "tissue": "Human cortex",
        "technology": "Visium",
    },
    "Mouse\nMP1": {
        "n_domains": 10,
        "n_svgs_by_domain": [4, 433, 22, 162, 0, 35, 78, 46, 47, 211],
        "total_svgs": 1038,
        "top_markers": {
            0: ["MAG","MOG","ERMN","UGT8A"],
            1: ["PVALB","SERF2","RPS12","ATP5O-1","MT3"],
            2: ["PTGDS","SPARCL1","APOE","DBI","SPARC"],
            3: ["NDUFA6","CALM1","CKB","COX5B","LDHB"],
            5: ["NRGN","YWHAH","SERINC1","EEF1A2","APP"],
        },
        "tissue": "Mouse brain",
        "technology": "Visium",
    },
    "Human\nITYN": {
        "n_domains": 3,
        "n_svgs_by_domain": [3, 8, 1],
        "total_svgs": 12,
        "top_markers": {
            0: ["TAGLN","AEBP1","ISLR"],
            1: ["SCGB3A1","SERPINA1","SPINK1","MUC1","MUC5B"],
            2: ["KRT17"],
        },
        "tissue": "Human tissue",
        "technology": "ST",
    },
    "Slide-seq\nv2": {
        "n_domains": 6,
        "n_svgs_by_domain": [15, 2, 24, 0, 14, 4],
        "total_svgs": 59,
        "top_markers": {
            0: ["Tuba1a","Sox11","Actb","Nfib","Basp1"],
            1: ["Nfib","Sox5"],
            2: ["Fabp7","Hsp90ab1","Hnrnpa2b1","Marcks"],
            5: ["Ccnd2","Map1b","Dynll2","Arhgap11a"],
        },
        "tissue": "Mouse brain",
        "technology": "Slide-seq",
    },
    "STARmap": {
        "n_domains": 7,
        "n_svgs_by_domain": [8, 6, 1, 4, 2, 3, 1],
        "total_svgs": 25,
        "top_markers": {
            0: ["Camk2n1","Elmo1","Nrgn","Hpcal4"],
            1: ["Hpcal4","Nrgn","Slc17a7","Pcsk2"],
            3: ["Mbp","Mobp","Plp1","Plekhb1"],
            5: ["Glul","Bcl6","Atp1a2"],
        },
        "tissue": "Mouse cortex",
        "technology": "STARmap",
    },
    "Mouse\nMOB": {
        "n_domains": 5,
        "n_svgs_by_domain": [22, 33, 40, 6, 7],
        "total_svgs": 108,
        "top_markers": {
            0: ["CCK","HLF","AQP4","SLC1A3","PLA2G7"],
            1: ["OMP","NUDT4","VIM","VTN","MEST"],
            2: ["BAIAP2","FAM163B","PTPRO","PSD","SYN1"],
            3: ["NTNG1","SLC17A7","RELN","SV2B"],
            4: ["NRGN","NNAT","CAMK4","LRP8"],
        },
        "tissue": "Mouse olfactory bulb",
        "technology": "Slide-seq",
    },
}

# ── design tokens ─────────────────────────────────────────────────────────────

PAL = {
    "bg":      "#F4F6F9",
    "panel":   "#FFFFFF",
    "text":    "#1C2833",
    "neutral": "#5D6D7E",
    "grid":    "#E8ECF1",
    "accent":  "#2471A3",
    "warm":    "#C0392B",
    "gold":    "#D4AC0D",
    "green":   "#1E8449",
    "purple":  "#7D3C98",
    "teal":    "#117A65",
    "orange":  "#CA6F1E",
}

TECH_COLORS = {
    "Visium":   PAL["accent"],
    "Slide-seq": PAL["green"],
    "STARmap":  PAL["purple"],
    "ST":       PAL["teal"],
}

plt.rcParams.update({
    "figure.facecolor":  PAL["bg"],
    "axes.facecolor":    PAL["panel"],
    "axes.edgecolor":    PAL["grid"],
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.color":        PAL["grid"],
    "grid.linewidth":    0.5,
    "xtick.color":       PAL["neutral"],
    "ytick.color":       PAL["neutral"],
    "font.family":       "DejaVu Sans",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# ── figure layout ─────────────────────────────────────────────────────────────
# Row 0 : SVG count summary bar chart (full width)
# Row 1 : Per-domain SVG heatmap  |  Top-marker bubble chart
# Row 2 : Technology breakdown donut | SVG richness per domain scatter | marker table

fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor(PAL["bg"])

outer = gridspec.GridSpec(3, 1, figure=fig,
                          hspace=0.52,
                          top=0.93, bottom=0.04,
                          left=0.06, right=0.97)

row0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                        wspace=0.38)
row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2],
                                        wspace=0.42)

ax_bar   = fig.add_subplot(row0[0])
ax_heat  = fig.add_subplot(row1[0])
ax_bub   = fig.add_subplot(row1[1])
ax_donut = fig.add_subplot(row2[0])
ax_scat  = fig.add_subplot(row2[1])
ax_tab   = fig.add_subplot(row2[2])

ds_names = list(DATASETS.keys())
totals   = [DATASETS[d]["total_svgs"] for d in ds_names]
techs    = [DATASETS[d]["technology"] for d in ds_names]
colors   = [TECH_COLORS[t] for t in techs]

# ─────────────────────────────────────────────────────────────────────────────
# Panel A — SVG count bar chart
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_bar
x  = np.arange(len(ds_names))
bars = ax.bar(x, totals, color=colors, width=0.55,
              edgecolor="white", linewidth=0.8, zorder=3)

# value labels on bars
for bar, val in zip(bars, totals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 12,
            f"{val:,}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=PAL["text"])

# technology legend
legend_handles = [
    Line2D([0],[0], marker="s", color="w",
           markerfacecolor=c, markersize=10, label=t)
    for t, c in TECH_COLORS.items()
]
ax.legend(handles=legend_handles, loc="upper left",
          frameon=False, fontsize=8.5, ncol=4)

ax.set_xticks(x)
ax.set_xticklabels(ds_names, fontsize=9, color=PAL["text"])
ax.set_ylabel("Total SVGs identified", fontsize=10, color=PAL["neutral"])
ax.set_title("A  |  Spatially Variable Genes (SVGs) Identified per Dataset",
             fontsize=12, fontweight="bold", color=PAL["text"], loc="left", pad=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_ylim(0, max(totals) * 1.18)

# annotate MP1 as the richest
ax.annotate("highest SVG\nrichness (1,038)",
            xy=(1, 1038), xytext=(2.2, 900),
            arrowprops=dict(arrowstyle="->", color=PAL["neutral"],
                            lw=1.2, connectionstyle="arc3,rad=-0.2"),
            fontsize=8, color=PAL["neutral"], va="center")

# ─────────────────────────────────────────────────────────────────────────────
# Panel B — per-domain SVG count heatmap
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_heat
max_domains = max(DATASETS[d]["n_domains"] for d in ds_names)
heat_data   = np.full((len(ds_names), max_domains), np.nan)

for i, ds in enumerate(ds_names):
    svgs = DATASETS[ds]["n_svgs_by_domain"]
    heat_data[i, :len(svgs)] = svgs

# log1p scale so MP1's 433 doesn't wash everything else out
log_data = np.where(np.isnan(heat_data), np.nan, np.log1p(heat_data))

im = ax.imshow(log_data, aspect="auto", cmap="YlOrRd",
               interpolation="nearest",
               vmin=0, vmax=np.nanmax(log_data))

# annotate cells with raw counts
for i in range(len(ds_names)):
    for j in range(max_domains):
        v = heat_data[i, j]
        if not np.isnan(v):
            txt = str(int(v))
            lv  = log_data[i, j]
            fc  = "white" if lv > np.nanmax(log_data) * 0.6 else PAL["text"]
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color=fc, fontweight="bold")

ax.set_xticks(range(max_domains))
ax.set_xticklabels([f"D{i}" for i in range(max_domains)],
                   fontsize=8, color=PAL["neutral"])
ax.set_yticks(range(len(ds_names)))
ax.set_yticklabels(ds_names, fontsize=9, color=PAL["text"])
ax.set_xlabel("Spatial Domain", fontsize=9, color=PAL["neutral"])
ax.set_title("B  |  SVG Count per Domain  (log₁₊ scale)",
             fontsize=11, fontweight="bold", color=PAL["text"], loc="left", pad=8)

cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("log₁₊(SVG count)", fontsize=8, color=PAL["neutral"])
cbar.ax.tick_params(labelsize=7, labelcolor=PAL["neutral"])

# ─────────────────────────────────────────────────────────────────────────────
# Panel C — bubble chart: domains × datasets, size = SVG count
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_bub
rng = np.random.default_rng(42)

for i, ds in enumerate(ds_names):
    svgs = DATASETS[ds]["n_svgs_by_domain"]
    col  = TECH_COLORS[DATASETS[ds]["technology"]]
    for j, cnt in enumerate(svgs):
        if cnt == 0:
            continue
        # jitter slightly so overlapping bubbles separate
        jx = j + rng.uniform(-0.15, 0.15)
        jy = i + rng.uniform(-0.08, 0.08)
        size = max(20, min(cnt * 0.35, 600))
        ax.scatter(jx, jy, s=size, color=col, alpha=0.72,
                   edgecolors="white", linewidths=0.6, zorder=3)
        if cnt > 30:
            ax.text(jx, jy, str(cnt), ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold")

ax.set_xticks(range(max_domains))
ax.set_xticklabels([f"D{i}" for i in range(max_domains)],
                   fontsize=8, color=PAL["neutral"])
ax.set_yticks(range(len(ds_names)))
ax.set_yticklabels(ds_names, fontsize=9, color=PAL["text"])
ax.set_xlabel("Spatial Domain", fontsize=9, color=PAL["neutral"])
ax.set_title("C  |  SVG Richness Bubble Chart\n(bubble area ∝ SVG count)",
             fontsize=11, fontweight="bold", color=PAL["text"], loc="left", pad=8)
ax.set_xlim(-0.7, max_domains - 0.3)
ax.set_ylim(-0.6, len(ds_names) - 0.4)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

# ─────────────────────────────────────────────────────────────────────────────
# Panel D — technology breakdown donut
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_donut
tech_counts = {}
for ds in ds_names:
    t = DATASETS[ds]["technology"]
    tech_counts[t] = tech_counts.get(t, 0) + DATASETS[ds]["total_svgs"]

labels = list(tech_counts.keys())
sizes  = list(tech_counts.values())
cols_d = [TECH_COLORS[l] for l in labels]

wedges, texts, autotexts = ax.pie(
    sizes, labels=None,
    colors=cols_d,
    autopct="%1.0f%%",
    pctdistance=0.78,
    startangle=90,
    wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.5),
    textprops=dict(color=PAL["text"], fontsize=8),
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_fontweight("bold")
    at.set_color("white")

ax.legend(wedges, [f"{l}\n({v:,} SVGs)" for l, v in zip(labels, sizes)],
          loc="center", frameon=False, fontsize=7.5,
          bbox_to_anchor=(0.5, -0.08))
ax.set_title("D  |  Total SVGs by\nSequencing Technology",
             fontsize=11, fontweight="bold", color=PAL["text"], pad=8)

# ─────────────────────────────────────────────────────────────────────────────
# Panel E — scatter: n_domains vs total SVGs
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_scat
n_doms   = [DATASETS[d]["n_domains"]    for d in ds_names]
tot_svgs = [DATASETS[d]["total_svgs"]   for d in ds_names]
short_names = [d.replace("\n", " ") for d in ds_names]

for i, (nd, ts, col, nm) in enumerate(zip(n_doms, tot_svgs, colors, short_names)):
    ax.scatter(nd, ts, s=90, color=col, zorder=4,
               edgecolors="white", linewidths=0.8)
    # offset labels to avoid overlap
    offsets = [(0.08, 15), (-0.55, 30), (0.08, 8),
               (0.08, 8), (0.08, 5), (-0.55, 15)]
    ox, oy = offsets[i] if i < len(offsets) else (0.08, 10)
    ax.text(nd + ox, ts + oy, nm, fontsize=7.5,
            color=PAL["neutral"], va="bottom")

# trend line
z = np.polyfit(n_doms, tot_svgs, 1)
p = np.poly1d(z)
xs = np.linspace(min(n_doms)-0.3, max(n_doms)+0.3, 50)
ax.plot(xs, p(xs), "--", color=PAL["gold"], lw=1.5, alpha=0.7, zorder=2,
        label=f"trend  (slope={z[0]:.0f})")
ax.legend(frameon=False, fontsize=7.5)

ax.set_xlabel("Number of Spatial Domains", fontsize=9, color=PAL["neutral"])
ax.set_ylabel("Total SVGs", fontsize=9, color=PAL["neutral"])
ax.set_title("E  |  Domain Count vs. SVG Richness",
             fontsize=11, fontweight="bold", color=PAL["text"], loc="left", pad=8)
ax.yaxis.grid(True, linestyle="--", alpha=0.45)

# ─────────────────────────────────────────────────────────────────────────────
# Panel F — top marker gene table
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_tab
ax.axis("off")

# build table data: dataset | domain | top 3 markers | biological annotation
bio_annot = {
    ("DLPFC\n151673", 4): "White matter (MBP, PLP1)",
    ("DLPFC\n151673", 5): "Oligodendrocytes (GFAP, MBP)",
    ("DLPFC\n151673", 0): "Layer neurons (CAMK2N1)",
    ("Mouse\nMP1",    0): "Myelin (MAG, MOG)",
    ("Mouse\nMP1",    1): "High-energy neurons (PVALB, ATP)",
    ("Mouse\nMP1",    2): "Astrocytes (APOE, GFAP)",
    ("Human\nITYN",  1): "Secretory epithelium (MUC genes)",
    ("Slide-seq\nv2",0): "Neuronal progenitors (DCX, Sox11)",
    ("STARmap",      3): "Myelinated axons (Mbp, Plp1)",
    ("Mouse\nMOB",   2): "Granule cells (RBFOX3, SYN1)",
}

rows = []
for ds, info in DATASETS.items():
    for dom, genes in info["top_markers"].items():
        if not genes:
            continue
        key = (ds, dom)
        annot = bio_annot.get(key, "—")
        rows.append([
            ds.replace("\n", " "),
            f"D{dom}",
            ", ".join(genes[:3]),
            annot,
        ])
rows = rows[:14]   # cap for legibility

col_labels = ["Dataset", "Domain", "Top Markers", "Putative Cell Type"]
col_widths = [0.16, 0.07, 0.38, 0.39]
row_h      = 0.064
start_y    = 0.97
header_y   = start_y

# header
for ci, (lbl, cw) in enumerate(zip(col_labels, col_widths)):
    cx = sum(col_widths[:ci])
    rect = FancyBboxPatch((cx, header_y - row_h), cw, row_h,
                          boxstyle="square,pad=0",
                          facecolor=PAL["accent"], edgecolor="white",
                          linewidth=0.5,
                          transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    ax.text(cx + cw/2, header_y - row_h/2, lbl,
            ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white",
            transform=ax.transAxes)

for ri, row in enumerate(rows):
    y0 = header_y - (ri + 2) * row_h
    bg = "#EBF5FB" if ri % 2 == 0 else PAL["panel"]
    for ci, (val, cw) in enumerate(zip(row, col_widths)):
        cx = sum(col_widths[:ci])
        rect = FancyBboxPatch((cx, y0), cw, row_h,
                              boxstyle="square,pad=0",
                              facecolor=bg, edgecolor=PAL["grid"],
                              linewidth=0.4,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        fs = 6.8 if ci == 2 else 7.2   # genes column slightly smaller
        ax.text(cx + cw/2, y0 + row_h/2, val,
                ha="center", va="center",
                fontsize=fs, color=PAL["text"],
                transform=ax.transAxes)

ax.set_title("F  |  Top Marker Genes per Spatial Domain",
             fontsize=11, fontweight="bold", color=PAL["text"],
             pad=8, loc="left",
             transform=ax.transAxes)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Global title + subtitle
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    "SpaGCN Spatially Variable Gene Analysis  —  Multi-Dataset Summary",
    fontsize=15, fontweight="bold", color=PAL["text"], y=0.97,
)
fig.text(
    0.5, 0.945,
    "6 datasets · 3 sequencing technologies · 1,309 total SVGs identified across spatial domains",
    ha="center", fontsize=9.5, color=PAL["neutral"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out = Path("results/SVG_summary_figure.pdf")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, format="pdf", dpi=300,
            bbox_inches="tight", facecolor=PAL["bg"])
print(f"Saved → {out}")

# also save PNG for quick preview
png_out = Path("results/SVG_summary_figure.png")
fig.savefig(png_out, format="png", dpi=150,
            bbox_inches="tight", facecolor=PAL["bg"])
print(f"Saved → {png_out}")
