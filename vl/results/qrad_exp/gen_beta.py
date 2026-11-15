"""QRAD beta sensitivity — synthetic placeholders, not experimental results.
Run: python plot_qrad_beta_v3.py  |  Dependencies: matplotlib, numpy
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 数据：每个列表依次对应 beta = 0, 0.1, 0.2, 0.3 ----------
# 仅为布局占位；实际填入每个配置在 10 个任务上的平均性能。
BETA = [0, 0.1, 0.2, 0.3]
DATA = {
    "Model A": {
        "10%": [58.0, 59.2, 58.7, 60.0],
        "20%": [64.0, 63.4, 65.1, 64.6],
        "30%": [69.0, 70.1, 69.5, 70.4],
    },
    "Model B": {
        "10%": [59.1, 58.5, 60.0, 59.4],
        "20%": [64.5, 65.4, 64.8, 65.7],
        "30%": [70.3, 69.6, 70.7, 70.0],
    },
    "Model C": {
        "10%": [57.8, 58.8, 59.4, 58.6],
        "20%": [63.5, 64.7, 64.0, 65.0],
        "30%": [68.9, 68.3, 69.7, 69.1],
    },
}

# ---------- 样式：优先在这里调整 ----------
PLACEHOLDER_DATA = True  # 全部换成真实数据后设为 False
COLORS = {"10%": "#4878A8", "20%": "#D58A50", "30%": "#479886"}
FIGSIZE = (7.0, 2.45)  # 英寸；按论文双栏宽度设计
Y_LIM = (56, 72)
Y_TICKS = [56, 60, 64, 68, 72]
FONT_SIZE, TITLE_SIZE, LABEL_SIZE = 9, 10, 10
LINE_WIDTH, MARKER_SIZE = 1.5, 4.2
MARGINS = dict(left=0.080, right=0.988, bottom=0.32, top=0.88, wspace=0.22)
DPI = 600
OUT_DIR = Path(__file__).resolve().parent


def make_figure():
    if len(DATA) != 3:
        raise ValueError("DATA 必须包含三个模型。")
    for model, budgets in DATA.items():
        if set(budgets) != set(COLORS):
            raise ValueError(f"{model}: 请提供 10%、20%、30% 三组数据。")
        for budget, values in budgets.items():
            values = np.asarray(values, dtype=float)
            if values.shape != (len(BETA),) or not np.isfinite(values).all():
                raise ValueError(f"{model}/{budget}: 需要四个有效数值。")
            if np.any((values < Y_LIM[0]) | (values > Y_LIM[1])):
                raise ValueError("数据超出纵轴范围，请调整 Y_LIM 和 Y_TICKS。")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": FONT_SIZE,
        "axes.labelsize": LABEL_SIZE, "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "normal", "axes.linewidth": 0.65,
        "axes.edgecolor": "#4A4A4A", "text.color": "#262626",
        "axes.labelcolor": "#262626",
        "xtick.color": "#454545", "ytick.color": "#454545",
        "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharex=True, sharey=True)
    fig.subplots_adjust(**MARGINS)

    for i, (ax, (model, budgets)) in enumerate(zip(axes, DATA.items())):
        for budget, color in COLORS.items():
            ax.plot(BETA, budgets[budget], label=f"{budget} retention",
                    color=color, linestyle="-", linewidth=LINE_WIDTH,
                    marker="o", markersize=MARKER_SIZE,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.6, solid_capstyle="round", zorder=3)
        ax.set_title(f"({chr(97 + i)}) {model}", pad=7)
        ax.set_xlabel(r"$\beta$", labelpad=2)
        ax.set_xlim(-0.012, 0.312)
        ax.set_xticks(BETA, ["0", "0.1", "0.2", "0.3"])
        ax.set_ylim(*Y_LIM)
        ax.set_yticks(Y_TICKS)
        ax.tick_params(axis="y", labelleft=True)
        ax.tick_params(direction="out", pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)

    axes[0].set_ylabel("Average Performance", labelpad=6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.534, 0.075), frameon=False,
               fontsize=9, handlelength=2.0, handletextpad=0.6,
               columnspacing=2.5, borderaxespad=0)
    if PLACEHOLDER_DATA:
        fig.text(0.534, 0.012, "Placeholder data — layout preview only",
                 ha="center", va="bottom", fontsize=7, color="#777777")
    return fig, axes


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = make_figure()
    stem = "qrad_beta_v3_placeholder" if PLACEHOLDER_DATA else "qrad_beta_v3"
    for suffix in ("png", "pdf", "svg"):
        # 不使用 bbox_inches='tight'，以保留设定的物理尺寸。
        output = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(output, dpi=DPI)
        print(output)
    plt.close(fig)
