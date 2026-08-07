#!/usr/bin/env python3
"""Build a defense-ready 5-8 page PDF report from the research outputs.

Generates charts with matplotlib and lays out the report with reportlab
(uses reportlab's built-in STSong-Light CID font, so no system font files
are needed for Chinese text). Output is written next to the report.

Usage: python scripts/build_defense_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

TBL = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
OUT_PDF = ROOT.parent / "reports" / "research_report_defense.pdf"

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
CN = "STSong-Light"

# ---------------------------------------------------------------- colors
INK = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#c0392b")
BLUE = colors.HexColor("#2c6fbb")
GREY = colors.HexColor("#8a8a8a")
LIGHT = colors.HexColor("#eef2f7")

# Matplotlib expects hex strings, not reportlab Color objects.
MBLUE, MRED, MGREY, MLIGHT, MINK = "#2c6fbb", "#c0392b", "#8a8a8a", "#eef2f7", "#1a1a2e"


def style_sheet():
    from reportlab.lib.styles import ParagraphStyle
    return {
        "h1": ParagraphStyle("h1", fontName=CN, fontSize=19, leading=24,
                             textColor=INK, spaceAfter=2),
        "tag": ParagraphStyle("tag", fontName=CN, fontSize=10, leading=14,
                              textColor=GREY, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName=CN, fontSize=13.5, leading=17,
                             textColor=ACCENT, spaceBefore=10, spaceAfter=4),
        "h3": ParagraphStyle("h3", fontName=CN, fontSize=11, leading=14,
                             textColor=INK, spaceBefore=6, spaceAfter=2),
        "body": ParagraphStyle("body", fontName=CN, fontSize=9.8, leading=14.5,
                               textColor=INK, spaceAfter=5),
        "bullet": ParagraphStyle("bullet", fontName=CN, fontSize=9.8, leading=14.2,
                                 textColor=INK, leftIndent=10, spaceAfter=3),
        "cell": ParagraphStyle("cell", fontName=CN, fontSize=8.6, leading=11,
                               textColor=INK),
        "cellhead": ParagraphStyle("cellhead", fontName=CN, fontSize=8.8, leading=11,
                                   textColor=colors.white),
    }


# ---------------------------------------------------------------- charts
def chart_test(models, mae, rmse, qlike, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    x = np.arange(len(models))
    axes[0].bar(x - 0.19, mae, 0.38, label="MAE", color=MBLUE)
    axes[0].bar(x + 0.19, rmse, 0.38, label="RMSE", color=MRED)
    for i in range(len(models)):
        axes[0].text(i - 0.19, mae[i], f"{mae[i]:.4f}", ha="center", va="bottom", fontsize=6.6)
        axes[0].text(i + 0.19, rmse[i], f"{rmse[i]:.4f}", ha="center", va="bottom", fontsize=6.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, fontsize=8)
    axes[0].set_ylabel("波动率尺度"); axes[0].legend(fontsize=8)
    axes[0].set_title("MAE / RMSE（越低越好）", fontsize=10)
    axes[1].bar(x, qlike, 0.55, color=MBLUE)
    for i in range(len(models)):
        axes[1].text(i, qlike[i], f"{qlike[i]:.3f}", ha="center", va="bottom" if qlike[i] < -2.3 else "top", fontsize=6.6)
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, fontsize=8)
    axes[1].set_ylabel("方差尺度")
    axes[1].set_title("QLIKE（越低越好）", fontsize=10)
    axes[1].axhline(0, color=MGREY, lw=0.6)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def chart_expanding(models, locked, exp, path):
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    x = np.arange(len(models)); w = 0.36
    b1 = ax.bar(x - w / 2, locked, w, label="锁定协议（一次性锁参）", color=MLIGHT, edgecolor=MINK)
    b2 = ax.bar(x + w / 2, exp, w, label="扩张窗口重估", color=MBLUE)
    for bars, vals in ((b1, locked), (b2, exp)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8.5)
    ax.set_ylabel("QLIKE（越低越好）")
    ax.set_title("测试段 pooled ALL：锁定 vs 扩张窗口重估（四模型收敛）", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def chart_importance(imp, path):
    imp = imp.sort_values("share")
    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    cols = []
    for f in imp["feature"]:
        base = f.replace("_rv", "").replace("_5d", "").replace("_22d", "")
        cols.append(MRED if "log_vix" in f or "parkinson" in f or "garman" in f else MBLUE)
    bars = ax.barh(imp["feature"], imp["share"], color=cols)
    for rect, v in zip(bars, imp["share"]):
        ax.text(rect.get_width() + 0.004, rect.get_y() + rect.get_height() / 2, f"{v:.1%}", va="center", fontsize=7)
    ax.set_xlabel("跨资产平均 gain 占比")
    ax.set_title("LightGBM 特征重要性：VIX 与区间特征主导，HAR 日/周近乎闲置", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def chart_targeting(items, path):
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9.2, 3.1))
    bars = ax.bar(names, vals, color=[MGREY] * len(names))
    for i in range(len(names)):
        if names[i] == "LightGBM":
            bars[i].set_color(MRED)
        if names[i] == "LGB·校正":
            bars[i].set_color(MBLUE)
        if names[i] == "GARCH":
            bars[i].set_color(MINK)
        ax.text(i, vals[i], f"{vals[i]:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylabel("平均目标偏差（pp）")
    ax.set_title("volatility-targeting 逐资产平均目标偏差（越低越好）", fontsize=10)
    ax.axhline(0, color=MGREY, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


# ---------------------------------------------------------------- data load
def load():
    wf = pd.read_csv(TBL / "walk_forward_metrics.csv")
    tt = wf[(wf.segment == "test") & (wf.asset == "ALL")]
    order = ["historical_rv_21d", "ewma_rv", "har_rv", "garch_rv", "ridge_rv", "lgb_rv"]
    labels = ["历史21日", "EWMA", "HAR", "GARCH", "Ridge", "LightGBM"]
    tt = tt.set_index("forecast").loc[order]
    ec = pd.read_csv(TBL / "expanding_comparison.csv")
    et = ec[(ec.segment == "test") & (ec.asset == "ALL")].pivot(index="model", columns="protocol", values="qlike")
    exp_models = ["garch_rv", "har_rv", "ridge_rv", "lgb_rv"]
    exp_labels = ["GARCH", "HAR", "Ridge", "LightGBM"]
    imp = pd.read_csv(TBL / "lightgbm_importance.csv").groupby("feature")["importance_gain_share"].mean().reset_index()
    imp = imp.rename(columns={"importance_gain_share": "share"}).sort_values("share")
    strat = pd.read_csv(TBL / "strategy_metrics.csv")
    dev = strat[strat.asset != "ALL"].groupby("forecast")["target_deviation"].mean() * 100
    dev = dev[["historical_rv_21d", "ewma_rv", "har_rv", "garch_rv", "ridge_rv", "lgb_rv", "lgb_rv_cal"]]
    dev_labels = ["历史21日", "EWMA", "HAR", "GARCH", "Ridge", "LightGBM", "LGB·校正"]
    return tt, labels, et, exp_models, exp_labels, imp, dev, dev_labels


# ---------------------------------------------------------------- pdf
def main() -> None:
    tt, labels, et, exp_models, exp_labels, imp, dev, dev_labels = load()
    S = style_sheet()
    FIG_DIR.mkdir(exist_ok=True)
    f1, f2, f3, f4 = (FIG_DIR / "def_1_test.png", FIG_DIR / "def_2_expanding.png",
                      FIG_DIR / "def_3_importance.png", FIG_DIR / "def_4_targeting.png")
    chart_test(labels, tt["mae"].to_numpy(), tt["rmse"].to_numpy(), tt["qlike"].to_numpy(), f1)
    chart_expanding(exp_labels, et["locked"].loc[exp_models].to_numpy(), et["expanding"].loc[exp_models].to_numpy(), f2)
    chart_importance(imp, f3)
    chart_targeting(list(zip(dev_labels, dev.to_numpy())), f4)

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=13 * mm,
                            title="多资产波动率预测研究——答辩版",
                            author="multi-asset-volatility-forecasting")
    el = []
    P, B = Paragraph, Spacer

    # ---- page 1: title + research question + headline
    el.append(P("多资产波动率预测研究：统计与树模型的边界", S["h1"]))
    el.append(P("答辩版 · 2026-08 · 项目 `multi-asset-volatility-forecasting` · 6 只 ETF 2010–2026 · 严格防泄漏 walk-forward", S["tag"]))
    el.append(B(1, 2 * mm))
    el.append(P("<b>研究问题</b>：交易日 t 收盘时可获得的信息，能否预测 t+1..t+5 实现波动率？预测精度能否转化为更好的风险控制？", S["body"]))
    el.append(P("<b>模型</b>：两个基线（历史21日、EWMA）+ 四个可学习模型（HAR、GARCH(1,1)、14 特征 Ridge、LightGBM）+ 实验性 HAR-VIX。参数只在训练/验证段锁定，测试段（2024 起）完全样本外；另做扩张窗口逐期重估。", S["body"]))
    el.append(B(1, 3 * mm))

    el.append(P("<b>一句结论</b>：可学习模型在预测与风险配置上确实带来真实但温和的样本外改进；但「谁最好」强烈取决于你用什么指标、在哪一层衡量——统计模型在 QLIKE 与单资产风险控制上领先，树模型在 MAE/RMSE 与稳健性上领先，且所有排序结论都对评估设定敏感。", S["body"]))
    el.append(B(1, 6 * mm))

    el.append(P("1  主结果：可学习模型 vs 基线", S["h2"]))
    el.append(P("测试段 pooled ALL（3,822 个观测）：四个可学习模型的 QLIKE 均显著优于历史基线（DM p&lt;0.05），但彼此之间统计不可分；LightGBM 的 MAE 与 RMSE 全场最优、且显著优于 HAR 与 GARCH，代价是 QLIKE 未进榜首。", S["body"]))
    el.append(Image(str(f1), width=178 * mm, height=63 * mm))
    el.append(B(1, 2 * mm))
    head = ["模型", "MAE", "RMSE", "QLIKE", "说明"]
    rows = [
        ["GARCH", "0.06394", "0.09767", "<b>-2.30494</b>", "QLIKE 最优"],
        ["LightGBM", "<b>0.05992</b>", "<b>0.09616</b>", "-2.28483", "MAE/RMSE 最优；QLIKE 第 3"],
        ["HAR", "0.06482", "0.09742", "-2.30093", "统计基线"],
        ["Ridge", "0.06931", "0.14546", "-2.29848", "高波 RMSE 异常高"],
        ["EWMA", "0.06720", "0.10176", "-2.27127", ""],
        ["历史21日", "0.06819", "0.10520", "-2.23166", "参考点"],
    ]
    data = [[Paragraph("模型", S["cellhead"])] + [Paragraph(h, S["cellhead"]) for h in head[1:]]] + [
        [Paragraph(r[0], S["cell"])] + [Paragraph(c, S["cell"]) for c in r[1:]] for r in rows
    ]
    tbl = Table(data, colWidths=[24 * mm, 20 * mm, 20 * mm, 24 * mm, 64 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    el.append(tbl)
    el.append(B(1, 2 * mm))
    el.append(P("DM 检验：GARCH/HAR/Ridge/LightGBM 两两 QLIKE 均 p&gt;0.25（顶层不可分）；但 MAE 尺度上 LightGBM 显著优于 HAR（p=0.001）与 GARCH（p=0.011）。", S["body"]))
    el.append(PageBreak())

    # ---- page 2: linear vs nonlinear
    el.append(P("2  线性 vs 非线性：特征的故事", S["h2"]))
    el.append(P("Ridge 用同一 14 特征的线性版本：训练段 QLIKE 大幅领先（vs HAR p&lt;0.0001），样本外却与 HAR 不可分、且 MAE/RMSE 更差——线性特征扩充没有兑现。LightGBM 用同样的特征，但其特征重要性显示它重度依赖 VIX 与区间特征，并把信息转化成了显著的 MAE/RMSE 增益。同一个特征集，非线性模型用得上、线性模型用不上。", S["body"]))
    el.append(Image(str(f3), width=158 * mm, height=75 * mm))
    el.append(B(1, 2 * mm))
    el.append(P("对照小结：Ridge（线性）＝样本内强、样本外无增量；LightGBM（树）＝样本外 MAE/RMSE 更好，但 QLIKE 仍与顶层不可分。「膨胀的特征集」本身不是问题，问题在线性假设挡不挡得住信号。", S["body"]))
    el.append(PageBreak())

    # ---- page 3: expanding robustness
    el.append(P("3  协议稳健性：锁参低估了 Ridge 与 LightGBM", S["h2"]))
    el.append(P("主协议把参数在训练段一次性锁死。扩张窗口重估改为：每个评估年用截至前一年底的扩张窗口重拟合（训练至 Y-3、验证 Y-2..Y-1 选超参、只预测 Y 年），无前视；首年 2024 与锁参严格等价（一致性校验已通过）。", S["body"]))
    el.append(Image(str(f2), width=172 * mm, height=60 * mm))
    el.append(B(1, 2 * mm))
    el.append(P("重估后四个模型 QLIKE 收敛到 ±0.003 之内（两两 DM 均 p&gt;0.84）：LightGBM 显著受益（锁参 vs 重估 DM=-2.88，p=0.004），Ridge 的高波 RMSE 从 0.1455 降到 0.1258，GARCH/HAR 几乎不变。「复杂模型无增益」只在一次性锁参协议下成立；参数自适应后，被压制的 Ridge 与 LightGBM 追上统计模型。", S["body"]))
    el.append(P("稳健性延伸：排除 2020 训练后 GARCH/HAR 大幅退化、Ridge 与 LightGBM 反超且仅 LightGBM 对训练样本稳健；Garman–Klass 标签下 GARCH/Ridge/LightGBM 显著优于 HAR。", S["body"]))
    el.append(PageBreak())

    # ---- page 4: strategy layer
    el.append(P("4  应用层：预测精度 → 风险控制（分层兑现）", S["h2"]))
    el.append(P("单资产 volatility-targeting（测试段、目标波动率 10%、杠杆上限 1.5、成本 10bps）：预测精度与风险实现大体同序，GARCH 在盯仓上最优；LightGBM 是例外——点预测最好但逐资产目标偏差最大，因为树模型对高波动资产系统性欠预测、仓位偏大。水平校正（方差-RMS，验证段校准）把偏差从 1.12pp 部分降到 0.67pp，但以约 0.6pp 净收益为代价。", S["body"]))
    el.append(Image(str(f4), width=172 * mm, height=58 * mm))
    el.append(B(1, 2 * mm))
    portfolio = Table([
        [Paragraph("周度风险平价（inverse-forecast）", S["cellhead"]), Paragraph("实现波动率", S["cellhead"]),
         Paragraph("Sharpe", S["cellhead"]), Paragraph("最大回撤", S["cellhead"]), Paragraph("vol-of-vol", S["cellhead"])],
        [Paragraph("GARCH", S["cell"]), Paragraph("10.68%", S["cell"]), Paragraph("1.695", S["cell"]), Paragraph("-10.7%", S["cell"]), Paragraph("3.67pp", S["cell"])],
        [Paragraph("Ridge（重估）", S["cell"]), Paragraph("10.87%", S["cell"]), Paragraph("<b>1.816</b>", S["cell"]), Paragraph("-10.7%", S["cell"]), Paragraph("3.89pp", S["cell"])],
        [Paragraph("LightGBM（重估）", S["cell"]), Paragraph("<b>10.63%</b>", S["cell"]), Paragraph("1.721", S["cell"]), Paragraph("<b>-10.3%</b>", S["cell"]), Paragraph("<b>3.63pp</b>", S["cell"])],
    ], colWidths=[50 * mm, 30 * mm, 28 * mm, 28 * mm, 30 * mm])
    portfolio.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ]))
    el.append(portfolio)
    el.append(B(1, 2 * mm))
    el.append(P("组合层格局反转：扩张重估后，逆 Ridge 波动的组合 Sharpe 大幅领先（1.816 vs 1.695），逆 LightGBM 波动的组合在波动率、回撤、vol-of-vol 三项风险指标上全场最优。预测层的「追上」主要兑现到收益维度与组合层；单资产风险控制仍是 GARCH 的主场。", S["body"]))
    el.append(PageBreak())

    # ---- page 5: conclusions
    el.append(P("5  结论与边界", S["h2"]))
    el.append(P("<b>可学习模型普遍优于历史基线。</b>四者 QLIKE 显著优于基线且顶层不可分；LightGBM 的 MAE/RMSE 最优，GARCH 的 QLIKE 最优。", S["bullet"]))
    el.append(P("<b>非线性改写了「特征扩充无效」的线性结论。</b>Ridge 用不上扩展特征，LightGBM 用上了并把信息变成 MAE/RMSE 增益，但 QLIKE 仍未拉开。", S["bullet"]))
    el.append(P("<b>参数自适应改变了排序。</b>扩张窗口重估下四者 QLIKE 收敛，Ridge 与 LightGBM 追上甚至（组合层）超过 GARCH/HAR；锁参协议系统性压低了复杂模型。", S["bullet"]))
    el.append(P("<b>预测精度 ≠ 风险实现质量。</b>单资产盯仓仍是 GARCH 最优；LightGBM 的例外可被水平校正部分修复但有收益代价；组合层吃的是预测相对质量，树模型因此更占优。", S["bullet"]))
    el.append(B(1, 3 * mm))
    el.append(P("局限：未来五日标签是噪声代理且重叠；测试窗口仅 2024–2026 且以低波动为主；报告主结果在锁参协议下呈现、扩张协议作稳健性层；Ridge/LGB 的水平偏差是结构性的、仅部分可校正。", S["body"]))
    el.append(B(1, 3 * mm))
    el.append(P("<b>一句话</b>：统计与树模型都带来真实而温和的样本外改进，价值在风险配置层面比在预测精度层面更容易兑现；在日频重叠标签、约五年训练的设定下，模型复杂度不提供稳定的 QLIKE 增益，但树模型在误差、稳健性与组合层上占优——没有普适的最优，只有「在哪个指标、哪一层、哪种协议下」的结果。", S["body"]))

    doc.build(el)
    print(f"OK -> {OUT_PDF}")


if __name__ == "__main__":
    main()
