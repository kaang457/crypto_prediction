"""Writes the results table and figures under reports/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import strategy_returns  # noqa: E402

# Fixed order so each model keeps its color in every figure; baselines are gray.
COLORS = {
    "Ridge": "#3987e5",
    "HAR": "#3987e5",
    "Random forest": "#d95926",
    "Gradient boosting": "#199e70",
    "Persistence": "#9a968f",
    "Drift": "#c9c5be",
    "Trailing 30 days": "#9a968f",
    "Trailing 7 days": "#c9c5be",
    "Buy and hold": "#2b2a28",
}
INK, MUTED, GRID = "#2b2a28", "#6f6b65", "#e6e3de"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def skill_chart(skill: pd.Series, title: str, xlabel: str, path: Path) -> None:
    rows = skill.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.55 * len(rows)))
    ax.barh(rows.index, rows.to_numpy(), color=[COLORS[n] for n in rows.index], height=0.55)
    ax.axvline(0, color=INK, linewidth=1)
    span = max(0.5, float(np.abs(rows).max()) * 1.6)
    pad = span * 0.02
    for y, v in enumerate(rows.to_numpy()):
        ax.text(v + (pad if v >= 0 else -pad), y, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right", color=INK)
    ax.set_xlim(-span, span)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def equity_chart(preds: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    actual = preds["actual"]
    ax.plot((np.exp(actual)).cumprod(), color=COLORS["Buy and hold"], label="Buy and hold", linewidth=2.4)
    for name in ["Ridge", "Random forest", "Gradient boosting"]:
        ax.plot((1 + strategy_returns(actual, preds[name])).cumprod(), color=COLORS[name], label=f"{name} long/cash")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title("Trading on the forecasts, after 0.1% fees per trade")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def recent_chart(preds: pd.DataFrame, close: pd.Series, model: str, path: Path, days: int = 120) -> None:
    window = preds.iloc[-days:]
    today = close.loc[window.index]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(window.index + pd.Timedelta(days=1), today * np.exp(window["actual"]), color=INK, label="Actual close", linewidth=2.4)
    ax.plot(window.index + pd.Timedelta(days=1), today * np.exp(window[model]), color=COLORS[model], label=f"{model} forecast")
    ax.plot(
        window.index + pd.Timedelta(days=1), today, color=COLORS["Persistence"], label="Yesterday's close", linestyle="--", linewidth=1.6
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    ax.set_title(f"Last {days} days of out-of-sample forecasts")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def volatility_chart(preds: pd.DataFrame, model: str, baseline: str, path: Path, days: int = 730) -> None:
    window = preds.iloc[-days:]
    annual = np.sqrt(365) * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(window.index, np.exp(window["actual"]) * annual, color=INK, label="Realised (next 7 days)", linewidth=1.4)
    ax.plot(window.index, np.exp(window[baseline]) * annual, color=COLORS[baseline], label=baseline, linestyle="--", linewidth=1.6)
    ax.plot(window.index, np.exp(window[model]) * annual, color=COLORS[model], label=f"{model} forecast")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.0f}%"))
    ax.set_ylim(0, float(np.exp(window["actual"]).max()) * annual * 1.3)  # headroom for the legend
    ax.set_ylabel("Annualised volatility")
    ax.set_title(f"Next-week volatility, last {days} days out of sample")
    ax.legend(loc="upper right", ncols=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


VOL_COLUMNS = {
    "mae_log": ("MAE (log vol)", "{:.3f}"),
    "typical_miss_pct": ("Typical miss", "{:.0f}%"),
    "r2_log": ("R² (log vol)", "{:.3f}"),
    "skill_pct": ("Skill vs trailing 30 days", "{:+.1f}%"),
    "p_value": ("p-value", "{:.3g}"),
}

TABLE_COLUMNS = {
    "mae_usd": ("MAE", "${:,.0f}"),
    "mape_pct": ("MAPE", "{:.2f}%"),
    "r2_price": ("R² price", "{:.4f}"),
    "r2_return": ("R² return", "{:+.4f}"),
    "skill_vs_persistence_pct": ("Skill vs persistence", "{:+.2f}%"),
    "p_value": ("p-value", "{:.2g}"),
    "direction_acc_pct": ("Direction hit rate", "{:.1f}%"),
    "strategy_cagr_pct": ("Long/cash CAGR", "{:+.1f}%"),
    "strategy_max_drawdown_pct": ("Max drawdown", "{:.0f}%"),
}


def markdown_table(summary: pd.DataFrame, columns: dict[str, tuple[str, str]] = TABLE_COLUMNS) -> str:
    head = "| Model | " + " | ".join(label for label, _ in columns.values()) + " |"
    sep = "|---|" + "---:|" * len(columns)
    lines = [head, sep]
    for model, row in summary.iterrows():
        cells = ["–" if pd.isna(row[col]) else fmt.format(row[col]) for col, (_, fmt) in columns.items()]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write(
    returns: pd.DataFrame,
    return_summary: pd.DataFrame,
    vol: pd.DataFrame,
    vol_summary: pd.DataFrame,
    close: pd.Series,
    out: Path,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    candidates = return_summary.drop(index=["Persistence", "Drift", "Buy and hold"])
    best_return = str(candidates["mae_usd"].idxmin())
    best_vol = str(vol_summary.drop(index=["Trailing 30 days", "Trailing 7 days"])["mae_log"].idxmin())

    skill_chart(
        return_summary["skill_vs_persistence_pct"].dropna().drop("Persistence"),
        "Direction: price error versus the random-walk forecast",
        'Lower price error than "tomorrow = today" (%)',
        out / "return_skill.png",
    )
    skill_chart(
        vol_summary["skill_pct"].drop("Trailing 30 days"),
        "Risk: volatility error versus the trailing 30-day estimate",
        "Lower error than the past 30 days' volatility (%)",
        out / "volatility_skill.png",
    )
    equity_chart(returns, out / "equity.png")
    recent_chart(returns, close, best_return, out / "recent.png")
    volatility_chart(vol, best_vol, "Trailing 30 days", out / "volatility.png")
    returns.to_csv(out / "return_predictions.csv", float_format="%.6f")
    vol.to_csv(out / "volatility_predictions.csv", float_format="%.6f")

    def span(frame: pd.DataFrame) -> str:
        return f"{frame.index[0].date()} → {frame.index[-1].date()}, {len(frame):,} days"

    (out / "results.md").write_text(
        "# Results\n\nOut-of-sample walk-forward forecasts. Generated by `btcforecast backtest`.\n\n"
        f"## Direction: next-day return ({span(returns)})\n\n"
        + markdown_table(return_summary)
        + f"\n\n## Risk: next-7-day volatility ({span(vol)})\n\n"
        + markdown_table(vol_summary, VOL_COLUMNS)
        + "\n",
        encoding="utf-8",
    )
