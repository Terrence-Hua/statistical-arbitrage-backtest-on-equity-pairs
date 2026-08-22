"""Statistical test summaries for selected pairs.

Collects Engle-Granger, ADF, and Johansen results from screened pairs and
formats them for display and reporting.

Usage
-----
    from statarb.stats import pair_stats_table

    df = pair_stats_table(pairs, ou_params_map)
    print(df.to_markdown(index=False))
"""

from __future__ import annotations

import pandas as pd

from statarb.screener import PairResult
from statarb.spread import OUParams


# Johansen 95 % critical values for trace test, r=0, k=2 series
# Source: Osterwald-Lenum (1992) Table 1
_JOHANSEN_CV_95 = 15.41


def pair_stats_table(
    pairs: list[PairResult],
    ou_map: dict[str, OUParams] | None = None,
) -> pd.DataFrame:
    """Build a summary table of statistical tests for each pair.

    Parameters
    ----------
    pairs : list[PairResult]
        Output of screen_pairs().
    ou_map : dict or None
        Mapping ``"A/B" -> OUParams`` for each pair.  If provided, adds
        half-life and sigma_eq columns.

    Returns
    -------
    pd.DataFrame
        One row per pair with columns:

        pair, hedge_ratio, eg_pvalue, eg_reject_05,
        adf_pvalue, adf_reject_05, johansen_trace, joh_reject_95,
        half_life_days (if ou_map provided), sigma_eq (if ou_map provided)
    """
    rows = []
    for r in pairs:
        key = f"{r.ticker_a}/{r.ticker_b}"
        row: dict = {
            "pair": key,
            "hedge_ratio": round(r.hedge_ratio, 4),
            "eg_pvalue": round(r.eg_pvalue, 4),
            "eg_reject_05": r.eg_pvalue < 0.05,
            "adf_pvalue": round(r.adf_pvalue, 4) if not pd.isna(r.adf_pvalue) else None,
            "adf_reject_05": (r.adf_pvalue < 0.05) if not pd.isna(r.adf_pvalue) else False,
            "johansen_trace": round(r.joh_trace, 2) if not pd.isna(r.joh_trace) else None,
            "joh_reject_95": (r.joh_trace > _JOHANSEN_CV_95) if not pd.isna(r.joh_trace) else False,
        }
        if ou_map and key in ou_map:
            p = ou_map[key]
            row["half_life_days"] = round(p.half_life_days, 1)
            row["sigma_eq"] = round(p.sigma_eq, 5)
        rows.append(row)

    return pd.DataFrame(rows)


def format_markdown_table(df: pd.DataFrame) -> str:
    """Render *df* as a GitHub-flavoured Markdown table.

    Parameters
    ----------
    df : pd.DataFrame
        Any DataFrame; booleans rendered as yes/no.

    Returns
    -------
    str
        Markdown table string.
    """
    display = df.copy()
    for col in display.select_dtypes(include=bool).columns:
        display[col] = display[col].map({True: "yes", False: "no"})

    try:
        from tabulate import tabulate
        return tabulate(display, headers="keys", tablefmt="pipe", showindex=False)
    except ImportError:
        # fallback: manual rendering
        lines = ["| " + " | ".join(str(c) for c in display.columns) + " |"]
        lines.append("|" + "|".join(["---"] * len(display.columns)) + "|")
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines)
