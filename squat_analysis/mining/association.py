"""
Association rule mining on binary form-error flags.

Uses FP-Growth (mlxtend) to discover co-occurring form errors, e.g.:
    {asymmetric_start, rushed_descent} → {asymmetric_depth}  (lift=7.38)

Falls back to Apriori if FP-Growth is unavailable.
Includes bootstrap stability scoring (5 × 80% subsamples).

Outputs (saved to output_dir):
    association_rules.csv      rules passing support/confidence/lift
    association_itemsets.csv   frequent itemsets before rule generation
    association_meta.json      parameters and summary statistics
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_association(
    df: pd.DataFrame,
    flag_cols: list,
    output_dir: Path,
    min_support: float = 0.10,
    min_confidence: float = 0.60,
    min_lift: float = 1.2,
) -> pd.DataFrame:
    """Mine association rules from binary form-error flags.

    Args:
        df:             DataFrame with binary flag columns (0/1/NaN).
        flag_cols:      Column names to use as items.
        output_dir:     Where to save outputs.
        min_support:    Minimum itemset support fraction.
        min_confidence: Minimum rule confidence.
        min_lift:       Minimum lift (filters trivial rules).

    Returns:
        DataFrame of rules sorted by lift, or empty if none found.
    """
    try:
        from mlxtend.frequent_patterns import fpgrowth, association_rules
        backend = "fpgrowth"
    except ImportError:
        try:
            from mlxtend.frequent_patterns import apriori as fpgrowth
            from mlxtend.frequent_patterns import association_rules
            backend = "apriori"
        except ImportError:
            raise ImportError("mlxtend required: pip install mlxtend")

    # ── Prepare binary transaction matrix ─────────────────────────────────
    available = [c for c in flag_cols if c in df.columns]
    missing   = [c for c in flag_cols if c not in df.columns]
    if missing:
        warnings.warn(f"Flag columns not found, skipping: {missing}")
    if not available:
        raise RuntimeError("No valid flag columns in dataset.")

    transactions = df[available].copy()

    # NaN → 0 (flag absent) instead of dropping rows
    n_nan_cells = int(transactions.isna().sum().sum())
    if n_nan_cells > 0:
        transactions = transactions.fillna(0)

    # Validate binary nature
    for col in available:
        unique = set(transactions[col].unique())
        if not unique.issubset({0, 1, 0.0, 1.0, True, False}):
            raise ValueError(f"Flag '{col}' non-binary: {unique}")

    transactions = transactions.astype(bool)
    n_transactions = len(transactions)
    n_items = len(available)

    logger.info("  %d reps × %d flags | backend=%s | "
                "sup=%.3f conf=%.2f lift=%.2f",
                n_transactions, n_items, backend,
                min_support, min_confidence, min_lift)

    # ── Flag prevalence ───────────────────────────────────────────────────
    prevalence = transactions.mean().sort_values(ascending=False)
    for flag, pct in prevalence.items():
        logger.info("    %-25s %5.1f%%", flag, pct * 100)

    dominant_flags = prevalence[prevalence > 0.85].index.tolist()
    if dominant_flags:
        warnings.warn(f"Flags >85% prevalence (uninformative rules): {dominant_flags}")

    rare_flags = prevalence[prevalence < min_support].index.tolist()
    if rare_flags:
        warnings.warn(f"Flags below min_support — won't appear in rules: {rare_flags}")

    # Correlation warning for redundant pairs
    corr = transactions.astype(float).corr()
    high_corr = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            c = abs(corr.iloc[i, j])
            if c > 0.7:
                high_corr.append((available[i], available[j], round(float(c), 3)))
    if high_corr:
        warnings.warn(f"Correlated flag pairs (r>0.7) — rules may be redundant: {high_corr}")

    # ── Mine frequent itemsets ────────────────────────────────────────────
    itemsets = fpgrowth(transactions, min_support=min_support, use_colnames=True)

    if len(itemsets) == 0:
        warnings.warn(f"No frequent itemsets at min_support={min_support}.")
        _save_empty(output_dir, min_support, min_confidence, min_lift,
                    n_transactions, n_items, available)
        return pd.DataFrame()

    logger.info("  Frequent itemsets: %d", len(itemsets))

    # ── Generate and filter rules ─────────────────────────────────────────
    rules = association_rules(itemsets, metric="confidence",
                              min_threshold=min_confidence)

    if len(rules) == 0:
        warnings.warn(f"No rules at min_confidence={min_confidence}.")
        _save_empty(output_dir, min_support, min_confidence, min_lift,
                    n_transactions, n_items, available)
        return pd.DataFrame()

    n_before_lift = len(rules)
    rules_filtered = rules[
        (rules["confidence"] >= min_confidence) &
        (rules["lift"] >= min_lift)
    ].copy()

    if len(rules_filtered) == 0:
        warnings.warn(f"All {n_before_lift} rules failed lift >= {min_lift}.")
        _save_empty(output_dir, min_support, min_confidence, min_lift,
                    n_transactions, n_items, available)
        return pd.DataFrame()

    rules_filtered = rules_filtered.sort_values("lift", ascending=False)

    # ── Bootstrap stability (5 × 80% subsamples) ─────────────────────────
    n_bootstrap = 5
    bootstrap_pct = 0.80
    rule_counts = {}

    for seed in range(n_bootstrap):
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_transactions,
                         size=int(n_transactions * bootstrap_pct), replace=False)
        t_boot = transactions.iloc[idx]
        try:
            items_b = fpgrowth(t_boot, min_support=min_support, use_colnames=True)
            if len(items_b) == 0:
                continue
            rules_b = association_rules(items_b, metric="confidence",
                                        min_threshold=min_confidence)
            rules_b = rules_b[rules_b["lift"] >= min_lift]
            for _, r in rules_b.iterrows():
                key = (", ".join(sorted(r["antecedents"])),
                       ", ".join(sorted(r["consequents"])))
                rule_counts[key] = rule_counts.get(key, 0) + 1
        except Exception:
            continue

    majority = n_bootstrap // 2 + 1

    # ── Format output columns ─────────────────────────────────────────────
    rules_filtered["antecedents"] = rules_filtered["antecedents"].apply(
        lambda x: ", ".join(sorted(x)))
    rules_filtered["consequents"] = rules_filtered["consequents"].apply(
        lambda x: ", ".join(sorted(x)))

    rules_filtered["bootstrap_support"] = rules_filtered.apply(
        lambda r: rule_counts.get((r["antecedents"], r["consequents"]), 0),
        axis=1)
    rules_filtered["is_stable"] = (
        rules_filtered["bootstrap_support"] >= majority).astype(int)

    rules_out = rules_filtered[[
        "antecedents", "consequents",
        "support", "confidence", "lift",
        "leverage", "conviction",
        "bootstrap_support", "is_stable",
    ]].round(4)

    n_stable = int(rules_out["is_stable"].sum())
    logger.info("  Rules: %d total (%d stable across %d bootstraps)",
                len(rules_out), n_stable, n_bootstrap)

    for _, row in rules_out.head(5).iterrows():
        stable = "stable" if row["is_stable"] else ""
        logger.info("    %s → %s  (lift=%.2f %s)",
                    row["antecedents"], row["consequents"],
                    row["lift"], stable)

    # ── Injury-relevant rules ─────────────────────────────────────────────
    risk_flags = {"excessive_lean", "asymmetric_depth", "high_jerk",
                  "unstable_return", "incomplete_depth"}
    risk_rules = rules_out[
        rules_out["consequents"].apply(
            lambda x: bool(risk_flags & set(x.split(", "))))
    ]
    if len(risk_rules) > 0:
        logger.info("  Injury-relevant rules: %d", len(risk_rules))

    # ── Save ──────────────────────────────────────────────────────────────
    itemsets_out = itemsets.copy()
    itemsets_out["itemsets"] = itemsets_out["itemsets"].apply(
        lambda x: ", ".join(sorted(x)))
    itemsets_out = itemsets_out.sort_values("support", ascending=False)

    rules_out.to_csv(output_dir / "association_rules.csv", index=False)
    itemsets_out.to_csv(output_dir / "association_itemsets.csv", index=False)

    with open(output_dir / "association_meta.json", "w") as f:
        json.dump({
            "backend":          backend,
            "n_transactions":   n_transactions,
            "n_flags":          n_items,
            "flags_used":       available,
            "flags_missing":    missing,
            "rare_flags":       rare_flags,
            "dominant_flags":   dominant_flags,
            "high_corr_pairs":  high_corr,
            "min_support":      min_support,
            "min_confidence":   min_confidence,
            "min_lift":         min_lift,
            "n_itemsets":       len(itemsets),
            "n_rules_total":    n_before_lift,
            "n_rules_filtered": len(rules_out),
            "n_rules_stable":   n_stable,
            "n_injury_rules":   len(risk_rules),
            "bootstrap_runs":   n_bootstrap,
            "bootstrap_pct":    bootstrap_pct,
            "nan_cells_filled": n_nan_cells,
        }, f, indent=2)

    logger.info("  Saved: association_rules.csv, association_itemsets.csv")
    return rules_out


def _save_empty(output_dir, min_support, min_confidence, min_lift,
                n_transactions, n_items, flags_used):
    """Save empty outputs when no rules are found."""
    pd.DataFrame().to_csv(output_dir / "association_rules.csv", index=False)
    pd.DataFrame().to_csv(output_dir / "association_itemsets.csv", index=False)
    with open(output_dir / "association_meta.json", "w") as f:
        json.dump({
            "n_transactions": n_transactions, "n_flags": n_items,
            "flags_used": flags_used,
            "min_support": min_support, "min_confidence": min_confidence,
            "min_lift": min_lift, "n_rules_filtered": 0,
            "note": "No rules found — try lowering thresholds.",
        }, f, indent=2)
