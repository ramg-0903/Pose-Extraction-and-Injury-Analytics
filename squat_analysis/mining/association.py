"""
association.py  —  Stage 4c
============================
Association rule mining on binary form-error flags.

Finds co-occurring patterns like:
    {shallow_squat, rushed_descent} → {excessive_lean}  (conf=0.82, lift=2.1)

Uses mlxtend's FP-Growth (faster than Apriori on sparse binary data).
Falls back to Apriori if mlxtend FP-Growth is unavailable.

Entry point:
    run_association(df, flag_cols, output_dir, ...) -> pd.DataFrame

Outputs saved to output_dir:
    association_rules.csv       all rules passing support/confidence/lift
    association_itemsets.csv    frequent itemsets before rule generation
    association_meta.json       run parameters and summary statistics
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


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
        df:             DataFrame containing binary flag columns (0/1/NaN).
        flag_cols:      Column names to use as items. Must be binary.
        output_dir:     Where to save outputs.
        min_support:    Minimum itemset support (default 0.10 = 10% of reps).
        min_confidence: Minimum rule confidence (default 0.60).
        min_lift:       Minimum lift to filter trivial rules (default 1.2).

    Returns:
        pd.DataFrame of association rules, sorted by lift descending.
        Empty DataFrame if no rules found.
    """
    try:
        from mlxtend.frequent_patterns import fpgrowth, association_rules
        _backend = "fpgrowth"
    except ImportError:
        try:
            from mlxtend.frequent_patterns import apriori as fpgrowth
            from mlxtend.frequent_patterns import association_rules
            _backend = "apriori"
        except ImportError:
            raise ImportError(
                "mlxtend is required for association rule mining. "
                "Install with: pip install mlxtend"
            )

    # ── Prepare binary transaction matrix ────────────────────────────────────
    available = [c for c in flag_cols if c in df.columns]
    missing   = [c for c in flag_cols if c not in df.columns]
    if missing:
        warnings.warn(f"Flag columns not found, skipping: {missing}")
    if not available:
        raise RuntimeError("No valid flag columns found in dataset.")

    transactions = df[available].copy()

    # Point 1 — fill NaN as 0 (flag absent) instead of dropping rows
    n_nan_cells = int(transactions.isna().sum().sum())
    if n_nan_cells > 0:
        transactions = transactions.fillna(0)
        warnings.warn(
            f"Filled {n_nan_cells} NaN flag values with 0 (flag absent). "
            f"All {len(transactions)} rows retained."
        )

    # Point 2 — validate binary nature after fill
    for col in available:
        unique = set(transactions[col].unique())
        if not unique.issubset({0, 1, 0.0, 1.0, True, False}):
            raise ValueError(
                f"Flag '{col}' contains non-binary values: {unique}. "
                f"All flags must be 0/1 before association mining."
            )

    # Convert to bool (mlxtend requirement)
    transactions = transactions.astype(bool)

    n_transactions = len(transactions)
    n_items        = len(available)

    print(f"  Transactions    : {n_transactions} reps × {n_items} flags")
    print(f"  Backend         : {_backend}")
    print(f"  min_support     : {min_support}  "
          f"(= {int(min_support * n_transactions)} reps)")
    print(f"  min_confidence  : {min_confidence}")
    print(f"  min_lift        : {min_lift}")

    # ── Flag prevalence report ────────────────────────────────────────────────
    prevalence = transactions.mean().sort_values(ascending=False)
    print(f"\n  Flag prevalence:")
    for flag, pct in prevalence.items():
        bar = "█" * int(pct * 20)
        print(f"    {flag:<25} {pct*100:5.1f}%  {bar}")

    # Point 5 — warn on very frequent flags (>85%)
    dominant_flags = prevalence[prevalence > 0.85].index.tolist()
    if dominant_flags:
        warnings.warn(
            f"Flags with >85% prevalence generate uninformative rules: "
            f"{dominant_flags}. Consider excluding from ARM."
        )

    # Warn if any flag below min_support
    rare_flags = prevalence[prevalence < min_support].index.tolist()
    if rare_flags:
        warnings.warn(
            f"Flags below min_support ({min_support}) — "
            f"will not appear in rules: {rare_flags}"
        )

    # Point 4 — flag correlation warning
    flag_vals  = transactions.astype(float)
    corr_matrix = flag_vals.corr()
    high_corr  = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            c = abs(corr_matrix.iloc[i, j])
            if c > 0.7:
                high_corr.append(
                    (available[i], available[j], round(float(c), 3))
                )
    if high_corr:
        warnings.warn(
            f"Highly correlated flag pairs (r>0.7) — rules between these "
            f"may be redundant: {high_corr}"
        )

    # ── Mine frequent itemsets ────────────────────────────────────────────────
    itemsets = fpgrowth(
        transactions,
        min_support=min_support,
        use_colnames=True,
    )

    if len(itemsets) == 0:
        warnings.warn(
            f"No frequent itemsets found at min_support={min_support}. "
            f"Try lowering ARM_MIN_SUPPORT in config."
        )
        _save_empty(output_dir, min_support, min_confidence, min_lift,
                    n_transactions, n_items, available)
        return pd.DataFrame()

    print(f"\n  Frequent itemsets: {len(itemsets)} "
          f"(support ≥ {min_support})")

    # ── Generate rules ────────────────────────────────────────────────────────
    # Point 8 — generate with low confidence threshold, then apply both
    # confidence AND lift simultaneously (avoids keeping rules that pass
    # confidence but fail lift)
    rules = association_rules(
        itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )

    if len(rules) == 0:
        warnings.warn(
            f"No rules found at min_confidence={min_confidence}. "
            f"Try lowering ARM_MIN_CONFIDENCE in config."
        )
        _save_empty(output_dir, min_support, min_confidence, min_lift,
                    n_transactions, n_items, available)
        return pd.DataFrame()

    # Point 8+9 — apply lift filter; return empty if nothing passes (no fallback)
    rules_filtered = rules[
        (rules["confidence"] >= min_confidence) &
        (rules["lift"]       >= min_lift)
    ].copy()

    n_before_lift = len(rules)
    if len(rules_filtered) == 0:
        warnings.warn(
            f"All {n_before_lift} rules failed lift >= {min_lift}. "
            f"Rules are no better than chance. "
            f"Try lowering ARM_MIN_LIFT in config.py."
        )
        _save_empty(output_dir, min_support, min_confidence, min_lift,
                    n_transactions, n_items, available)
        return pd.DataFrame()

    rules_filtered = rules_filtered.sort_values("lift", ascending=False)

    # ── Bootstrap stability (point 13) ────────────────────────────────────────
    # Mine rules on 5 random 80% subsets — report what fraction of final rules
    # appear in majority (≥3/5) of bootstrap runs
    n_bootstrap   = 5
    bootstrap_pct = 0.80
    rule_counts   = {}

    for seed in range(n_bootstrap):
        rng     = np.random.default_rng(seed)
        idx     = rng.choice(len(transactions), size=int(len(transactions) * bootstrap_pct),
                             replace=False)
        t_boot  = transactions.iloc[idx]
        try:
            items_b = fpgrowth(t_boot, min_support=min_support, use_colnames=True)
            if len(items_b) == 0:
                continue
            rules_b = association_rules(items_b, metric="confidence",
                                        min_threshold=min_confidence)
            rules_b = rules_b[rules_b["lift"] >= min_lift]
            for _, r in rules_b.iterrows():
                key = (
                    ", ".join(sorted(r["antecedents"])),
                    ", ".join(sorted(r["consequents"])),
                )
                rule_counts[key] = rule_counts.get(key, 0) + 1
        except Exception:
            continue

    majority_threshold = n_bootstrap // 2 + 1  # 3 out of 5

    # ── Clean up output columns ───────────────────────────────────────────────
    rules_filtered["antecedents"] = rules_filtered["antecedents"].apply(
        lambda x: ", ".join(sorted(x))
    )
    rules_filtered["consequents"] = rules_filtered["consequents"].apply(
        lambda x: ", ".join(sorted(x))
    )

    # Point 13 — attach bootstrap stability score to each rule
    rules_filtered["bootstrap_support"] = rules_filtered.apply(
        lambda r: rule_counts.get((r["antecedents"], r["consequents"]), 0),
        axis=1
    )
    rules_filtered["is_stable"] = (
        rules_filtered["bootstrap_support"] >= majority_threshold
    ).astype(int)

    rules_out = rules_filtered[[
        "antecedents", "consequents",
        "support", "confidence", "lift",
        "leverage", "conviction",
        "bootstrap_support", "is_stable",
    ]].round(4)

    # ── Print top rules ───────────────────────────────────────────────────────
    n_stable = int(rules_out["is_stable"].sum())
    print(f"\n  Rules generated : {len(rules_out)} "
          f"(from {n_before_lift} before lift filter, "
          f"{n_stable} stable across {n_bootstrap} bootstrap runs)")
    print(f"\n  Top 10 rules by lift:")
    print(f"  {'Antecedents':<35} {'→':2} {'Consequents':<25} "
          f"{'Sup':>6} {'Conf':>6} {'Lift':>6} {'Stable':>7}")
    print(f"  {'-'*97}")
    for _, row in rules_out.head(10).iterrows():
        stable = "✓" if row["is_stable"] else " "
        print(f"  {row['antecedents']:<35}  → "
              f"{row['consequents']:<25} "
              f"{row['support']:6.3f} "
              f"{row['confidence']:6.3f} "
              f"{row['lift']:6.3f} "
              f"{'['+stable+']':>7}")

    # ── Highlight injury-relevant rules ──────────────────────────────────────
    # Rules where high-risk flags appear in consequent
    risk_flags  = {"excessive_lean", "asymmetric_depth", "high_jerk",
                   "unstable_return", "incomplete_depth"}
    risk_rules  = rules_out[
        rules_out["consequents"].apply(
            lambda x: bool(risk_flags & set(x.split(", ")))
        )
    ]
    if len(risk_rules) > 0:
        print(f"\n  Injury-relevant rules "
              f"(risk flag in consequent): {len(risk_rules)}")
        for _, row in risk_rules.head(5).iterrows():
            print(f"    {row['antecedents']} → {row['consequents']} "
                  f"(lift={row['lift']:.2f})")

    # ── Save outputs ──────────────────────────────────────────────────────────
    itemsets_out = itemsets.copy()
    itemsets_out["itemsets"] = itemsets_out["itemsets"].apply(
        lambda x: ", ".join(sorted(x))
    )
    itemsets_out = itemsets_out.sort_values("support", ascending=False)

    rules_out.to_csv(output_dir / "association_rules.csv", index=False)
    itemsets_out.to_csv(output_dir / "association_itemsets.csv", index=False)

    with open(output_dir / "association_meta.json", "w") as f:
        json.dump({
            "backend":             _backend,
            "n_transactions":      int(n_transactions),
            "n_flags":             int(n_items),
            "flags_used":          available,
            "flags_missing":       missing,
            "rare_flags":          rare_flags,
            "dominant_flags":      dominant_flags,
            "high_corr_pairs":     high_corr,
            "min_support":         min_support,
            "min_confidence":      min_confidence,
            "min_lift":            min_lift,
            "n_itemsets":          int(len(itemsets)),
            "n_rules_total":       int(n_before_lift),
            "n_rules_filtered":    int(len(rules_out)),
            "n_rules_stable":      int(n_stable),
            "n_injury_rules":      int(len(risk_rules)),
            "bootstrap_runs":      n_bootstrap,
            "bootstrap_pct":       bootstrap_pct,
            "nan_cells_filled":    int(n_nan_cells),
        }, f, indent=2)

    print(f"\n  Saved: association_rules.csv, association_itemsets.csv, "
          f"association_meta.json")

    return rules_out


def _save_empty(output_dir, min_support, min_confidence, min_lift,
                n_transactions, n_items, flags_used):
    """Save empty outputs when no rules are found."""
    pd.DataFrame().to_csv(output_dir / "association_rules.csv",   index=False)
    pd.DataFrame().to_csv(output_dir / "association_itemsets.csv", index=False)
    with open(output_dir / "association_meta.json", "w") as f:
        json.dump({
            "n_transactions": int(n_transactions),
            "n_flags":        int(n_items),
            "flags_used":     flags_used,
            "min_support":    min_support,
            "min_confidence": min_confidence,
            "min_lift":       min_lift,
            "n_rules_filtered": 0,
            "note": "No rules found — try lowering thresholds in config.",
        }, f, indent=2)

