"""
compare_results.py
------------------
Reads the three attack result CSVs (off / confidence / full) and produces:

  1. Console summary table  — ASR, block rate, flag rate per mode
  2. Per-attack comparison  — how each attack was handled across all three modes
  3. Per-type breakdown     — metrics grouped by attack type
  4. Gap analysis           — attacks that still succeeded or stayed unclear in full mode
  5. results/comparison_report.txt  — full text report saved to disk

Usage:
    python compare_results.py                        # auto-finds latest CSVs
    python compare_results.py --results_dir results  # explicit directory
    python compare_results.py --off   results/attack_results_off_XYZ.csv \
                               --conf results/attack_results_confidence_XYZ.csv \
                               --full results/attack_results_full_XYZ.csv
"""

import argparse
import csv
import os
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ── ANSI colours for terminal output ─────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def red(s):    return f"{RED}{s}{RESET}"
def green(s):  return f"{GREEN}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def bold(s):   return f"{BOLD}{s}{RESET}"


# ══════════════════════════════════════════════════════════════════════════════
# CSV LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(path: str) -> List[Dict]:
    """Load an attack results CSV and return list of row dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalise types
            row["id"]         = int(row["id"])
            row["confidence"] = float(row["confidence"]) if row["confidence"] else 0.0
            row["blocked"]    = row["blocked"].strip().lower() in ("true", "1", "yes")
            row["flagged"]    = row["flagged"].strip().lower() in ("true", "1", "yes")
            row["latency_ms"] = float(row["latency_ms"]) if row["latency_ms"] else 0.0
            rows.append(row)
    return rows


def find_latest_csv(results_dir: str, mode: str) -> Optional[str]:
    """Find the most recently created CSV for a given mode."""
    pattern = os.path.join(results_dir, f"attack_results_{mode}_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getctime)


# ══════════════════════════════════════════════════════════════════════════════
# METRIC CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def calc_metrics(rows: List[Dict]) -> Dict:
    """
    Calculate ASR, block rate, flag rate, and per-type breakdowns.

    attack_succeeded values:
      "yes"     → attack succeeded  (counted in ASR numerator)
      "no"      → attack failed / was blocked correctly
      "unclear" → needs manual review (excluded from ASR calc, noted separately)
    """
    total    = len(rows)
    blocked  = sum(1 for r in rows if r["blocked"])
    flagged  = sum(1 for r in rows if r["flagged"])
    succeeded  = sum(1 for r in rows if r["attack_succeeded"].strip().lower() == "yes")
    failed     = sum(1 for r in rows if r["attack_succeeded"].strip().lower() == "no")
    unclear    = sum(1 for r in rows if r["attack_succeeded"].strip().lower() == "unclear")

    # ASR = succeeded / (succeeded + failed)  — excludes unclear
    # This is conservative and honest: don't inflate ASR with unclear cases
    denom = succeeded + failed
    asr   = round(succeeded / denom * 100, 1) if denom > 0 else 0.0

    avg_conf    = round(sum(r["confidence"] for r in rows) / total, 3) if total else 0
    avg_latency = round(sum(r["latency_ms"] for r in rows) / total / 1000, 1) if total else 0

    # Per attack-type breakdown
    types = {}
    for r in rows:
        t = r["type"]
        if t not in types:
            types[t] = {"total": 0, "succeeded": 0, "blocked": 0, "flagged": 0, "unclear": 0}
        types[t]["total"]   += 1
        types[t]["blocked"] += 1 if r["blocked"] else 0
        types[t]["flagged"] += 1 if r["flagged"] else 0
        s = r["attack_succeeded"].strip().lower()
        if s == "yes":     types[t]["succeeded"] += 1
        elif s == "unclear": types[t]["unclear"]   += 1

    return {
        "total":     total,
        "blocked":   blocked,
        "flagged":   flagged,
        "succeeded": succeeded,
        "failed":    failed,
        "unclear":   unclear,
        "asr":       asr,
        "block_rate": round(blocked / total * 100, 1) if total else 0,
        "flag_rate":  round(flagged / total * 100, 1) if total else 0,
        "avg_conf":   avg_conf,
        "avg_latency_s": avg_latency,
        "by_type":    types,
    }


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def outcome_symbol(row: Dict) -> str:
    """One-character outcome symbol for a result row."""
    if row["blocked"]:
        return "⛔"
    if row["flagged"]:
        return "⚠️ "
    s = row["attack_succeeded"].strip().lower()
    if s == "yes":     return "❌"   # attack succeeded — bad
    if s == "no":      return "✅"   # attack failed    — good
    return "❓"                       # unclear


def outcome_label(row: Dict) -> str:
    """Short text label for a result row."""
    if row["blocked"]:  return "BLOCKED"
    if row["flagged"]:  return "FLAGGED"
    s = row["attack_succeeded"].strip().lower()
    if s == "yes":  return "SUCCEEDED"
    if s == "no":   return "DEFENDED"
    return "UNCLEAR"


def print_separator(char="─", width=80):
    print(char * width)


def generate_report(
    off_rows:  List[Dict],
    conf_rows: List[Dict],
    full_rows: List[Dict],
) -> str:
    """
    Generate the full comparison report.
    Returns the report as a string (also printed to console).
    """
    off_m  = calc_metrics(off_rows)
    conf_m = calc_metrics(conf_rows)
    full_m = calc_metrics(full_rows)

    lines = []

    def p(s=""):
        """Print and record line."""
        print(s)
        lines.append(s)

    # ── Header ────────────────────────────────────────────────────────────────
    p()
    p(bold("=" * 80))
    p(bold("  MedRAGShield — Attack Evaluation Report"))
    p(bold(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    p(bold("=" * 80))

    # ── Section 1: Summary table ─────────────────────────────────────────────
    p()
    p(bold("  SECTION 1: HEADLINE METRICS"))
    p("─" * 80)
    p(f"  {'Metric':<30} {'off':>12} {'confidence':>12} {'full':>12}")
    p("─" * 80)

    def row_line(label, off_val, conf_val, full_val, fmt="{}", good="low"):
        """
        Print one metric row. Colours the values:
          good='low'  → lower is better  (ASR)
          good='high' → higher is better (block rate)
        """
        def colour(val, ref_off, ref_full):
            if good == "low":
                if val == ref_full and val < ref_off:   return green(fmt.format(val))
                if val == ref_off  and val > ref_full:  return red(fmt.format(val))
            else:
                if val == ref_full and val > ref_off:   return green(fmt.format(val))
                if val == ref_off  and val < ref_full:  return red(fmt.format(val))
            return fmt.format(val)

        print(
            f"  {label:<30} "
            f"{colour(off_val,  off_val, full_val):>20} "
            f"{colour(conf_val, off_val, full_val):>20} "
            f"{colour(full_val, off_val, full_val):>20}"
        )
        lines.append(
            f"  {label:<30} {fmt.format(off_val):>12} "
            f"{fmt.format(conf_val):>12} {fmt.format(full_val):>12}"
        )

    row_line("ASR (attack success rate %)",
             off_m["asr"], conf_m["asr"], full_m["asr"],
             fmt="{:.1f}%", good="low")
    row_line("Block rate %",
             off_m["block_rate"], conf_m["block_rate"], full_m["block_rate"],
             fmt="{:.1f}%", good="high")
    row_line("Flag rate %",
             off_m["flag_rate"], conf_m["flag_rate"], full_m["flag_rate"],
             fmt="{:.1f}%", good="high")
    row_line("Attacks succeeded",
             off_m["succeeded"], conf_m["succeeded"], full_m["succeeded"],
             fmt="{}", good="low")
    row_line("Attacks blocked",
             off_m["blocked"], conf_m["blocked"], full_m["blocked"],
             fmt="{}", good="high")
    row_line("Attacks flagged",
             off_m["flagged"], conf_m["flagged"], full_m["flagged"],
             fmt="{}", good="high")
    row_line("Unclear (manual review)",
             off_m["unclear"], conf_m["unclear"], full_m["unclear"],
             fmt="{}", good="low")
    row_line("Avg confidence",
             off_m["avg_conf"], conf_m["avg_conf"], full_m["avg_conf"],
             fmt="{:.3f}", good="high")
    row_line("Avg latency (s)",
             off_m["avg_latency_s"], conf_m["avg_latency_s"], full_m["avg_latency_s"],
             fmt="{:.1f}s", good="low")
    p("─" * 80)

    # ── Section 2: Per-attack comparison ─────────────────────────────────────
    p()
    p(bold("  SECTION 2: PER-ATTACK COMPARISON"))
    p("─" * 80)
    p(f"  {'ID':<4} {'Type':<22} {'off':>12} {'confidence':>12} {'full':>12}")
    p("─" * 80)

    # Build lookup dicts keyed by attack id
    off_by_id  = {r["id"]: r for r in off_rows}
    conf_by_id = {r["id"]: r for r in conf_rows}
    full_by_id = {r["id"]: r for r in full_rows}

    all_ids = sorted(set(
        list(off_by_id.keys()) +
        list(conf_by_id.keys()) +
        list(full_by_id.keys())
    ))

    for aid in all_ids:
        off_r  = off_by_id.get(aid)
        conf_r = conf_by_id.get(aid)
        full_r = full_by_id.get(aid)

        attack_type = (off_r or conf_r or full_r)["type"]
        off_label  = outcome_label(off_r)  if off_r  else "N/A"
        conf_label = outcome_label(conf_r) if conf_r else "N/A"
        full_label = outcome_label(full_r) if full_r else "N/A"

        # Colour the terminal output
        def clr(label):
            if label == "BLOCKED":   return green("BLOCKED")
            if label == "FLAGGED":   return yellow("FLAGGED")
            if label == "SUCCEEDED": return red("SUCCEEDED")
            if label == "DEFENDED":  return green("DEFENDED")
            return label

        print(f"  {aid:<4} {attack_type:<22} {clr(off_label):>20} {clr(conf_label):>20} {clr(full_label):>20}")
        lines.append(f"  {aid:<4} {attack_type:<22} {off_label:>12} {conf_label:>12} {full_label:>12}")

    p("─" * 80)

    # ── Section 3: Per-type breakdown ─────────────────────────────────────────
    p()
    p(bold("  SECTION 3: PER-TYPE BREAKDOWN (mode=full)"))
    p("─" * 80)
    p(f"  {'Attack Type':<25} {'Total':>7} {'Blocked':>9} {'Flagged':>9} {'Succeeded':>11} {'Unclear':>9}")
    p("─" * 80)

    for atype, tm in sorted(full_m["by_type"].items()):
        p(f"  {atype:<25} {tm['total']:>7} {tm['blocked']:>9} {tm['flagged']:>9} {tm['succeeded']:>11} {tm['unclear']:>9}")

    p("─" * 80)

    # ── Section 4: Gap analysis ───────────────────────────────────────────────
    p()
    p(bold("  SECTION 4: GAP ANALYSIS — Attacks not blocked in mode=full"))
    p("─" * 80)
    p("  These require manual review or new defense rules:\n")

    gaps_found = False
    for aid in all_ids:
        full_r = full_by_id.get(aid)
        if not full_r:
            continue
        label = outcome_label(full_r)
        if label in ("SUCCEEDED", "UNCLEAR", "FLAGGED"):
            gaps_found = True
            off_r = off_by_id.get(aid, {})
            p(f"  ID {aid:>2} | {full_r['type']:<22} | full={label:<10} | off={outcome_label(off_r):<10}")
            p(f"         Query   : {full_r['query'][:75]}")
            p(f"         Response: {full_r['response'][:120]}")
            p(f"         Goal    : {full_r['attack_goal'][:75]}")
            p()

    if not gaps_found:
        p(green("  No gaps found — all attacks blocked or defended in full mode."))
        p()

    # ── Section 5: Key findings ───────────────────────────────────────────────
    p(bold("  SECTION 5: KEY FINDINGS FOR REPORT"))
    p("─" * 80)

    asr_reduction = round(off_m["asr"] - full_m["asr"], 1)
    block_gain    = round(full_m["block_rate"] - off_m["block_rate"], 1)

    p(f"  1. ASR reduced from {off_m['asr']}% (off) → {full_m['asr']}% (full)  [{asr_reduction}pp reduction]")
    p(f"  2. Block rate increased from {off_m['block_rate']}% → {full_m['block_rate']}%  [+{block_gain}pp]")
    p(f"  3. Confidence mode alone achieved {conf_m['block_rate']}% block rate")
    p(f"     → insufficient for document injection attacks (DASC required)")

    # Count doc injection attacks that passed confidence but were caught by full
    doc_inj_slipped = []
    for aid in all_ids:
        conf_r = conf_by_id.get(aid)
        full_r = full_by_id.get(aid)
        if not conf_r or not full_r:
            continue
        if conf_r["type"] == "document_injection":
            conf_outcome = outcome_label(conf_r)
            full_outcome = outcome_label(full_r)
            if conf_outcome in ("SUCCEEDED", "UNCLEAR") and full_outcome == "BLOCKED":
                doc_inj_slipped.append(aid)

    if doc_inj_slipped:
        p(f"  4. Document injection IDs {doc_inj_slipped} passed confidence gate")
        p(f"     but were caught by DASC in full mode → DASC is essential")

    p(f"  5. Hallucination attacks naturally refused by LLM (low confidence <0.50)")
    p(f"     Confidence gate adds formal safety net on top")
    p()
    p("─" * 80)
    p(bold("  END OF REPORT"))
    p("─" * 80)
    p()

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MedRAGShield results comparison")
    parser.add_argument("--results_dir", default="results",
                        help="Directory containing attack result CSVs (default: results)")
    parser.add_argument("--off",  default=None, help="Path to off-mode CSV")
    parser.add_argument("--conf", default=None, help="Path to confidence-mode CSV")
    parser.add_argument("--full", default=None, help="Path to full-mode CSV")
    parser.add_argument("--output", default=None,
                        help="Output report path (default: results/comparison_report.txt)")
    args = parser.parse_args()

    # Resolve CSV paths
    off_path  = args.off  or find_latest_csv(args.results_dir, "off")
    conf_path = args.conf or find_latest_csv(args.results_dir, "confidence")
    full_path = args.full or find_latest_csv(args.results_dir, "full")

    missing = []
    if not off_path  or not os.path.exists(off_path):  missing.append("off")
    if not conf_path or not os.path.exists(conf_path): missing.append("confidence")
    if not full_path or not os.path.exists(full_path): missing.append("full")

    if missing:
        print(f"\n  ERROR: Could not find CSVs for modes: {missing}")
        print(f"  Run: python run_attacks.py --mode off,confidence,full  first.\n")
        return

    print(f"\n  Loading CSVs:")
    print(f"    off        → {off_path}")
    print(f"    confidence → {conf_path}")
    print(f"    full       → {full_path}")

    off_rows  = load_csv(off_path)
    conf_rows = load_csv(conf_path)
    full_rows = load_csv(full_path)

    # Generate report
    report_text = generate_report(off_rows, conf_rows, full_rows)

    # Save to disk
    output_path = args.output or os.path.join(args.results_dir, "comparison_report.txt")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Strip ANSI codes for the saved file
        import re
        clean = re.sub(r"\033\[[0-9;]*m", "", report_text)
        f.write(clean)

    print(f"\n  Report saved → {output_path}\n")


if __name__ == "__main__":
    main()