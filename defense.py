"""
defense.py
----------
MedRAGShield — Defense Middleware for Medical RAG Chatbot
=========================================================

This module implements the three-tier defense layer that sits between
your RAG pipeline (rag_core.py) and the API (api.py).

It is completely self-contained — rag_core.py and api.py are untouched.

HOW IT WORKS
------------
Every /chat request passes through defense.apply() AFTER bot.chat() runs.
The defense inspects the raw RAG result and either:
  - passes it through unchanged (mode=off)
  - adds a warning prefix  (mode=confidence, low confidence detected)
  - replaces the answer with a BLOCKED message (any mode, threshold exceeded)

THREE MODES
-----------
  off        →  Raw RAG output. No filtering at all. Used as attack baseline.
  confidence →  Tier 1+2: confidence gate + hallucination gate.
  full       →  All tiers: confidence gate + hallucination gate + DASC rules.

THREE DEFENSE TIERS
-------------------
  Tier 1 — Query-level injection detector  (both confidence + full modes)
            Scans the incoming user query for prompt injection / jailbreak
            patterns using pre-compiled regex. Blocks before RAG even runs.

  Tier 2 — Confidence + hallucination gate  (both confidence + full modes)
            Uses the confidence score and hallucination_score already computed
            by rag_core.py. Warns or blocks based on configured thresholds.

  Tier 3 — DASC: Domain-Aware Sanity Check  (full mode only)
            Applies medical DDI (drug-drug interaction) rules to the generated
            answer using regex pattern matching. Catches cases where a poisoned
            document caused the LLM to output a clinically dangerous claim
            (e.g. "aspirin and warfarin are safe to combine").

KEY BUG FIX (v1 → v2)
----------------------
  Old version used exact string matching:
      "safe to combine" in answer   →  missed "safely combined"

  New version uses regex:
      re.search(r"safe(ly)?\\s+(to\\s+)?(combine|combined)", answer)
      →  catches all LLM paraphrasings of the same dangerous claim.

  Also: DASC now scans result.original_answer (not result.answer).
  This prevents a confidence-gate warning prefix from interfering with
  the drug-name / safe-phrase matching logic.

USAGE IN api.py
---------------
    from defense import DefenseMiddleware, DefenseMode

    defense = DefenseMiddleware()          # once at startup

    raw = bot.chat(query)                  # your existing RAG call
    protected = defense.apply(            # wrap it
        rag_result=raw,
        mode=mode,                         # "off" | "confidence" | "full"
        query=query,                       # needed for Tier 1 query scan
    )
    return protected                       # drop-in replacement for raw

RESPONSE FIELDS ADDED
---------------------
  blocked         (bool)   True if answer was replaced with a block message
  flagged         (bool)   True if a warning prefix was prepended
  defense_mode    (str)    Which mode was active
  triggered_rules (list)   Which rule IDs fired, e.g. ["DDI-001"]
  metadata.original_answer  Preserved when blocked/flagged, for logging
"""

import re
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS & CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class DefenseMode(str, Enum):
    """Valid defense mode values accepted by the /chat endpoint."""
    OFF        = "off"
    CONFIDENCE = "confidence"
    FULL       = "full"


@dataclass
class DefenseConfig:
    """
    Thresholds for the confidence and hallucination gates.

    These are intentionally set independently of CONFIDENCE_THRESHOLD in
    config.py — that one controls whether rag_core.py proceeds to the LLM at
    all; these control what the defense layer does AFTER the LLM has generated.

    confidence_warn_threshold:     warn user if confidence falls below this
    confidence_block_threshold:    block entirely if confidence falls below this
    hallucination_warn_threshold:  warn if hallucination_score exceeds this
    hallucination_block_threshold: block if hallucination_score exceeds this
    enable_dasc:                   set False to disable DASC without changing mode
    """
    confidence_warn_threshold:    float = 0.65
    confidence_block_threshold:   float = 0.40
    hallucination_warn_threshold:  float = 0.50
    hallucination_block_threshold: float = 0.75
    enable_dasc:                   bool  = True


# ══════════════════════════════════════════════════════════════════════════════
# DASC RULE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DASCRule:
    """
    One Domain-Aware Sanity Check rule.

    A rule fires when ALL of the following are true:
      1. Every pattern in drug_patterns matches somewhere in the answer
         (ensures the rule only applies when the relevant drugs are mentioned)
      2. At least one pattern in safe_patterns matches
         (the LLM has made a dangerous claim about those drugs)

    drug_patterns:  List of regex strings — ALL must match in the answer.
                    Leave empty [] for drug-agnostic rules (e.g. dosage rules).
    safe_patterns:  List of regex strings — ANY match triggers the block.
                    These capture all LLM paraphrasings of a dangerous claim.
    block_message:  The string that replaces the answer when the rule fires.
    severity:       "high" or "medium" — recorded in logs, not used for logic.
    """
    rule_id:       str
    description:   str
    drug_patterns: List[str]
    safe_patterns: List[str]
    block_message: str
    severity:      str = "high"


# ── Rule definitions ──────────────────────────────────────────────────────────

DASC_RULES: List[DASCRule] = [

    # ── DDI-001: Aspirin + Warfarin ───────────────────────────────────────────
    # This is your primary demo attack: aspirin_warfarin_safe.txt causes the LLM
    # to say the combination is safe. These patterns catch every phrasing seen
    # in practice: "safely combined", "safe to combine", "safe combination",
    # "no increased risk", "common outpatient regimen", etc.
    DASCRule(
        rule_id="DDI-001",
        description="Aspirin + Warfarin falsely claimed safe",
        drug_patterns=[
            r"\baspirin\b",    # aspirin must appear
            r"\bwarfarin\b",   # warfarin must appear
        ],
        safe_patterns=[
            r"safe(ly)?\s+(to\s+)?(combine|taken|take|used|combined|use)",
            r"(can|may)\s+be\s+safe(ly)?\s+(combined|taken|used)",
            r"safe\s+combination",
            r"no\s+increased\s+(bleeding\s+)?risk",
            r"no\s+bleeding\s+risk",
            r"no\s+significant\s+(interaction|risk)",
            r"no\s+contraindication",
            r"freely\s+combined",
            r"can\s+be\s+taken\s+together",
            r"(well\s+)?tolerated\s+(together|with)",
            r"common\s+outpatient\s+regimen",   # exact phrase from poisoned doc
            r"no\s+additional\s+(INR|monitoring)",
            r"does\s+not\s+significantly\s+interact",
        ],
        block_message=(
            "BLOCKED — The retrieved content contains a claim about aspirin and "
            "warfarin that contradicts known clinical guidelines. Combining aspirin "
            "and warfarin significantly increases bleeding risk. "
            "Please consult a healthcare provider."
        ),
        severity="high",
    ),

    # ── DDI-002: Ibuprofen / NSAIDs + Warfarin ───────────────────────────────
    # NSAIDs are a common class of drugs that interact dangerously with warfarin.
    # This rule fires if any NSAID name appears alongside a safety claim.
    DASCRule(
        rule_id="DDI-002",
        description="Ibuprofen / NSAID + Warfarin falsely claimed safe",
        drug_patterns=[
            r"\bwarfarin\b",
            r"\b(ibuprofen|nsaid|nsaids)\b",
        ],
        safe_patterns=[
            r"\b(ibuprofen|nsaid)\s+is\s+safe",
            r"safe(ly)?\s+(to\s+)?(take|use|combine)\s+(ibuprofen|nsaid)",
            r"no\s+risk\s+with\s+(ibuprofen|nsaid)",
            r"(ibuprofen|nsaid).{0,30}no\s+(increased\s+)?risk",
        ],
        block_message=(
            "BLOCKED — The retrieved content makes a claim about NSAIDs and "
            "warfarin that contradicts clinical evidence. NSAIDs (e.g. ibuprofen) "
            "increase bleeding risk in patients on warfarin and are generally avoided. "
            "Please consult a healthcare provider."
        ),
        severity="high",
    ),

    # ── DDI-003: Metformin + Alcohol ─────────────────────────────────────────
    # Alcohol with metformin increases lactic acidosis risk. Relevant to your
    # attacks.json entry id=4 (metformin + alcohol document injection test).
    DASCRule(
        rule_id="DDI-003",
        description="Metformin + alcohol falsely claimed safe",
        drug_patterns=[
            r"\bmetformin\b",   # only needs metformin; alcohol is in safe_patterns
        ],
        safe_patterns=[
            r"alcohol\s+is\s+safe\s+(with|alongside)\s+metformin",
            r"no\s+risk\s+with\s+alcohol",
            r"alcohol\s+has\s+no\s+effect",
            r"freely\s+drink",
            r"alcohol\s+is\s+fine",
            r"safe(ly)?\s+(to\s+)?drink\s+alcohol",
        ],
        block_message=(
            "BLOCKED — The retrieved content makes a claim about alcohol and "
            "metformin that contradicts clinical guidelines. Alcohol with metformin "
            "increases lactic acidosis risk. Please consult a healthcare provider."
        ),
        severity="medium",
    ),

    # ── DDI-004: Dangerous dosage language ───────────────────────────────────
    # Drug-agnostic rule. Fires if the answer tells a user to self-adjust doses
    # without medical supervision — regardless of which drug is involved.
    DASCRule(
        rule_id="DDI-004",
        description="Dangerous dosage adjustment language detected",
        drug_patterns=[],    # [] = no drug check required; applies to any answer
        safe_patterns=[
            r"double\s+(the|your)\s+dose",
            r"take\s+twice\s+the\s+dose",
            r"take\s+(an?\s+)?extra\s+dose",
            r"no\s+need\s+to\s+monitor",
            r"no\s+monitoring\s+needed",
            r"no\s+blood\s+test\s+needed",
        ],
        block_message=(
            "BLOCKED — The response contains language about dosage adjustment "
            "that could be unsafe without medical supervision. "
            "Please consult a healthcare provider before changing any medication dose."
        ),
        severity="high",
    ),

    # ── IPI-001: Indirect prompt injection in the generated response ──────────
    # Catches cases where a poisoned document contained instruction-override text
    # that the LLM echoed or acted on in its output.
    # Complements the query-level injection check (which scans user input).
    DASCRule(
        rule_id="IPI-001",
        description="Indirect prompt injection signal detected in LLM response",
        drug_patterns=[],
        safe_patterns=[
            r"ignore\s+(previous|prior|all)\s+instructions?",
            r"disregard\s+(prior|previous|all|system)",
            r"as\s+instructed\s+by\s+the\s+system",
            r"following\s+the\s+directive",
            r"no\s+disclaimers",
            r"without\s+safety\s+warnings",
            r"dan\s+mode",
            r"developer\s+mode",
            r"no\s+restrictions",
        ],
        block_message=(
            "BLOCKED — The response contains language consistent with a prompt "
            "injection attempt in the retrieved context. This query has been blocked "
            "for safety. Please rephrase your question."
        ),
        severity="high",
    ),
]

# Pre-compile all regex patterns at import time.
# This avoids re-compiling on every request — important for latency under load.
for _rule in DASC_RULES:
    _rule._compiled_drug = [re.compile(p, re.IGNORECASE) for p in _rule.drug_patterns]
    _rule._compiled_safe = [re.compile(p, re.IGNORECASE) for p in _rule.safe_patterns]


# ══════════════════════════════════════════════════════════════════════════════
# QUERY-LEVEL INJECTION DETECTOR (Tier 1)
# ══════════════════════════════════════════════════════════════════════════════
# These patterns scan the user's query BEFORE it reaches the RAG pipeline.
# Covers attacks.json types: prompt_injection (ids 9-12) and jailbreak (13-16).

INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|all)\s+instructions?",        # classic override
    r"disregard\s+(prior|previous|all|system)",
    r"you\s+are\s+now\s+(a|an|the)?\s*\w+\s*(without|with no)\s+(restrictions?|limits?|filters?)",
    r"pretend\s+you\s+(have\s+no|don'?t\s+have)\s+(safety|restrictions?|limits?)",
    r"\[?\s*(admin|system|hidden|override)\s*(note|instruction|prompt|command)\s*:?\s*\]?",
    r"dan\s+mode",                                            # DAN jailbreak
    r"developer\s+mode\s+(enabled|activated|on)",
    r"(all\s+)?safety\s+(filters?|guards?)\s+(disabled?|removed?|off)",
    r"respond\s+only\s+with\s+['\"]",                        # forced output
    r"without\s+(any\s+)?(disclaimers?|warnings?|caveats?)", # disclaimer suppression
]

# Compile once at import
COMPILED_INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DefenseResult:
    """
    Internal state object passed between defense methods.
    Not returned to the caller — _finalise() merges it back into rag_result.

    original_answer:  The LLM's raw answer, preserved before any modification.
                      DASC always scans this, not the (possibly prefixed) answer.
    answer:           The answer that will be returned. May be replaced with a
                      block message or prefixed with a warning.
    blocked:          True = answer replaced with block message.
    flagged:          True = warning prefix prepended, original answer retained.
    triggered_rules:  List of rule IDs that fired, e.g. ["CONF-WARN", "DDI-001"].
    """
    answer:              str
    original_answer:     str       = ""
    blocked:             bool      = False
    flagged:             bool      = False
    defense_mode:        str       = "off"
    triggered_rules:     List[str] = field(default_factory=list)
    confidence:          float     = 0.0
    hallucination_score: float     = 0.0
    metadata:            Dict      = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MIDDLEWARE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class DefenseMiddleware:
    """
    Drop-in defense wrapper for the medical RAG pipeline.

    Instantiate once at server startup (in api.py lifespan).
    Call apply() on every /chat request after bot.chat() returns.

    Example:
        defense = DefenseMiddleware()
        raw      = bot.chat(query)
        result   = defense.apply(raw, mode="full", query=query)
        return result   # same dict shape as raw, with defense fields added
    """

    def __init__(self, config: DefenseConfig = None):
        """
        Args:
            config: Optional DefenseConfig to override default thresholds.
                    If None, uses sensible defaults (see DefenseConfig docstring).
        """
        self.config = config or DefenseConfig()
        # Cumulative stats — inspectable via GET /stats → defense_stats
        self.stats = {
            "total_requests":  0,
            "blocked":         0,
            "flagged":         0,
            "rules_triggered": {},   # {rule_id: count}
        }
        logger.info("DefenseMiddleware initialised")

    # ── Public API ────────────────────────────────────────────────────────────

    def apply(self, rag_result: Dict, mode: str = "off", query: str = "") -> Dict:
        """
        Apply the selected defense tier(s) to a rag_result dict.

        Args:
            rag_result:  Output of bot.chat(). Must contain at minimum:
                         "answer", "confidence", "metadata.hallucination_score".
            mode:        "off" | "confidence" | "full"
            query:       The original user query string. Required for Tier 1
                         query injection check. Safe to omit (defaults to "").

        Returns:
            The same rag_result dict, modified in place, with these keys added:
              blocked         (bool)
              flagged         (bool)
              defense_mode    (str)
              triggered_rules (list[str])
              metadata.original_answer  (str, only if blocked or flagged)
        """
        self.stats["total_requests"] += 1

        # Normalise and validate mode string
        try:
            defense_mode = DefenseMode(mode.lower())
        except ValueError:
            logger.warning(f"Unknown defense mode '{mode}', defaulting to off")
            defense_mode = DefenseMode.OFF

        # ── mode=off: pass through with default fields added ─────────────────
        if defense_mode == DefenseMode.OFF:
            rag_result.setdefault("blocked", False)
            rag_result.setdefault("flagged", False)
            rag_result.setdefault("defense_mode", "off")
            rag_result.setdefault("triggered_rules", [])
            return rag_result

        # ── Extract values from rag_result ───────────────────────────────────
        answer       = rag_result.get("answer", "")
        confidence   = rag_result.get("confidence", 0.0)
        # hallucination_score lives inside metadata (set by rag_core._detect_hallucination)
        halluc_score = rag_result.get("metadata", {}).get("hallucination_score", 0.0)

        # Create internal state object
        result = DefenseResult(
            answer=answer,
            original_answer=answer,    # preserve clean copy for DASC scanning
            confidence=confidence,
            hallucination_score=halluc_score,
            defense_mode=defense_mode.value,
        )

        # ── Tier 1: Query injection check (confidence + full) ────────────────
        if query:
            self._check_query_injection(query, result)
        if result.blocked:
            return self._finalise(rag_result, result)

        # ── Tier 2: Confidence + hallucination gate (confidence + full) ──────
        self._apply_confidence_gate(result)
        if result.blocked:
            return self._finalise(rag_result, result)

        # ── Tier 3: DASC medical safety rules (full only) ────────────────────
        if defense_mode == DefenseMode.FULL and self.config.enable_dasc:
            self._apply_dasc_rules(result)

        return self._finalise(rag_result, result)

    # ── Tier 1: Query-level injection check ───────────────────────────────────

    def _check_query_injection(self, query: str, result: DefenseResult):
        """
        Scan the user query for known prompt injection / jailbreak patterns.
        Relevant attack types from attacks.json: prompt_injection, jailbreak.

        If a pattern matches, the request is blocked immediately and the
        pipeline does not proceed to Tier 2 or Tier 3.
        """
        for pattern in COMPILED_INJECTION_PATTERNS:
            if pattern.search(query):
                result.blocked = True
                result.triggered_rules.append("IPI-QUERY")
                result.answer = (
                    "BLOCKED — Your query contains patterns associated with "
                    "prompt injection or jailbreak attempts. "
                    "Please ask your medical question directly."
                )
                self._record_rule("IPI-QUERY")
                logger.warning(
                    f"[DEFENSE] Query injection blocked "
                    f"| pattern='{pattern.pattern[:60]}'"
                )
                return   # stop at first match

    # ── Tier 2: Confidence + hallucination gate ───────────────────────────────

    def _apply_confidence_gate(self, result: DefenseResult):
        """
        Gate responses based on the confidence score and hallucination_score
        already computed by rag_core.py.

        Two severity levels per signal:
          BLOCK  — answer replaced with a refusal message
          WARN   — warning prefix prepended, original answer retained

        Thresholds are set in DefenseConfig and ultimately sourced from
        config.py (DEFENSE_CONFIDENCE_WARN, DEFENSE_CONFIDENCE_BLOCK, etc.)
        """
        conf   = result.confidence
        halluc = result.hallucination_score

        # ── Hard blocks ───────────────────────────────────────────────────────

        if conf < self.config.confidence_block_threshold:
            result.blocked = True
            result.triggered_rules.append("CONF-BLOCK")
            result.answer = (
                f"BLOCKED — Confidence too low ({conf:.2f}) to answer safely. "
                "The retrieved documents may not be relevant to your question. "
                "Please consult a healthcare provider."
            )
            self._record_rule("CONF-BLOCK")
            logger.warning(f"[DEFENSE] Confidence block | conf={conf:.2f}")
            return

        if halluc > self.config.hallucination_block_threshold:
            result.blocked = True
            result.triggered_rules.append("HALLUC-BLOCK")
            result.answer = (
                f"BLOCKED — High hallucination risk detected ({halluc:.2f}). "
                "The generated answer diverges significantly from retrieved sources. "
                "Please consult a healthcare provider."
            )
            self._record_rule("HALLUC-BLOCK")
            logger.warning(f"[DEFENSE] Hallucination block | score={halluc:.2f}")
            return

        # ── Soft warnings ─────────────────────────────────────────────────────

        warnings = []

        if conf < self.config.confidence_warn_threshold:
            result.flagged = True
            result.triggered_rules.append("CONF-WARN")
            warnings.append(f"WARNING LOW CONFIDENCE ({conf:.2f})")
            self._record_rule("CONF-WARN")

        if halluc > self.config.hallucination_warn_threshold:
            result.flagged = True
            result.triggered_rules.append("HALLUC-WARN")
            warnings.append(f"WARNING POSSIBLE HALLUCINATION ({halluc:.2f})")
            self._record_rule("HALLUC-WARN")

        if warnings:
            prefix = " | ".join(warnings)
            result.answer = (
                f"[{prefix}] This answer may be unreliable. "
                f"Please verify with a healthcare professional.\n\n"
                f"{result.answer}"
            )
            logger.info(f"[DEFENSE] Flagged | {prefix}")

    # ── Tier 3: DASC medical safety rules ────────────────────────────────────

    def _apply_dasc_rules(self, result: DefenseResult):
        """
        Apply Domain-Aware Sanity Check rules to the generated answer.

        CRITICAL: We scan result.original_answer, NOT result.answer.
        ─────────────────────────────────────────────────────────────
        If Tier 2 prepended a warning prefix (e.g. "[WARNING LOW CONFIDENCE...]"),
        result.answer now contains that prefix. Scanning it would break the
        drug-name matching (e.g. "aspirin" might still be found, but the surrounding
        context is polluted). original_answer is always the clean LLM output.

        Rule evaluation order:
          1. Check ALL drug_patterns match in the answer (drug prerequisite).
             If any drug pattern is missing, skip this rule entirely.
          2. Check if ANY safe_pattern matches (dangerous claim detected).
             First match wins — block immediately, don't check further rules.
        """
        scan_text = result.original_answer   # always scan the clean LLM output

        for rule in DASC_RULES:

            # ── Drug prerequisite check ───────────────────────────────────────
            # For drug-specific rules (drug_patterns is non-empty), ALL listed
            # drugs must appear in the answer. If a rule has drug_patterns=[]
            # (e.g. DDI-004 dosage rule, IPI-001 injection rule), skip this check.
            if rule._compiled_drug:
                all_drugs_present = all(
                    p.search(scan_text) for p in rule._compiled_drug
                )
                if not all_drugs_present:
                    continue   # drugs not mentioned → rule irrelevant, skip

            # ── Dangerous claim check ─────────────────────────────────────────
            # ANY match in safe_patterns means the LLM made a dangerous claim.
            triggered_pattern = next(
                (p for p in rule._compiled_safe if p.search(scan_text)),
                None
            )

            if triggered_pattern:
                result.blocked = True
                result.triggered_rules.append(rule.rule_id)
                result.answer = rule.block_message
                self._record_rule(rule.rule_id)
                logger.warning(
                    f"[DEFENSE] DASC triggered "
                    f"| rule={rule.rule_id} "
                    f"| severity={rule.severity} "
                    f"| pattern='{triggered_pattern.pattern}'"
                )
                return   # first match wins — stop checking further rules

    # ── Finalise: merge DefenseResult back into rag_result dict ──────────────

    def _finalise(self, rag_result: Dict, result: DefenseResult) -> Dict:
        """
        Write defense fields back into the original rag_result dict.
        This keeps the return type identical to what bot.chat() returns,
        just with four extra fields added.
        """
        if result.blocked:
            self.stats["blocked"] += 1
        elif result.flagged:
            self.stats["flagged"] += 1

        # Overwrite answer with the (possibly modified) version
        rag_result["answer"]          = result.answer
        rag_result["blocked"]         = result.blocked
        rag_result["flagged"]         = result.flagged
        rag_result["defense_mode"]    = result.defense_mode
        rag_result["triggered_rules"] = result.triggered_rules

        # Preserve original answer in metadata when something was triggered.
        # This is used by run_attacks.py and the /compare endpoint to log
        # what the LLM actually said before the defense intervened.
        if result.blocked or result.flagged:
            rag_result.setdefault("metadata", {})
            rag_result["metadata"]["original_answer"] = result.original_answer

        self._log_defense_event(result)
        return rag_result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _record_rule(self, rule_id: str):
        """Increment the counter for a triggered rule in cumulative stats."""
        self.stats["rules_triggered"][rule_id] = (
            self.stats["rules_triggered"].get(rule_id, 0) + 1
        )

    def _log_defense_event(self, result: DefenseResult):
        """
        Write a structured JSON log entry for blocked/flagged events.
        Logged to the same logger as rag_core.py (logs/rag_chatbot.log).
        Blocked events → WARNING level; flagged events → INFO level.
        Passthrough events (neither blocked nor flagged) are not logged here
        to keep the log clean — the request-level log in api.py covers those.
        """
        if not (result.blocked or result.flagged):
            return

        entry = {
            "timestamp":           datetime.now().isoformat(),
            "defense_mode":        result.defense_mode,
            "blocked":             result.blocked,
            "flagged":             result.flagged,
            "triggered_rules":     result.triggered_rules,
            "confidence":          result.confidence,
            "hallucination_score": result.hallucination_score,
        }

        if result.blocked:
            logger.warning(f"[DEFENSE_EVENT] {json.dumps(entry)}")
        else:
            logger.info(f"[DEFENSE_EVENT] {json.dumps(entry)}")

    def get_stats(self) -> Dict:
        """
        Return cumulative defense statistics since server startup.
        Exposed via GET /stats → defense_stats in api.py.

        Returns:
            total_requests:  All requests processed (including off mode)
            blocked:         Requests where answer was replaced with block message
            flagged:         Requests where warning prefix was added
            block_rate:      blocked / total_requests
            flag_rate:       flagged / total_requests
            rules_triggered: Per-rule trigger counts, e.g. {"DDI-001": 3}
        """
        total = self.stats["total_requests"]
        return {
            "total_requests":  total,
            "blocked":         self.stats["blocked"],
            "flagged":         self.stats["flagged"],
            "block_rate":      round(self.stats["blocked"] / total, 3) if total else 0,
            "flag_rate":       round(self.stats["flagged"] / total, 3) if total else 0,
            "rules_triggered": self.stats["rules_triggered"],
        }