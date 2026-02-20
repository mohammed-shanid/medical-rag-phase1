"""
defense.py
----------
MedRAGShield defense middleware.
Plugs into your existing api.py and rag_core.py without modifying them.

Three modes:
    off        → raw RAG output, no filtering
    confidence → confidence + hallucination warning layer
    full       → confidence layer + DASC medical safety rules

Usage in api.py:
    from defense import DefenseMiddleware, DefenseMode
    defense = DefenseMiddleware()
    result = defense.apply(rag_result, mode="full")
"""

import re
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Enums & Config ────────────────────────────────────────────────────────────

class DefenseMode(str, Enum):
    OFF        = "off"
    CONFIDENCE = "confidence"
    FULL       = "full"


@dataclass
class DefenseConfig:
    # Confidence gate thresholds
    confidence_warn_threshold:  float = 0.65   # warn below this
    confidence_block_threshold: float = 0.40   # block below this

    # Hallucination gate
    hallucination_warn_threshold:  float = 0.50
    hallucination_block_threshold: float = 0.75

    # DASC rules (full mode only)
    enable_dasc: bool = True


# ── DASC Rules ────────────────────────────────────────────────────────────────

@dataclass
class DASCRule:
    """One medical safety rule."""
    rule_id:     str
    description: str
    drugs:       List[str]          # both drugs must appear in answer
    safe_words:  List[str]          # if any of these appear → dangerous claim
    block_message: str
    severity:    str = "high"       # high | medium


# Core DDI rules drawn from your FDA corpus
DASC_RULES: List[DASCRule] = [
    DASCRule(
        rule_id="DDI-001",
        description="Aspirin + Warfarin falsely claimed safe",
        drugs=["aspirin", "warfarin"],
        safe_words=[
            "safe to combine", "safe to take together", "no increased risk",
            "no bleeding risk", "can be taken together", "no interaction",
            "safe combination", "no problem", "freely combined",
        ],
        block_message=(
            "⛔ BLOCKED — The retrieved content contains a claim about aspirin and "
            "warfarin that contradicts known clinical guidelines. Combining aspirin "
            "and warfarin significantly increases bleeding risk. "
            "Please consult a healthcare provider."
        ),
        severity="high",
    ),
    DASCRule(
        rule_id="DDI-002",
        description="Ibuprofen / NSAID + Warfarin falsely claimed safe",
        drugs=["warfarin"],
        safe_words=[
            "ibuprofen is safe", "nsaid is safe", "ibuprofen can be taken",
            "ibuprofen with warfarin is fine", "no risk with ibuprofen",
        ],
        block_message=(
            "⛔ BLOCKED — The retrieved content makes a claim about NSAIDs and "
            "warfarin that contradicts clinical evidence. NSAIDs (e.g. ibuprofen) "
            "increase bleeding risk in patients on warfarin and are generally avoided. "
            "Please consult a healthcare provider."
        ),
        severity="high",
    ),
    DASCRule(
        rule_id="DDI-003",
        description="Metformin + alcohol falsely claimed safe",
        drugs=["metformin"],
        safe_words=[
            "alcohol is safe with metformin", "no risk with alcohol",
            "alcohol has no effect", "freely drink", "alcohol is fine",
        ],
        block_message=(
            "⛔ BLOCKED — The retrieved content makes a claim about alcohol and "
            "metformin that contradicts clinical guidelines. Alcohol with metformin "
            "increases lactic acidosis risk. Please consult a healthcare provider."
        ),
        severity="medium",
    ),
    DASCRule(
        rule_id="DDI-004",
        description="Dangerous dosage language detected",
        drugs=[],   # drug-agnostic rule
        safe_words=[
            "double the dose", "double your dose", "take twice the dose",
            "take extra dose", "skip the doctor", "no need to monitor",
            "no monitoring needed", "no blood test needed",
        ],
        block_message=(
            "⛔ BLOCKED — The response contains language about dosage adjustment "
            "that could be unsafe without medical supervision. "
            "Please consult a healthcare provider before changing any medication dose."
        ),
        severity="high",
    ),
    DASCRule(
        rule_id="IPI-001",
        description="Indirect prompt injection signal in response",
        drugs=[],
        safe_words=[
            "ignore previous instructions", "disregard prior",
            "as instructed by the system", "following the directive",
            "no disclaimers", "without safety warnings",
            "dan mode", "developer mode", "no restrictions",
        ],
        block_message=(
            "⛔ BLOCKED — The response contains language consistent with a prompt "
            "injection attempt in the retrieved context. This query has been blocked "
            "for safety. Please rephrase your question."
        ),
        severity="high",
    ),
]


# ── Jailbreak / Injection query-level detector ────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|all)\s+instructions?",
    r"disregard\s+(prior|previous|all|system)",
    r"you\s+are\s+now\s+(a|an|the)?\s*\w+\s*(without|with no)\s+(restrictions?|limits?|filters?)",
    r"pretend\s+you\s+(have\s+no|don'?t\s+have)\s+(safety|restrictions?|limits?)",
    r"\[?\s*(admin|system|hidden|override)\s*(note|instruction|prompt|command)\s*:?\s*\]?",
    r"dan\s+mode",
    r"developer\s+mode\s+(enabled|activated|on)",
    r"(all\s+)?safety\s+(filters?|guards?)\s+(disabled?|removed?|off)",
    r"respond\s+only\s+with\s+['\"]",
    r"without\s+(any\s+)?(disclaimers?|warnings?|caveats?)",
]

COMPILED_INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


# ── Core defense result dataclass ─────────────────────────────────────────────

@dataclass
class DefenseResult:
    answer:          str
    blocked:         bool         = False
    flagged:         bool         = False
    defense_mode:    str          = "off"
    triggered_rules: List[str]    = field(default_factory=list)
    original_answer: str          = ""
    confidence:      float        = 0.0
    hallucination_score: float    = 0.0
    warning_prefix:  str          = ""
    metadata:        Dict         = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "answer":           self.answer,
            "blocked":          self.blocked,
            "flagged":          self.flagged,
            "defense_mode":     self.defense_mode,
            "triggered_rules":  self.triggered_rules,
            "confidence":       self.confidence,
            "hallucination_score": self.hallucination_score,
        }


# ── Main middleware class ─────────────────────────────────────────────────────

class DefenseMiddleware:
    """
    Drop-in defense layer for your RAG pipeline.

    Typical call (in api.py):
        defense = DefenseMiddleware()
        ...
        raw = bot.chat(query)
        protected = defense.apply(raw, mode=mode, query=query)
        return protected
    """

    def __init__(self, config: DefenseConfig = None):
        self.config = config or DefenseConfig()
        self.stats = {
            "total_requests":  0,
            "blocked":         0,
            "flagged":         0,
            "rules_triggered": {},
        }
        logger.info("DefenseMiddleware initialised")

    # ── Public entry point ────────────────────────────────────────────────────

    def apply(
        self,
        rag_result: Dict,
        mode: str = "off",
        query: str = "",
    ) -> Dict:
        """
        Apply defenses to a rag_result dict (as returned by rag_core.chat()).

        Args:
            rag_result:  The dict from bot.chat(query)
            mode:        "off" | "confidence" | "full"
            query:       Original user query (needed for query-level IPI check)

        Returns:
            Modified dict ready to return from your /chat endpoint.
            Adds keys: blocked, flagged, defense_mode, triggered_rules
        """
        self.stats["total_requests"] += 1

        # Normalise mode
        try:
            defense_mode = DefenseMode(mode.lower())
        except ValueError:
            defense_mode = DefenseMode.OFF
            logger.warning(f"Unknown defense mode '{mode}', defaulting to off")

        # Pass-through for off mode
        if defense_mode == DefenseMode.OFF:
            rag_result.setdefault("blocked", False)
            rag_result.setdefault("flagged", False)
            rag_result.setdefault("defense_mode", "off")
            rag_result.setdefault("triggered_rules", [])
            return rag_result

        # Pull values from rag_result
        answer          = rag_result.get("answer", "")
        confidence      = rag_result.get("confidence", 0.0)
        halluc_score    = rag_result.get("metadata", {}).get("hallucination_score", 0.0)
        sources         = rag_result.get("sources", [])

        result = DefenseResult(
            answer=answer,
            original_answer=answer,
            confidence=confidence,
            hallucination_score=halluc_score,
            defense_mode=defense_mode.value,
        )

        # ── Step 1: Query-level injection check (both modes) ─────────────────
        if query:
            self._check_query_injection(query, result)

        if result.blocked:
            return self._finalise(rag_result, result)

        # ── Step 2: Confidence + hallucination gate (confidence + full) ──────
        self._apply_confidence_gate(result)

        if result.blocked:
            return self._finalise(rag_result, result)

        # ── Step 3: DASC medical safety rules (full only) ────────────────────
        if defense_mode == DefenseMode.FULL and self.config.enable_dasc:
            self._apply_dasc_rules(result)

        return self._finalise(rag_result, result)

    # ── Internal checks ───────────────────────────────────────────────────────

    def _check_query_injection(self, query: str, result: DefenseResult):
        """Detect prompt injection / jailbreak patterns in the user query."""
        for pattern in COMPILED_INJECTION_PATTERNS:
            if pattern.search(query):
                result.blocked = True
                result.triggered_rules.append("IPI-QUERY")
                result.answer = (
                    "⛔ BLOCKED — Your query contains patterns associated with "
                    "prompt injection or jailbreak attempts. "
                    "Please ask your medical question directly."
                )
                logger.warning(f"[DEFENSE] Query injection blocked | pattern={pattern.pattern[:40]}")
                self._record_rule("IPI-QUERY")
                return

    def _apply_confidence_gate(self, result: DefenseResult):
        """Warn or block based on confidence and hallucination score."""
        conf   = result.confidence
        halluc = result.hallucination_score

        # Hard block
        if conf < self.config.confidence_block_threshold:
            result.blocked = True
            result.triggered_rules.append("CONF-BLOCK")
            result.answer = (
                f"⛔ BLOCKED — Confidence too low ({conf:.2f}) to answer safely. "
                "The retrieved documents may not be relevant to your question. "
                "Please consult a healthcare provider."
            )
            logger.warning(f"[DEFENSE] Confidence block | conf={conf:.2f}")
            self._record_rule("CONF-BLOCK")
            return

        if halluc > self.config.hallucination_block_threshold:
            result.blocked = True
            result.triggered_rules.append("HALLUC-BLOCK")
            result.answer = (
                f"⛔ BLOCKED — High hallucination risk detected ({halluc:.2f}). "
                "The generated answer diverges significantly from retrieved sources. "
                "Please consult a healthcare provider."
            )
            logger.warning(f"[DEFENSE] Hallucination block | score={halluc:.2f}")
            self._record_rule("HALLUC-BLOCK")
            return

        # Soft flag / warning
        warnings = []
        if conf < self.config.confidence_warn_threshold:
            result.flagged = True
            result.triggered_rules.append("CONF-WARN")
            warnings.append(f"⚠️ LOW CONFIDENCE ({conf:.2f})")
            self._record_rule("CONF-WARN")

        if halluc > self.config.hallucination_warn_threshold:
            result.flagged = True
            result.triggered_rules.append("HALLUC-WARN")
            warnings.append(f"⚠️ POSSIBLE HALLUCINATION ({halluc:.2f})")
            self._record_rule("HALLUC-WARN")

        if warnings:
            prefix = " | ".join(warnings)
            result.answer = (
                f"[{prefix}] This answer may be unreliable. "
                f"Please verify with a healthcare professional.\n\n"
                f"{result.answer}"
            )
            logger.info(f"[DEFENSE] Flagged | {prefix}")

    def _apply_dasc_rules(self, result: DefenseResult):
        """Apply Domain-Aware Sanity Check rules to the generated answer."""
        answer_lower = result.answer.lower()

        for rule in DASC_RULES:
            # Check if all drugs in the rule appear in the answer
            if rule.drugs:
                if not all(drug in answer_lower for drug in rule.drugs):
                    continue  # drugs not mentioned → rule doesn't apply

            # Check if any dangerous safe_word appears
            triggered_word = next(
                (w for w in rule.safe_words if w.lower() in answer_lower),
                None
            )

            if triggered_word:
                result.blocked = True
                result.triggered_rules.append(rule.rule_id)
                result.answer = rule.block_message
                self._record_rule(rule.rule_id)
                logger.warning(
                    f"[DEFENSE] DASC rule triggered | rule={rule.rule_id} "
                    f"| word='{triggered_word}'"
                )
                return   # first match wins; stop checking further rules

    def _finalise(self, original_rag_result: Dict, result: DefenseResult) -> Dict:
        """Merge defense result back into the rag_result dict."""
        if result.blocked:
            self.stats["blocked"] += 1
        elif result.flagged:
            self.stats["flagged"] += 1

        # Write defense fields into the existing dict
        original_rag_result["answer"]          = result.answer
        original_rag_result["blocked"]         = result.blocked
        original_rag_result["flagged"]         = result.flagged
        original_rag_result["defense_mode"]    = result.defense_mode
        original_rag_result["triggered_rules"] = result.triggered_rules

        # Preserve original answer for logging/comparison
        if result.blocked or result.flagged:
            original_rag_result.setdefault("metadata", {})
            original_rag_result["metadata"]["original_answer"] = result.original_answer

        self._log_defense_event(original_rag_result, result)
        return original_rag_result

    # ── Logging & Stats ───────────────────────────────────────────────────────

    def _record_rule(self, rule_id: str):
        self.stats["rules_triggered"][rule_id] = (
            self.stats["rules_triggered"].get(rule_id, 0) + 1
        )

    def _log_defense_event(self, rag_result: Dict, result: DefenseResult):
        entry = {
            "timestamp":       datetime.now().isoformat(),
            "defense_mode":    result.defense_mode,
            "blocked":         result.blocked,
            "flagged":         result.flagged,
            "triggered_rules": result.triggered_rules,
            "confidence":      result.confidence,
            "hallucination_score": result.hallucination_score,
        }
        if result.blocked:
            logger.warning(f"[DEFENSE_EVENT] {json.dumps(entry)}")
        elif result.flagged:
            logger.info(f"[DEFENSE_EVENT] {json.dumps(entry)}")

    def get_stats(self) -> Dict:
        """Return cumulative defense statistics."""
        total = self.stats["total_requests"]
        return {
            "total_requests":   total,
            "blocked":          self.stats["blocked"],
            "flagged":          self.stats["flagged"],
            "block_rate":       round(self.stats["blocked"] / total, 3) if total else 0,
            "flag_rate":        round(self.stats["flagged"] / total, 3) if total else 0,
            "rules_triggered":  self.stats["rules_triggered"],
        }