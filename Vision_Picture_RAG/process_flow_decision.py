"""
Process flow decision logic for the RAG/3D-scan pipeline.
This module provides a small rule engine that selects a scenario
and returns next steps based on available inputs and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


class Scenario(str, Enum):
    IDENTIFY_FROM_SCAN = "identify_from_scan"
    SEARCH_TEXT = "search_text"
    SEARCH_IMAGE = "search_image"
    SEARCH_HYBRID = "search_hybrid"
    REQUEST_RESCAN = "request_rescan"
    REQUEST_MORE_DATA = "request_more_data"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True)
class InputContext:
    query_type: str
    has_text: bool
    has_image: bool
    has_scan: bool
    scan_quality: Optional[float] = None
    top1_score: Optional[float] = None
    top1_margin: Optional[float] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionResult:
    scenario: Scenario
    reason: str
    confidence: float
    next_steps: Tuple[str, ...]


@dataclass(frozen=True)
class Rule:
    name: str
    priority: int
    predicate: Callable[[InputContext], bool]
    scenario: Scenario
    reason: str
    next_steps: Tuple[str, ...]


class DecisionEngine:
    def __init__(self, rules: Optional[List[Rule]] = None) -> None:
        self._rules = sorted(rules or build_default_rules(), key=lambda r: r.priority)

    def decide(self, ctx: InputContext) -> DecisionResult:
        for rule in self._rules:
            if rule.predicate(ctx):
                confidence = estimate_confidence(ctx, rule.scenario)
                return DecisionResult(
                    scenario=rule.scenario,
                    reason=rule.reason,
                    confidence=confidence,
                    next_steps=rule.next_steps,
                )
        return DecisionResult(
            scenario=Scenario.MANUAL_REVIEW,
            reason="No rule matched; fallback to manual review.",
            confidence=0.1,
            next_steps=("route_to_operator",),
        )


def build_default_rules() -> List[Rule]:
    return [
        Rule(
            name="rescan_low_quality",
            priority=10,
            predicate=lambda c: c.has_scan and (c.scan_quality is not None) and c.scan_quality < 0.6,
            scenario=Scenario.REQUEST_RESCAN,
            reason="Scan quality below threshold.",
            next_steps=("request_new_scan", "check_scanner_settings"),
        ),
        Rule(
            name="identify_from_scan",
            priority=20,
            predicate=lambda c: c.has_scan and (c.scan_quality is None or c.scan_quality >= 0.6),
            scenario=Scenario.IDENTIFY_FROM_SCAN,
            reason="Valid 3D scan available.",
            next_steps=("run_3d_comparator", "rank_candidates"),
        ),
        Rule(
            name="hybrid_search",
            priority=30,
            predicate=lambda c: c.query_type == "hybrid" and c.has_text and c.has_image,
            scenario=Scenario.SEARCH_HYBRID,
            reason="Text and image provided; hybrid search recommended.",
            next_steps=("build_text_embedding", "build_image_embedding", "run_hybrid_retrieval"),
        ),
        Rule(
            name="image_search",
            priority=40,
            predicate=lambda c: c.query_type == "image" and c.has_image,
            scenario=Scenario.SEARCH_IMAGE,
            reason="Image provided; image search selected.",
            next_steps=("build_image_embedding", "run_image_retrieval"),
        ),
        Rule(
            name="text_search",
            priority=50,
            predicate=lambda c: c.query_type == "text" and c.has_text,
            scenario=Scenario.SEARCH_TEXT,
            reason="Text query provided; text search selected.",
            next_steps=("build_text_embedding", "run_text_retrieval"),
        ),
        Rule(
            name="request_more_data",
            priority=60,
            predicate=lambda c: not c.has_text and not c.has_image and not c.has_scan,
            scenario=Scenario.REQUEST_MORE_DATA,
            reason="No input data available.",
            next_steps=("request_text_or_image",),
        ),
    ]


def estimate_confidence(ctx: InputContext, scenario: Scenario) -> float:
    if scenario == Scenario.IDENTIFY_FROM_SCAN:
        if ctx.top1_score is not None:
            return min(0.95, 0.5 + ctx.top1_score / 2.0)
        return 0.7 if (ctx.scan_quality or 0) >= 0.6 else 0.4
    if scenario in (Scenario.SEARCH_TEXT, Scenario.SEARCH_IMAGE, Scenario.SEARCH_HYBRID):
        if ctx.top1_score is not None and ctx.top1_margin is not None:
            return min(0.9, 0.4 + 0.5 * ctx.top1_score + 0.2 * ctx.top1_margin)
        return 0.6
    if scenario == Scenario.REQUEST_RESCAN:
        return 0.8
    if scenario == Scenario.REQUEST_MORE_DATA:
        return 0.7
    return 0.3


def post_check_low_confidence(result: DecisionResult, ctx: InputContext) -> DecisionResult:
    if ctx.top1_score is not None and ctx.top1_score < 0.55:
        return DecisionResult(
            scenario=Scenario.MANUAL_REVIEW,
            reason="Top-1 similarity below acceptance threshold.",
            confidence=0.2,
            next_steps=("route_to_operator", "collect_more_data"),
        )
    return result


def decide_with_postcheck(ctx: InputContext) -> DecisionResult:
    engine = DecisionEngine()
    initial = engine.decide(ctx)
    return post_check_low_confidence(initial, ctx)


if __name__ == "__main__":
    example_ctx = InputContext(
        query_type="hybrid",
        has_text=True,
        has_image=True,
        has_scan=False,
        scan_quality=None,
        top1_score=0.82,
        top1_margin=0.15,
    )
    decision = decide_with_postcheck(example_ctx)
    print(decision)
