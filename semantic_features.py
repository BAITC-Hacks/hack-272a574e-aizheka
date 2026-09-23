"""Cached semantic judgments over aggregate candidate evidence."""

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ai_provider import canonical


FEATURE_QUESTIONS = {
    "offer_fit": {"type": "choice", "instructions":
        "Classify the target offer for this aggregate audience using only the supplied fields. "
        "Unknown consumption is not zero. No inference about campaign conversion.",
        "criteria": {"upgrade": "Meaningfully meets needs unmet by current tariff",
                     "similar": "Comparable offer with no clear needs improvement",
                     "mismatch": "Important supplied needs are unmet",
                     "unknown": "Evidence is insufficient"}},
    "consumption_fit": {"type": "score", "instructions":
        "How well do target allowances fit the known consumption?",
        "criteria": ["Known important needs are unmet", "Partial fit or insufficient evidence",
                     "Known consumption is adequately covered"]},
    "negative_reaction": {"type": "score", "instructions":
        "Rate evident negative-reaction risk from price increase or lost allowances, not conversion probability.",
        "criteria": ["No evident increase or lost needed allowance", "Mixed or incomplete evidence",
                     "Clear price burden or loss of needed allowance"]},
    "needs_mismatch": {"type": "noul", "instructions":
        "Is there an evident mismatch between target allowances and known audience needs?"},
}


def validate_answers(answers, questions):
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("answer_ids")
    def number(value, low, high):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("answer_number")
    for name, question in questions.items():
        answer = answers[name]
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise ValueError("answer_type")
        if question["type"] == "noul":
            number(answer.get("noul"), 0, 1)
            continue
        number(answer.get("confidence"), 0, 1)
        probabilities = answer.get("probabilities")
        criteria = ({str(i): value for i, value in enumerate(question["criteria"])}
                    if question["type"] == "score" else question["criteria"])
        if not isinstance(probabilities, dict) or set(probabilities) != set(criteria):
            raise ValueError("probability_labels")
        for probability in probabilities.values():
            number(probability, 0, 1)
        if abs(sum(probabilities.values()) - 1) > 0.01:
            raise ValueError("probability_sum")
        if question["type"] == "choice":
            if answer.get("choice") not in question["criteria"]:
                raise ValueError("choice_label")
        else:
            levels = [float(level) for level in criteria]
            number(answer.get("score"), min(levels), max(levels))
            if answer.get("legend") != criteria:
                raise ValueError("score_legend")
            expectation = sum(float(level) * probability for level, probability in probabilities.items())
            if abs(expectation - answer["score"]) > 0.02:
                raise ValueError("score_expectation")
    return answers


def candidate_state(belief, env):
    rows = env.customer_profile
    rows = rows[(rows.current_tariff == belief.current_tariff) & (rows.arpu_segment == belief.arpu_segment)]
    needs = {}
    for column in ("DATA_VOLUME", "LTE_DATA_VOLUME", "OUT_LOC_OFFNET_MIN", "OUT_LOC_ONNET_MIN", "predicted_arpu"):
        values = pd.to_numeric(rows[column], errors="coerce") if column in rows else pd.Series(dtype=float)
        needs[column] = {"median": None if values.dropna().empty else float(values.median()),
                         "missing_fraction": 1.0 if values.empty else float(values.isna().mean())}
    def tariff(code):
        record = env.tariffs[env.tariffs.tariff_plan_code == code]
        if record.empty:
            return None
        return json.loads(record.iloc[0].to_json())
    return {"schema": 1, "candidate_id": belief.candidate_id,
            "audience": {"size": len(rows), "arpu_segment": belief.arpu_segment, "needs": needs},
            "current_tariff": tariff(belief.current_tariff), "target_tariff": tariff(belief.target_tariff),
            "channel": belief.channel, "units": {"data": "MB", "voice": "minutes", "price": "case currency"},
            "limitations": "Aggregate medians; data columns may overlap and must not be added. No response labels."}


def cache_key(state, model, questions=FEATURE_QUESTIONS):
    return hashlib.sha256(canonical({"schema": 1, "state": state, "model": model,
                                    "questions": questions}).encode()).hexdigest()


class SemanticFeatures:
    def __init__(self, mode="offline", provider=None, cache_dir=".cache/jev", model="vercel:typesafe-ai/jev"):
        if mode not in {"offline", "live", "replay"}:
            raise ValueError("invalid_ai_mode")
        self.mode, self.provider, self.model = mode, provider, model
        self.cache_dir = Path(cache_dir)
        self.events = []

    def evaluate(self, state):
        if self.mode == "offline":
            return None
        key = cache_key(state, self.model)
        path = self.cache_dir / f"{key}.json"
        try:
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("key") != key or record.get("requested_model") != self.model:
                    raise ValueError("stale_cache")
                return validate_answers(record["answers"], FEATURE_QUESTIONS)
            if self.mode == "replay" or self.provider is None:
                return None
            response = self.provider.jev(state, FEATURE_QUESTIONS)
            answers = validate_answers(response["answers"], FEATURE_QUESTIONS)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            record = {"key": key, "requested_model": self.model, "model": response.get("model"),
                      "answers": answers}
            temporary = path.with_suffix(".tmp")
            temporary.write_text(canonical(record), encoding="utf-8")
            temporary.replace(path)
            return answers
        except Exception:
            self.events.append({"state_hash": key, "reason": "semantic_unavailable"})
            return None

    def collect(self, beliefs, env):
        if self.mode == "offline":
            return {}
        states = [candidate_state(belief, env) for belief in beliefs]
        with ThreadPoolExecutor(max_workers=2) as pool:
            answers = list(pool.map(self.evaluate, states))
        return {belief.candidate_id: answer for belief, answer in zip(beliefs, answers) if answer is not None}


def feature_multiplier(answers):
    """Small, confidence-weighted ranking influence; never a hard filter."""
    fit, risk = answers["consumption_fit"], answers["negative_reaction"]
    return 1 + 0.05 * (fit["confidence"] * (fit["score"] - 1)
                      - risk["confidence"] * (risk["score"] - 1))
