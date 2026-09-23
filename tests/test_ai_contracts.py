import copy
import tempfile
import unittest

from ai_provider import Budget, response_usage
from semantic_features import FEATURE_QUESTIONS, SemanticFeatures, cache_key, validate_answers
from verified_cascade import evidence_packet, run_cascade, validate_proposal


def answers():
    result = {}
    for name, question in FEATURE_QUESTIONS.items():
        kind = question['type']
        if kind == 'noul':
            result[name] = {'type': kind, 'noul': 0.5}
        elif kind == 'score':
            legend = {str(i): value for i, value in enumerate(question['criteria'])}
            result[name] = {'type': kind, 'score': 1, 'confidence': 1,
                            'legend': legend, 'probabilities': {'0': 0, '1': 1, '2': 0}}
        else:
            result[name] = {'type': kind, 'choice': 'unknown', 'confidence': 1,
                            'probabilities': {key: int(key == 'unknown') for key in question['criteria']}}
    return result


class AIContracts(unittest.TestCase):
    def test_score_request_and_response_shapes(self):
        self.assertIsInstance(FEATURE_QUESTIONS['consumption_fit']['criteria'], list)
        validate_answers(answers(), FEATURE_QUESTIONS)

    def test_missing_and_nonfinite_answers_rejected(self):
        bad = answers()
        bad['consumption_fit']['score'] = float('nan')
        with self.assertRaises(ValueError):
            validate_answers(bad, FEATURE_QUESTIONS)
        with self.assertRaises(ValueError):
            validate_answers({}, FEATURE_QUESTIONS)

    def test_budget_and_vercel_accounting(self):
        budget = Budget(.02)
        budget.reserve(.02)
        with self.assertRaises(RuntimeError):
            budget.reserve(.02)
        usage = response_usage({'usage': {'input_tokens': 10},
                                'provider_metadata': {'gateway': {'cost': '0.001'}}})
        budget.settle(.02, usage, 'test', 'vercel')
        self.assertAlmostEqual(budget.committed, .001)

    def test_cache_changes_with_evidence_and_model(self):
        self.assertNotEqual(cache_key({'x': 1}, 'a'), cache_key({'x': 2}, 'a'))
        self.assertNotEqual(cache_key({'x': 1}, 'a'), cache_key({'x': 1}, 'b'))

    def test_outage_and_replay_miss_return_none(self):
        class Unavailable:
            def jev(self, *args):
                raise RuntimeError('unavailable')
        with tempfile.TemporaryDirectory() as directory:
            features = SemanticFeatures('live', Unavailable(), directory)
            self.assertIsNone(features.evaluate({'candidate_id': 'a'}))
            self.assertEqual(len(features.events), 1)
            self.assertIsNone(SemanticFeatures('replay', cache_dir=directory).evaluate({}))

    def test_stale_evidence_and_invented_claim_rejected(self):
        packet = evidence_packet([{'candidate_id': 'a', 'size': 100}])
        proposal = {'candidate_id': 'a', 'evidence_hash': packet['evidence_hash'],
                    'rationale': 'The audience contains 100 customers.', 'source_ids': ['a'],
                    'claims': [{'source_id': 'a', 'field': 'size', 'value': 100}]}
        validate_proposal(proposal, packet)
        bad = copy.deepcopy(proposal)
        bad['claims'][0]['value'] = 101
        with self.assertRaises(ValueError):
            validate_proposal(bad, packet)
        packet['sources']['a']['size'] = 101
        with self.assertRaises(ValueError):
            validate_proposal(proposal, packet)

    def test_cascade_outage_cannot_approve(self):
        class Unavailable:
            def draft(self, *args):
                raise RuntimeError('unavailable')
        proposal, events = run_cascade(Unavailable(), evidence_packet([{'candidate_id': 'a'}]), 'cheap', 'strong')
        self.assertIsNone(proposal)
        self.assertEqual(events[0]['reason'], 'provider_unavailable')
