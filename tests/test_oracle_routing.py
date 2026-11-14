"""Scoped synthetic audit results test routing, never a framework defect claim."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from mirage.inspector.oracles import (AUDIT_VERSION, audit_execution, audit_pair,
                                      verified_violation_calls)


def record():
    reference = {'call': 0, 'node': 0, 'invocation': 0}
    return {'completed': True, 'inspection': {'valid': True},
            'target_calls': [{'index': 0}],
            'backward_invocations': [{'index': 0, 'completed': True}],
            'gradient_evidence': {'version': 'gradient-evidence-1',
                                  'events': [dict(reference, status='complete')]},
            'gradient_audit': {'version': AUDIT_VERSION, 'scope': 'target_vjp',
                               'status': 'violation',
                               'events': [dict(reference, status='violation', baseline_verified=True,
                                               scope='target_vjp', stable_steps=True, one_sided_consistent=True,
                                               directions=[{'status': 'violation', 'stable_steps': True,
                                                            'one_sided_consistent': True}])]}}


class OracleRoutingTests(unittest.TestCase):
    def test_violation_requires_complete_original_evidence_and_baseline(self):
        original = record()
        self.assertEqual(verified_violation_calls(original), (0,))
        changes = [
            lambda r: r.update(completed=False),
            lambda r: r['inspection'].update(valid=False),
            lambda r: r['backward_invocations'][0].update(completed=False),
            lambda r: r['gradient_evidence'].update(version='legacy'),
            lambda r: r['gradient_evidence']['events'][0].update(status='unknown'),
            lambda r: r['gradient_audit']['events'][0].update(baseline_verified=False),
            lambda r: r['gradient_audit']['events'][0].pop('scope'),
            lambda r: r['gradient_audit']['events'][0].update(scope='whole_program'),
            lambda r: r['gradient_audit']['events'][0].pop('stable_steps'),
            lambda r: r['gradient_audit']['events'][0].update(stable_steps=False),
            lambda r: r['gradient_audit']['events'][0].pop('one_sided_consistent'),
            lambda r: r['gradient_audit']['events'][0].update(one_sided_consistent=False),
            lambda r: r['gradient_audit']['events'][0].pop('directions'),
            lambda r: r['gradient_audit']['events'][0].update(directions=[]),
            lambda r: r['gradient_audit']['events'][0]['directions'][0].update(status='agreement'),
            lambda r: r['gradient_audit']['events'][0]['directions'][0].pop('stable_steps'),
            lambda r: r['gradient_audit']['events'][0]['directions'][0].update(stable_steps=False),
            lambda r: r['gradient_audit']['events'][0]['directions'][0].pop('one_sided_consistent'),
            lambda r: r['gradient_audit']['events'][0]['directions'][0].update(one_sided_consistent=False),
            lambda r: r['gradient_audit']['events'][0].update(invocation=1),
            lambda r: r['gradient_audit']['events'][0].update(call=1),
            lambda r: r['gradient_audit'].update(version='unknown'),
            lambda r: r['gradient_audit'].update(scope='whole_program'),
            lambda r: r['gradient_audit'].update(status='agreement'),
        ]
        for change in changes:
            with self.subTest(change=change):
                changed = deepcopy(original)
                change(changed)
                self.assertEqual(verified_violation_calls(changed), ())
        self.assertEqual(verified_violation_calls({}), ())

    def test_agreement_does_not_upgrade_coverage_only_or_mr_verdict(self):
        left = record()
        result = dict(left['gradient_audit'], status='agreement', events=[])
        for verdict in ('inapplicable', 'unassessed', 'suspected_failure'):
            with self.subTest(verdict=verdict):
                summary = {'verdict': verdict, 'correctness': 'unassessed', 'reason': 'existing MR reason'}
                with patch('mirage.inspector.oracles.audit_execution', return_value=deepcopy(result)):
                    audit_pair(left, None, summary)
                self.assertEqual(summary['mr_verdict'], verdict)
                self.assertEqual(summary['verdict'], verdict)
                self.assertEqual(summary['correctness'], 'unassessed')
                self.assertEqual(summary['reason'], 'existing MR reason')
                self.assertEqual(summary['numerical_violation_members'], [])

    def test_numerical_violation_preserves_separate_mr_result(self):
        left = record()
        result = deepcopy(left['gradient_audit'])
        summary = {'verdict': 'pass', 'correctness': 'assessed'}
        with patch('mirage.inspector.oracles.audit_execution', return_value=result):
            audit_pair(left, None, summary)
        self.assertEqual(summary['mr_verdict'], 'pass')
        self.assertEqual(summary['verdict'], 'suspected_failure')
        self.assertEqual(summary['numerical_violation_members'], ['source'])
        self.assertNotIn('confirmed_defect', summary)

    def test_failed_baseline_cannot_raise_overall_suspected_failure(self):
        left = record()
        result = deepcopy(left['gradient_audit'])
        result['events'][0]['baseline_verified'] = False
        summary = {'verdict': 'inapplicable', 'correctness': 'unassessed'}
        with patch('mirage.inspector.oracles.audit_execution', return_value=result):
            audit_pair(left, None, summary)
        self.assertEqual(summary['verdict'], 'inapplicable')
        self.assertEqual(summary['numerical_violation_members'], [])

    def test_disabled_audit_preserves_mr_and_records_unknown(self):
        left = record()
        summary = {'verdict': 'pass'}
        with patch('mirage.inspector.oracles.audit_execution') as run:
            audit_pair(left, None, summary, enabled=False)
        run.assert_not_called()
        self.assertEqual(summary['verdict'], 'pass')
        self.assertEqual(left['gradient_audit']['status'], 'unknown')
        self.assertEqual(summary['numerical_violation_members'], [])

    def test_routing_rejects_missing_or_unverified_execution(self):
        self.assertEqual(audit_execution({})['status'], 'unknown')
        original = record()
        original.pop('gradient_evidence')
        self.assertIn('legacy', audit_execution(original)['reason'])
        original = record()
        self.assertEqual(audit_execution(original)['status'], 'unsupported')
        original['run_rng_unchanged'] = True
        self.assertEqual(audit_execution(original)['status'], 'unsupported')


if __name__ == '__main__':
    unittest.main()
