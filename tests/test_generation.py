import time
import unittest
from unittest.mock import patch

from mirage.generator.provider import Request, Response, Stage, Usage
from mirage.generator.repair import Diagnostic, DiagnosticKind, generate


class FakeProvider:
    def __init__(self):
        self.requests = []

    def generate(self, request, *, deadline):
        self.requests.append(request)
        return Response(f"version {len(self.requests)}")


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.provider = FakeProvider()
        self.request = Request(Stage.CANONICAL, "generate a legal program")

    def test_two_repairs_preserve_three_versions(self):
        saved = []
        result = generate(self.provider, self.request,
                          lambda _: Diagnostic(DiagnosticKind.SYNTAX, "bad syntax"), record=saved.append)
        self.assertEqual(result.status, "repair_exhausted")
        self.assertEqual(len(saved), 3)
        self.assertEqual([a.response.content for a in saved], ["version 1", "version 2", "version 3"])
        self.assertEqual(saved[1].request.previous_content, "version 1")
        self.assertIsNone(saved[0].request.previous_content)
        self.assertIsNone(saved[0].response.usage.input_tokens)

    def test_failure_candidates_never_repaired(self):
        for kind in (DiagnosticKind.UNKNOWN_EXCEPTION, DiagnosticKind.SUSPECTED_FAILURE,
                     DiagnosticKind.UNRESOLVED_PLAN, DiagnosticKind.STATE_NOT_REACHED):
            result = generate(FakeProvider(), self.request, lambda _: Diagnostic(kind))
            self.assertEqual(len(result.attempts), 1)
            self.assertEqual(result.status, kind.value)

    def test_accepted_and_repairable_diagnostics(self):
        for kind in (DiagnosticKind.CONFIRMED_ILLEGAL, DiagnosticKind.MISSING_PATH):
            diagnostics = iter([Diagnostic(kind), Diagnostic(DiagnosticKind.ACCEPTED)])
            result = generate(FakeProvider(), self.request, lambda _: next(diagnostics))
            self.assertEqual(result.status, "accepted")
            self.assertEqual(len(result.attempts), 2)

    def test_validator_exception_preserves_original(self):
        def validate(_):
            raise RuntimeError("unexplained crash")

        result = generate(self.provider, self.request, validate)
        self.assertEqual(result.status, "unresolved")
        self.assertEqual(result.attempts[0].response.content, "version 1")
        self.assertIn("unexplained crash", result.attempts[0].error)

    def test_provider_failure_is_saved(self):
        class BrokenProvider:
            def generate(self, request, *, deadline):
                raise RuntimeError("model unavailable")

        result = generate(BrokenProvider(), self.request, lambda _: None)
        self.assertEqual(result.status, "unresolved")
        self.assertIsNone(result.attempts[0].response)
        self.assertIn("model unavailable", result.attempts[0].error)

    def test_expired_budget_does_not_call_provider(self):
        result = generate(self.provider, self.request, lambda _: None, deadline=time.monotonic() - 1)
        self.assertEqual(result.status, "timeout")
        self.assertEqual(self.provider.requests, [])

    def test_limits_and_usage(self):
        for repairs in (-1, 3, True):
            with self.assertRaises(ValueError):
                generate(self.provider, self.request, lambda _: None, max_repairs=repairs)
        with self.assertRaises(ValueError):
            Usage(input_tokens=-1)
        self.assertIsNone(Usage().input_tokens)
        self.assertEqual(Usage(input_tokens=0).input_tokens, 0)

    def test_late_provider_response_is_retained_but_not_accepted(self):
        with patch("mirage.generator.repair.time.monotonic", side_effect=[0, 0, 11, 11]):
            result = generate(self.provider, self.request,
                              lambda _: Diagnostic(DiagnosticKind.ACCEPTED), deadline=10)
        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.attempts[0].response.content, "version 1")

    def test_validation_time_is_part_of_budget(self):
        with patch("mirage.generator.repair.time.monotonic", side_effect=[0, 0, 1, 11, 11]):
            result = generate(self.provider, self.request,
                              lambda _: Diagnostic(DiagnosticKind.ACCEPTED), deadline=10)
        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.attempts[0].diagnostic.kind, DiagnosticKind.ACCEPTED)

    def test_nonfinite_deadline_is_rejected(self):
        with self.assertRaises(ValueError):
            generate(self.provider, self.request, lambda _: None, deadline=float("nan"))
