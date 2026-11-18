from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from mirage.generator.http_provider import HTTPConfig, HTTPProvider
from mirage.generator.provider import IsolatedProvider, ProviderFailure, Request, Stage
from mirage.generator.repair import Diagnostic, DiagnosticKind, generate


@contextmanager
def stub_service(respond):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            status, body = respond(payload)
            encoded = body if isinstance(body, bytes) else json.dumps(body).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def completion(payload, *, usage=True):
    result = {"model": payload["model"], "choices": [
        {"message": {"content": '{"source": "code", "actions": []}'}, "finish_reason": "stop"},
    ]}
    if usage:
        result["usage"] = {"prompt_tokens": 11, "completion_tokens": 7}
    return result


class HTTPProviderTests(unittest.TestCase):
    def test_training_path_payload_is_constrained_to_offered_complete_paths(self):
        paths = [["zero:none"], ["forward", "backward", "zero:none"]]
        request = Request(Stage.TRANSITION, "Choose", {"contract": "state-plan", "offered_paths": paths})
        payload = HTTPProvider(HTTPConfig("http://localhost/v1"))._payload(request)
        actions = payload["response_format"]["json_schema"]["schema"]["properties"]["actions"]
        self.assertEqual(actions["enum"], paths)

    def setUp(self):
        self.request = Request(Stage.CANONICAL, "Generate", {"configuration": {"seed": 17}})

    def test_payload_repairs_usage_and_original_evidence(self):
        with stub_service(lambda p: (200, completion(p))) as (url, requests), tempfile.TemporaryDirectory() as temporary:
            provider = HTTPProvider(HTTPConfig(url), directory=Path(temporary))
            response = provider.generate(replace(self.request, previous_content="broken", diagnostic="missing path"),
                                         deadline=time.monotonic() + 5)
            path, payload = requests[0]
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(payload["seed"], 17)
            self.assertEqual(payload["messages"][-2]["content"], "broken")
            self.assertIn("missing path", payload["messages"][-1]["content"])
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertEqual(response.usage.input_tokens, 11)
            saved = json.loads(next(Path(temporary).glob("http-*.json")).read_text())
            self.assertEqual(json.loads(saved["response"])["model"], response.model)

    def test_missing_usage_is_unknown(self):
        with stub_service(lambda p: (200, completion(p, usage=False))) as (url, _):
            response = HTTPProvider(HTTPConfig(url)).generate(self.request, deadline=None)
            self.assertIsNone(response.usage.input_tokens)
            self.assertIsNone(response.usage.output_tokens)

    def test_service_and_protocol_errors_never_enter_program_repair(self):
        for status, body, category in ((503, {"error": "unavailable"}, "http"),
                                       (200, b"not json", "protocol")):
            with self.subTest(category=category), stub_service(lambda p: (status, body)) as (url, requests):
                result = generate(HTTPProvider(HTTPConfig(url)), self.request,
                                  lambda _: Diagnostic(DiagnosticKind.ACCEPTED), deadline=time.monotonic() + 5)
                self.assertEqual(result.status, "provider_" + category)
                self.assertEqual(len(requests), 1)
                self.assertIsNone(result.attempts[0].diagnostic)

    def test_wrong_model_truncated_content_and_invalid_usage(self):
        def wrong_model(payload):
            value = completion(payload)
            value["model"] = "other"
            return value

        def truncated(payload):
            value = completion(payload)
            value["choices"][0]["finish_reason"] = "length"
            return value

        def invalid_usage(payload):
            value = completion(payload)
            value["usage"]["prompt_tokens"] = -1
            return value

        for factory, category in ((wrong_model, "protocol"), (truncated, "incomplete"), (invalid_usage, "protocol")):
            with self.subTest(category=category), stub_service(lambda p: (200, factory(p))) as (url, _):
                with self.assertRaises(ProviderFailure) as caught:
                    HTTPProvider(HTTPConfig(url)).generate(self.request, deadline=None)
                self.assertEqual(caught.exception.category, category)

    def test_timeout_is_bounded_and_cancellation_is_not_claimed(self):
        def slow(payload):
            time.sleep(0.3)
            return 200, completion(payload)

        with stub_service(slow) as (url, _), tempfile.TemporaryDirectory() as temporary:
            provider = HTTPProvider(HTTPConfig(url, timeout=0.05), directory=Path(temporary))
            with self.assertRaises(ProviderFailure) as caught:
                provider.generate(self.request, deadline=time.monotonic() + 2)
            self.assertEqual(caught.exception.category, "timeout")
            self.assertTrue(caught.exception.cancellation_uncertain)
            saved = json.loads(next(Path(temporary).glob("http-*.json")).read_text())
            self.assertEqual(saved["cancellation"], "unconfirmed")

    def test_isolated_adapter_retains_metadata_and_error_category(self):
        for status in (200, 503):
            with stub_service(lambda p: (status, completion(p))) as (url, _), tempfile.TemporaryDirectory() as temporary:
                with patch.dict(os.environ, {"MIRAGE_VLLM_BASE_URL": url}):
                    provider = IsolatedProvider("vllm", Path(temporary), 5)
                    if status == 200:
                        response = provider.generate(self.request, deadline=time.monotonic() + 5)
                        self.assertIn("response", response.metadata)
                    else:
                        with self.assertRaises(ProviderFailure) as caught:
                            provider.generate(self.request, deadline=time.monotonic() + 5)
                        self.assertEqual(caught.exception.category, "http")

    def test_expired_deadline_never_sends_and_invalid_config_is_rejected(self):
        with stub_service(lambda p: (200, completion(p))) as (url, requests):
            with self.assertRaises(TimeoutError):
                HTTPProvider(HTTPConfig(url)).generate(self.request, deadline=time.monotonic() - 1)
            self.assertEqual(requests, [])
            for values in ({"max_tokens": 0}, {"temperature": float("nan")}, {"timeout": 0}):
                with self.assertRaises(ValueError):
                    HTTPConfig(url, **values)


class StatePlanPayloadTests(unittest.TestCase):
    def test_separate_schemas_and_stable_api_prefix(self):
        provider = HTTPProvider(HTTPConfig('http://localhost:8000/v1'))
        for contract, fields in [('state-goal-plan',['plans', 'examined_goal_ids']), ('region-patches',['patches'])]:
            request = Request(Stage.STATE_GOAL, 'variable task', {'contract':contract,
                'api_context':{'documentation':'installed spec', 'documentation_references':[{'documentation':'referenced spec'}]}})
            payload = provider._payload(request)
            schema = payload['response_format']['json_schema']['schema']
            self.assertEqual(schema['required'], fields)
            if contract == 'state-goal-plan':
                for branch in schema['properties']['plans']['items']['anyOf']:
                    self.assertIn('status', branch['required'])
                    self.assertIn('blockers', branch['required'])
            content = payload['messages'][1]['content']
            self.assertLess(content.index('installed spec'), content.index('variable task'))
            self.assertIn('referenced spec', content)


    def test_plan_payload_limits_both_id_fields_to_this_request_without_mutating_template(self):
        from mirage.generator.state_plan import PLAN_SCHEMA
        provider = HTTPProvider(HTTPConfig('http://localhost:8000/v1'))
        original = json.dumps(PLAN_SCHEMA, sort_keys=True)
        def payload(ids):
            request = Request(Stage.STATE_GOAL, 'Plan', {'contract':'state-goal-plan',
                'goal_catalog':[{'id':identifier} for identifier in ids]})
            return provider._payload(request)['response_format']['json_schema']['schema']
        first, second = payload(['goal-a', 'goal-b']), payload(['goal-c'])
        for schema, identifiers in ((first, ['goal-a','goal-b']), (second, ['goal-c'])):
            self.assertEqual(schema['properties']['examined_goal_ids']['items']['enum'], identifiers)
            for branch in schema['properties']['plans']['items']['anyOf']:
                self.assertEqual(branch['properties']['goal_ids']['items']['enum'], identifiers)
        first['properties']['plans']['items']['anyOf'][0]['properties']['goal_ids']['items']['enum'].append('changed')
        self.assertEqual(json.dumps(PLAN_SCHEMA, sort_keys=True), original)
        self.assertEqual(second['properties']['examined_goal_ids']['items']['enum'], ['goal-c'])
        self.assertEqual(first['properties']['examined_goal_ids']['items']['enum'], ['goal-a','goal-b'])

    def test_plan_schema_discriminates_executable_actions_from_unresolved_blockers(self):
        from mirage.generator.state_plan import planning_schema
        schema = planning_schema([{'id':'goal-a'}])
        branches = {branch['properties']['status']['enum'][0]:branch
                    for branch in schema['properties']['plans']['items']['anyOf']}
        self.assertEqual(set(branches), {'planned', 'unresolved'})
        for status, branch in branches.items():
            self.assertFalse(branch['additionalProperties'])
            self.assertEqual(set(branch['required']), set(branch['properties']))
        planned, unresolved = (branches[status]['properties'] for status in ('planned','unresolved'))
        self.assertEqual(planned['actions']['minItems'], 1)
        self.assertEqual(planned['regions']['minItems'], 1)
        self.assertEqual(planned['blockers']['maxItems'], 0)
        self.assertEqual(unresolved['actions']['maxItems'], 0)
        self.assertEqual(unresolved['regions']['maxItems'], 0)
        self.assertEqual(unresolved['blockers']['minItems'], 1)

    def test_empty_catalog_can_only_emit_empty_arrays(self):
        from mirage.generator.state_plan import planning_schema
        schema = planning_schema([])
        self.assertEqual(schema['properties']['plans']['maxItems'], 0)
        self.assertEqual(schema['properties']['examined_goal_ids']['maxItems'], 0)
        self.assertNotIn('"enum": []', json.dumps(schema))
        for catalog in ([{'id':'same'}, {'id':'same'}], [{'id':''}], [{'id':None}]):
            with self.subTest(catalog=catalog), self.assertRaises(ValueError):
                planning_schema(catalog)
