"""Dependency-free adapter for a prepared vLLM chat-completions service."""

from dataclasses import dataclass
import http.client
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid

from mirage.generator.provider import ProviderFailure, Request, Response, Usage


PROGRAM_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "actions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["source", "actions"],
    "additionalProperties": False,
}
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class HTTPConfig:
    base_url: str
    model: str = "Qwen/Qwen3-Coder-Next-FP8"
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    timeout: float = 120.0

    def __post_init__(self):
        url = urlsplit(self.base_url)
        if (url.scheme not in ("http", "https") or not url.hostname
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("Expected an HTTP(S) base URL without credentials, query or fragment")
        if not self.model.strip() or type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("A model and positive max_tokens are required")
        for name in ("temperature", "top_p", "timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1 or self.timeout <= 0:
            raise ValueError("Invalid sampling parameters or HTTP timeout")


class HTTPProvider:
    def __init__(self, config: HTTPConfig, *, directory: Path | None = None):
        self.config = config
        self.directory = directory

    @classmethod
    def from_environment(cls, *, directory=None):
        base_url = os.environ.get("MIRAGE_VLLM_BASE_URL")
        if not base_url:
            raise ValueError("Set MIRAGE_VLLM_BASE_URL to a prepared service's /v1 endpoint")
        return cls(HTTPConfig(
            base_url=base_url,
            model=os.environ.get("MIRAGE_VLLM_MODEL", "Qwen/Qwen3-Coder-Next-FP8"),
            max_tokens=int(os.environ.get("MIRAGE_MAX_TOKENS", "4096")),
            temperature=float(os.environ.get("MIRAGE_TEMPERATURE", "0")),
            top_p=float(os.environ.get("MIRAGE_TOP_P", "1")),
            timeout=float(os.environ.get("MIRAGE_HTTP_TIMEOUT", "120")),
        ), directory=directory)

    def _payload(self, request: Request) -> dict:
        contract = request.context.get("contract")
        field = {"training-program": "source", "state-plan": "actions"}.get(contract)
        schema = PROGRAM_SCHEMA if field is None else {
            "type": "object", "properties": {field: PROGRAM_SCHEMA["properties"][field]},
            "required": [field], "additionalProperties": False,
        }
        if contract == "state-plan" and "offered_paths" in request.context:
            paths = request.context["offered_paths"]
            if not paths:
                raise ValueError("A legal path catalogue must not be empty")
            schema["properties"]["actions"] = {"type": "array", "items": {"type": "string"}, "enum": paths}
        if contract == "state-goal-plan":
            from mirage.generator.state_plan import planning_schema
            schema = planning_schema(request.context.get("goal_catalog", []))
        elif contract == "region-patches":
            from mirage.generator.mutation import PATCH_SCHEMA
            schema = PATCH_SCHEMA
        # Stable API evidence precedes per-attempt goals and patches for prefix caching.
        context = dict(request.context)
        api_context = context.pop('api_context', None)
        prefix = ("Installed API specification:\n" + json.dumps(api_context, sort_keys=True) + "\n") if api_context else ""
        messages = [
            {"role": "system", "content": "Follow the supplied testing contract. Return only the requested JSON object."},
            {"role": "user", "content": prefix + request.prompt + "\nContext:\n" + json.dumps(context)},
        ]
        if request.previous_content is not None:
            messages.extend([
                {"role": "assistant", "content": request.previous_content},
                {"role": "user", "content": "Repair this diagnosed problem only:\n" + (request.diagnostic or "")},
            ])
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "mirage_program", "strict": True, "schema": schema},
            },
        }
        seed = request.context.get("configuration", {}).get("seed")
        if seed is not None:
            payload["seed"] = seed
        return payload

    def generate(self, request: Request, *, deadline: float | None) -> Response:
        started = time.monotonic()
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        expires = min(deadline, started + self.config.timeout) if deadline is not None else started + self.config.timeout
        if expires <= started:
            raise TimeoutError("HTTP deadline expired before dispatch")
        payload = self._payload(request)
        token_budget = None
        if request.context.get("api_context", {}).get("spec_version") == "api-spec-1":
            from urllib.request import Request as URLRequest, urlopen
            endpoint = self.config.base_url.rstrip("/").removesuffix("/v1") + "/tokenize"
            query = {"model": self.config.model, "messages": payload["messages"]}
            try:
                with urlopen(URLRequest(endpoint, data=json.dumps(query).encode(),
                                        headers={"Content-Type": "application/json"}),
                             timeout=min(5., max(.01, expires-time.monotonic()))) as response:
                    counts = json.load(response)
                available = int(counts["max_model_len"]) - int(counts["count"])
                if available < 256:
                    raise ValueError("Insufficient output context for a program")
                payload["max_tokens"] = min(self.config.max_tokens, available)
                token_budget = dict(prompt_tokens=counts["count"], context_limit=counts["max_model_len"],
                                    output_cap=payload["max_tokens"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ProviderFailure("context_budget", str(error)) from error
        request_id = uuid.uuid4().hex
        evidence = {
            "request_id": request_id, "endpoint": self.config.base_url,
            "request": payload, "token_budget": token_budget, "response": None, "http_status": None,
            "cancellation": "not_requested",
        }
        artifact = self.directory / f"http-{request_id}.json" if self.directory else None

        def save():
            if artifact:
                artifact.write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")

        save()
        url = urlsplit(self.config.base_url)
        connection_class = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        connection = connection_class(url.hostname, url.port, timeout=expires - started)
        sent = False
        try:
            sent = True
            connection.request("POST", url.path.rstrip("/") + "/chat/completions",
                               body=json.dumps(payload).encode("utf-8"),
                               headers={"Content-Type": "application/json", "X-Request-Id": request_id})
            response = connection.getresponse()
            evidence["http_status"] = response.status
            chunks = []
            total = 0
            while True:
                remaining = expires - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("HTTP response deadline expired")
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise ProviderFailure("protocol", "HTTP response exceeded the size limit",
                                          cancellation_uncertain=True)
            evidence["response"] = b"".join(chunks).decode("utf-8", errors="replace")
            if response.status != 200:
                raise ProviderFailure("http", f"Service returned HTTP {response.status}")
            try:
                value = json.loads(evidence["response"])
                if value["model"] != self.config.model or len(value["choices"]) != 1:
                    raise ValueError("Unexpected model or number of choices")
                choice = value["choices"][0]
                content = choice["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("Expected text content")
                if choice["finish_reason"] != "stop":
                    raise ProviderFailure("incomplete", f"Completion ended with {choice['finish_reason']}")
                usage = value.get("usage") or {}
                tokens = Usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))
            except (KeyError, TypeError, ValueError, IndexError) as error:
                raise ProviderFailure("protocol", f"Malformed completion: {error}") from error
            evidence["elapsed_seconds"] = time.monotonic() - started
            return Response(content, tokens, value["model"], dict(evidence))
        except (TimeoutError, OSError, http.client.HTTPException) as error:
            category = "timeout" if isinstance(error, TimeoutError) else "connection"
            evidence.update(error=str(error), cancellation="unconfirmed" if sent else "not_sent")
            raise ProviderFailure(category, str(error), cancellation_uncertain=sent) from error
        except ProviderFailure as error:
            evidence.update(error=str(error), category=error.category)
            if error.cancellation_uncertain:
                evidence["cancellation"] = "unconfirmed"
            raise
        finally:
            connection.close()
            evidence["elapsed_seconds"] = time.monotonic() - started
            save()
