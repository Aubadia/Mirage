"""Request-specific terminal receipts for a Mirage-owned vLLM service."""
import asyncio
import os
from pathlib import Path
import re


class RequestReceipts:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get('path') != '/v1/chat/completions':
            return await self.app(scope, receive, send)
        key = dict(scope['headers']).get(b'x-request-id', b'')
        if not re.fullmatch(b'[a-f0-9]{32}', key):
            return await self.app(scope, receive, send)
        try:
            return await self.app(scope, receive, send)
        finally:
            # The request handler has returned: no subsequent submissions for this ID.
            # Wait for any already-submitted engine work to leave the output processor.
            engine = getattr(scope['app'].state, 'engine_client', None)
            registry = getattr(getattr(engine, 'output_processor', None), 'external_req_ids', None)
            if registry is not None:
                tag = 'chatcmpl-' + key.decode()
                while any(x == tag or x.startswith(tag + '-') for x in registry):
                    await asyncio.sleep(.1)
                (Path(os.environ['MIRAGE_REQUEST_RECEIPTS']) / key.decode()).touch()
