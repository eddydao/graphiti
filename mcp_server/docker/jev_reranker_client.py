"""Cross-encoder backed by OpenRouter's decisions endpoint (e.g. typesafe/jev-1.13).

Each passage becomes one `noul` (yes/no) question; the returned probability is the
relevance score. Passages are batched into a few requests because per-request latency
is dominated by the gateway, not the model.
"""

import asyncio
import logging

import httpx

from graphiti_core.cross_encoder.client import CrossEncoderClient

logger = logging.getLogger(__name__)

BATCH_SIZE = 20
NEUTRAL_SCORE = 0.5


class JevRerankerClient(CrossEncoderClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = 'https://openrouter.ai/api/v1',
        timeout: float = 30.0,
    ):
        root = base_url.rstrip('/').removesuffix('/v1')
        self.url = f'{root}/alpha/decisions'
        self.model = model
        self.client = httpx.AsyncClient(
            headers={'Authorization': f'Bearer {api_key}'}, timeout=timeout
        )

    async def _score_batch(self, query: str, passages: list[str]) -> list[float]:
        keys = [f'p{i}' for i in range(len(passages))]
        payload = {
            'model': self.model,
            'state': {'query': query, 'passages': dict(zip(keys, passages, strict=True))},
            'questions': {
                key: {
                    'type': 'noul',
                    'instructions': f'Is passage {key} relevant to answering the query?',
                }
                for key in keys
            },
        }
        response = await self.client.post(self.url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f'decisions endpoint returned {response.status_code}: {response.text[:300]}'
            )
        answers = response.json().get('answers', {})
        scores = []
        for key in keys:
            answer = answers.get(key) or {}
            value = answer.get('noul')
            if answer.get('type') != 'noul' or value is None:
                logger.warning('Jev returned no score for %s (%s)', key, answer.get('type'))
                value = NEUTRAL_SCORE
            scores.append(float(value))
        return scores

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []
        batches = [passages[i : i + BATCH_SIZE] for i in range(0, len(passages), BATCH_SIZE)]
        batch_scores = await asyncio.gather(*(self._score_batch(query, b) for b in batches))
        scores = [s for batch in batch_scores for s in batch]
        results = list(zip(passages, scores, strict=True))
        results.sort(reverse=True, key=lambda x: x[1])
        return results
