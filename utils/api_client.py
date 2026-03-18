import os
import time
import json
import httpx
from datetime import datetime
from typing import Dict, Optional
from utils.types import ProviderConfig, ProviderType, CompletionResult, TokenInfo


class ModelClient:
    """
    Unified API client for OpenAI-compatible and Anthropic APIs.
    Supports both standard and streaming completions with latency metrics.
    """

    def __init__(self, config: ProviderConfig, model: str):
        self.config = config
        self.model = model
        self._api_key: Optional[str] = None

    @property
    def api_key(self) -> str:
        if self._api_key is None:
            self._api_key = os.environ.get(self.config.api_key_env)
            if not self._api_key:
                raise ValueError(
                    f"API key not found in environment variable: {self.config.api_key_env}"
                )
        return self._api_key

    @property
    def supports_logprobs(self) -> bool:
        """Check if the provider supports logprobs (OpenAI-compatible only)."""
        return self.config.type in [ProviderType.OPENAI, ProviderType.OPENAI_COMPATIBLE]

    def _get_headers(self) -> Dict[str, str]:
        if self.config.type in [ProviderType.OPENAI, ProviderType.OPENAI_COMPATIBLE]:
            return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        elif self.config.type == ProviderType.ANTHROPIC:
            return {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        return {}

    def complete(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024, **kwargs
    ) -> CompletionResult:
        if self.config.type in [ProviderType.OPENAI, ProviderType.OPENAI_COMPATIBLE]:
            return self._call_openai_compatible(
                prompt, temperature, max_tokens, stream=False, **kwargs
            )
        elif self.config.type == ProviderType.ANTHROPIC:
            return self._call_anthropic(prompt, temperature, max_tokens, stream=False)
        else:
            raise ValueError(f"Unsupported provider type: {self.config.type}")

    def complete_streaming(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024
    ) -> CompletionResult:
        if self.config.type in [ProviderType.OPENAI, ProviderType.OPENAI_COMPATIBLE]:
            return self._call_openai_compatible(prompt, temperature, max_tokens, stream=True)
        elif self.config.type == ProviderType.ANTHROPIC:
            return self._call_anthropic(prompt, temperature, max_tokens, stream=True)
        else:
            raise ValueError(f"Unsupported provider type: {self.config.type}")

    def _call_openai_compatible(
        self, prompt: str, temperature: float, max_tokens: int, stream: bool = False, **kwargs
    ) -> CompletionResult:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = self._get_headers()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }

        if not stream and kwargs.get("logprobs") and self.supports_logprobs:
            payload["logprobs"] = True
            if "top_logprobs" not in payload:
                payload["top_logprobs"] = 5

        start_time = time.time()

        if not stream:
            response = self._make_request("POST", url, headers, payload)
            latency_ms = (time.time() - start_time) * 1000
            data = response.json()

            choice = data["choices"][0]
            text = choice["message"]["content"]
            usage = data.get("usage", {})

            logprobs = []
            if "logprobs" in choice and choice["logprobs"] and "content" in choice["logprobs"]:
                for lp in choice["logprobs"]["content"]:
                    top_lp = {item["token"]: item["logprob"] for item in lp.get("top_logprobs", [])}
                    logprobs.append(
                        TokenInfo(token=lp["token"], logprob=lp["logprob"], top_logprobs=top_lp)
                    )

            return CompletionResult(
                text=text,
                model=data.get("model", self.model),
                latency_ms=latency_ms,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                logprobs=logprobs,
                finish_reason=choice.get("finish_reason"),
                raw_response=data,
            )
        else:
            ttft_ms = None
            first_token_time = None
            full_text = ""
            chunks_count = 0
            finish_reason = None

            def perform_stream():
                nonlocal ttft_ms, first_token_time, full_text, chunks_count, finish_reason
                with httpx.stream(
                    "POST", url, headers=headers, json=payload, timeout=self.config.timeout
                ) as response:
                    if response.status_code == 429:
                        return "RETRY"
                    if response.status_code == 401:
                        raise ValueError(f"Authentication error (401): {response.text}")
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            if ttft_ms is None:
                                ttft_ms = (time.time() - start_time) * 1000
                                first_token_time = time.time()

                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data_str)
                                if not chunk.get("choices"):
                                    continue
                                choice = chunk["choices"][0]
                                delta = choice.get("delta", {})
                                if "content" in delta and delta["content"]:
                                    full_text += delta["content"]
                                    chunks_count += 1
                                if choice.get("finish_reason"):
                                    finish_reason = choice["finish_reason"]
                            except json.JSONDecodeError:
                                continue
                return "OK"

            for attempt in range(self.config.max_retries + 1):
                try:
                    res = perform_stream()
                    if res == "OK":
                        break
                    if res == "RETRY" and attempt < self.config.max_retries:
                        time.sleep(2**attempt)
                        continue
                except (httpx.TimeoutException, httpx.NetworkError):
                    if attempt < self.config.max_retries:
                        time.sleep(2**attempt)
                        continue
                    raise

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            tps = None
            if chunks_count > 0 and first_token_time:
                duration_after_first = end_time - first_token_time
                if duration_after_first > 0:
                    tps = chunks_count / duration_after_first

            return CompletionResult(
                text=full_text,
                model=self.model,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                tps=tps,
                finish_reason=finish_reason,
            )

    def _call_anthropic(
        self, prompt: str, temperature: float, max_tokens: int, stream: bool = False
    ) -> CompletionResult:
        url = f"{self.config.base_url.rstrip('/')}/v1/messages"
        headers = self._get_headers()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        start_time = time.time()

        if not stream:
            response = self._make_request("POST", url, headers, payload)
            latency_ms = (time.time() - start_time) * 1000
            data = response.json()

            content_blocks = data.get("content", [])
            if content_blocks and len(content_blocks) > 0:
                text = content_blocks[0].get("text", "")
            else:
                text = ""
            usage = data.get("usage", {})

            return CompletionResult(
                text=text,
                model=data.get("model", self.model),
                latency_ms=latency_ms,
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
                total_tokens=(usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
                finish_reason=data.get("stop_reason"),
                raw_response=data,
            )
        else:
            ttft_ms = None
            first_token_time = None
            full_text = ""
            chunks_count = 0
            finish_reason = None
            prompt_tokens = None
            completion_tokens = None

            def perform_stream():
                nonlocal \
                    ttft_ms, \
                    first_token_time, \
                    full_text, \
                    chunks_count, \
                    finish_reason, \
                    prompt_tokens, \
                    completion_tokens
                with httpx.stream(
                    "POST", url, headers=headers, json=payload, timeout=self.config.timeout
                ) as response:
                    if response.status_code == 429:
                        return "RETRY"
                    if response.status_code == 401:
                        raise ValueError(f"Authentication error (401): {response.text}")
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                event = json.loads(data_str)
                                event_type = event.get("type")

                                if event_type == "message_start":
                                    prompt_tokens = (
                                        event.get("message", {})
                                        .get("usage", {})
                                        .get("input_tokens")
                                    )
                                elif event_type == "content_block_delta":
                                    if ttft_ms is None:
                                        ttft_ms = (time.time() - start_time) * 1000
                                        first_token_time = time.time()

                                    delta = event.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        full_text += delta.get("text", "")
                                        chunks_count += 1
                                elif event_type == "message_delta":
                                    finish_reason = event.get("delta", {}).get("stop_reason")
                                    usage = event.get("usage", {})
                                    if usage.get("output_tokens"):
                                        completion_tokens = usage.get("output_tokens")
                            except json.JSONDecodeError:
                                continue
                return "OK"

            for attempt in range(self.config.max_retries + 1):
                try:
                    res = perform_stream()
                    if res == "OK":
                        break
                    if res == "RETRY" and attempt < self.config.max_retries:
                        time.sleep(2**attempt)
                        continue
                except (httpx.TimeoutException, httpx.NetworkError):
                    if attempt < self.config.max_retries:
                        time.sleep(2**attempt)
                        continue
                    raise

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            tps = None
            if chunks_count > 0 and first_token_time:
                duration_after_first = end_time - first_token_time
                if duration_after_first > 0:
                    tps = chunks_count / duration_after_first

            return CompletionResult(
                text=full_text,
                model=self.model,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                tps=tps,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
                finish_reason=finish_reason,
            )

    def _make_request(self, method: str, url: str, headers: dict, payload: dict):
        max_retries = self.config.max_retries
        for attempt in range(max_retries + 1):
            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    response = client.request(method, url, headers=headers, json=payload)

                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    if attempt < max_retries:
                        time.sleep(2**attempt)
                        continue
                elif response.status_code == 401:
                    raise ValueError(f"Authentication error (401): {response.text}")

                response.raise_for_status()
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt < max_retries:
                    time.sleep(2**attempt)
                    continue
                raise
        raise Exception(f"Failed after {max_retries} retries")
