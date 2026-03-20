from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


class PioneerAPIError(RuntimeError):
    """Raised when Pioneer API returns an error."""


class PioneerClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 90) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    def _request(
        self, method: str, path: str, *, json_body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = requests.request(
            method=method,
            url=url,
            headers=self._headers,
            json=json_body,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise PioneerAPIError(
                f"{method} {url} failed ({response.status_code}): {response.text}"
            )
        payload = response.json()
        if isinstance(payload, dict):
            return payload
        raise PioneerAPIError(f"Unexpected response payload from {url}: {payload!r}")

    def list_models(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/v1/models")
        return payload.get("data", [])

    def model_available_for_inference(self, model_id: str) -> bool:
        models = self.list_models()
        known = {str(model.get("id")) for model in models if model.get("id")}
        return model_id in known

    def chat_completion(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        payload = self._request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise PioneerAPIError(f"Missing chat completion choices: {payload}")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise PioneerAPIError(f"Invalid chat completion content: {payload}")
        return content

    def upload_dataset(
        self,
        path: str | Path,
        *,
        dataset_name: str,
        dataset_type: str = "decoder",
        fmt: str = "jsonl",
    ) -> dict[str, Any]:
        upload_url = f"{self.base_url}/felix/datasets/upload"
        with Path(path).open("rb") as handle:
            response = requests.post(
                upload_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (Path(path).name, handle, "application/jsonl")},
                data={
                    "dataset_name": dataset_name,
                    "dataset_type": dataset_type,
                    "format": fmt,
                },
                timeout=self.timeout,
            )
        if response.status_code >= 400:
            raise PioneerAPIError(
                f"POST {upload_url} failed ({response.status_code}): {response.text}"
            )
        return response.json()

    def list_datasets(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/felix/datasets")
        return payload.get("datasets", [])

    def create_finetune_job(
        self,
        *,
        model_name: str,
        datasets: list[dict[str, str]],
        base_model: str,
        training_type: str = "lora",
        hyperparameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model_name": model_name,
            "datasets": datasets,
            "base_model": base_model,
            "training_type": training_type,
        }
        if hyperparameters:
            body.update(hyperparameters)
        return self._request("POST", "/felix/training-jobs", json_body=body)

    def get_finetune_job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/felix/training-jobs/{job_id}")

    def get_finetune_checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        payload = self._request("GET", f"/felix/training-jobs/{job_id}/checkpoints")
        return payload.get("checkpoints", [])

    def wait_for_finetune_job(
        self, job_id: str, *, poll_seconds: int = 30, max_wait_seconds: int = 7200
    ) -> dict[str, Any]:
        elapsed = 0
        while elapsed <= max_wait_seconds:
            job = self.get_finetune_job(job_id)
            status = str(job.get("status", "")).lower()
            if status in {"complete", "failed", "cancelled", "error"}:
                return job
            time.sleep(poll_seconds)
            elapsed += poll_seconds
        raise PioneerAPIError(
            f"Timed out waiting for fine-tune job {job_id} after {max_wait_seconds}s"
        )

