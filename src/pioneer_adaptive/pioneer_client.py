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
        payload = self._request("GET", "/models")
        return payload.get("data", [])

    def upload_file(self, path: str | Path, purpose: str = "fine-tune") -> str:
        upload_url = f"{self.base_url}/files"
        with Path(path).open("rb") as handle:
            response = requests.post(
                upload_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": handle},
                data={"purpose": purpose},
                timeout=self.timeout,
            )
        if response.status_code >= 400:
            raise PioneerAPIError(
                f"POST {upload_url} failed ({response.status_code}): {response.text}"
            )
        payload = response.json()
        file_id = payload.get("id")
        if not file_id:
            raise PioneerAPIError(f"Upload did not return file id: {payload}")
        return file_id

    def create_finetune_job(
        self,
        base_model: str,
        training_file_id: str,
        *,
        validation_file_id: str | None = None,
        suffix: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": base_model,
            "training_file": training_file_id,
        }
        if validation_file_id:
            body["validation_file"] = validation_file_id
        if suffix:
            body["suffix"] = suffix
        if hyperparameters:
            body["hyperparameters"] = hyperparameters
        return self._request("POST", "/fine_tuning/jobs", json_body=body)

    def get_finetune_job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/fine_tuning/jobs/{job_id}")

    def wait_for_finetune_job(
        self, job_id: str, *, poll_seconds: int = 30, max_wait_seconds: int = 7200
    ) -> dict[str, Any]:
        elapsed = 0
        while elapsed <= max_wait_seconds:
            job = self.get_finetune_job(job_id)
            status = str(job.get("status", "")).lower()
            if status in {"succeeded", "failed", "cancelled"}:
                return job
            time.sleep(poll_seconds)
            elapsed += poll_seconds
        raise PioneerAPIError(
            f"Timed out waiting for fine-tune job {job_id} after {max_wait_seconds}s"
        )

