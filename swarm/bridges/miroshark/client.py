"""HTTP client for the MiroShark backend lifecycle.

Matches the actual Flask handlers in ``backend/app/api/`` (which differ from
the published openapi.yaml — ontology generation is synchronous multipart,
build is async, etc.).
"""

import json
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

import requests

from swarm.bridges.miroshark.config import MirosharkConfig

logger = logging.getLogger(__name__)


class MirosharkAPIError(RuntimeError):
    pass


class MirosharkClient:
    def __init__(self, cfg: MirosharkConfig):
        self.cfg = cfg
        self.session = requests.Session()
        if cfg.admin_token:
            self.session.headers["Authorization"] = f"Bearer {cfg.admin_token}"
        if cfg.internal_key:
            self.session.headers["x-miroshark-internal-key"] = cfg.internal_key

    def _url(self, path: str) -> str:
        return self.cfg.api_url.rstrip("/") + path

    def _envelope(self, r: requests.Response, where: str) -> Dict[str, Any]:
        try:
            payload = r.json()
        except ValueError as e:
            raise MirosharkAPIError(f"{where}: non-JSON response ({r.status_code}): {r.text[:200]}") from e
        if r.status_code >= 400 or (isinstance(payload, dict) and payload.get("success") is False):
            err = payload.get("error") if isinstance(payload, dict) else r.text[:200]
            raise MirosharkAPIError(f"{where} → {r.status_code}: {err}")
        if not isinstance(payload, dict):
            raise MirosharkAPIError(f"{where}: expected JSON object, got {type(payload).__name__}")
        return payload

    def health(self) -> Dict[str, Any]:
        r = self.session.get(self._url("/health"), timeout=self.cfg.request_timeout_s)
        r.raise_for_status()
        body = r.json()
        return body if isinstance(body, dict) else {"raw": body}

    def generate_ontology(
        self,
        simulation_requirement: str,
        seed_text: str,
        seed_title: str = "scenario_seed.md",
        project_name: str = "swarm",
        additional_context: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        url_docs = [{"title": seed_title, "url": f"swarm://{seed_title}", "text": seed_text}]
        form: Dict[str, Any] = {
            "simulation_requirement": simulation_requirement,
            "project_name": project_name,
            "url_docs": json.dumps(url_docs),
        }
        if additional_context:
            form["additional_context"] = additional_context
        r = self.session.post(
            self._url("/api/graph/ontology/generate"),
            data=form,
            timeout=self.cfg.request_timeout_s * 5,
        )
        payload = self._envelope(r, "POST /api/graph/ontology/generate")
        data = payload.get("data") or {}
        pid = data.get("project_id")
        ontology = data.get("ontology") or {}
        if not isinstance(pid, str) or not pid:
            raise MirosharkAPIError(f"no project_id returned: {payload}")
        return pid, ontology if isinstance(ontology, dict) else {}

    def build_graph(
        self,
        project_id: str,
        graph_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        force: bool = False,
    ) -> str:
        body: Dict[str, Any] = {"project_id": project_id, "force": force}
        if graph_name:
            body["graph_name"] = graph_name
        if chunk_size is not None:
            body["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            body["chunk_overlap"] = chunk_overlap
        r = self.session.post(
            self._url("/api/graph/build"), json=body, timeout=self.cfg.request_timeout_s
        )
        payload = self._envelope(r, "POST /api/graph/build")
        data = payload.get("data") or {}
        tid = data.get("task_id") or payload.get("task_id")
        if not isinstance(tid, str) or not tid:
            raise MirosharkAPIError(f"no task_id in build response: {payload}")
        return tid

    def graph_task(self, task_id: str) -> Dict[str, Any]:
        r = self.session.get(
            self._url(f"/api/graph/task/{task_id}"), timeout=self.cfg.request_timeout_s
        )
        return self._envelope(r, f"GET /api/graph/task/{task_id}")

    def wait_graph_task(self, task_id: str) -> Dict[str, Any]:
        def check() -> Dict[str, Any]:
            payload = self.graph_task(task_id)
            data = payload.get("data") or {}
            status = (data.get("status") or "").lower()
            if status in {"completed", "success", "finished", "complete"}:
                return {"_done": True, **data}
            if status in {"failed", "error"}:
                raise MirosharkAPIError(f"graph task failed: {data}")
            return {"progress": data}
        return self._poll(check, label=f"graph task {task_id}")

    def create_simulation(
        self,
        project_id: str,
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        enable_polymarket: bool = False,
    ) -> str:
        r = self.session.post(
            self._url("/api/simulation/create"),
            json={
                "project_id": project_id,
                "enable_twitter": enable_twitter,
                "enable_reddit": enable_reddit,
                "enable_polymarket": enable_polymarket,
            },
            timeout=self.cfg.request_timeout_s,
        )
        payload = self._envelope(r, "POST /api/simulation/create")
        sid = (payload.get("data") or {}).get("simulation_id")
        if not isinstance(sid, str) or not sid:
            raise MirosharkAPIError(f"no simulation_id in create response: {payload}")
        return sid

    def prepare(self, simulation_id: str, force_regenerate: bool = False) -> Dict[str, Any]:
        r = self.session.post(
            self._url("/api/simulation/prepare"),
            json={"simulation_id": simulation_id, "force_regenerate": force_regenerate},
            timeout=self.cfg.request_timeout_s,
        )
        return self._envelope(r, "POST /api/simulation/prepare")

    def prepare_status(self, simulation_id: str) -> Dict[str, Any]:
        r = self.session.post(
            self._url("/api/simulation/prepare/status"),
            json={"simulation_id": simulation_id},
            timeout=self.cfg.request_timeout_s,
        )
        return self._envelope(r, "POST /api/simulation/prepare/status")

    def wait_prepared(self, simulation_id: str) -> Dict[str, Any]:
        def check() -> Dict[str, Any]:
            payload = self.prepare_status(simulation_id)
            data = payload.get("data") or {}
            status = (data.get("status") or data.get("preparation_status") or "").lower()
            if status in {"ready", "completed", "complete", "done"} or data.get("already_prepared"):
                return {"_done": True, **data}
            if status in {"failed", "error"}:
                raise MirosharkAPIError(f"prepare failed: {data}")
            return {"progress": data}
        return self._poll(check, label=f"prepare {simulation_id}")

    def start(
        self,
        simulation_id: str,
        platform: str = "parallel",
        max_rounds: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"simulation_id": simulation_id, "platform": platform, "force": force}
        if max_rounds is not None:
            body["max_rounds"] = max_rounds
        r = self.session.post(
            self._url("/api/simulation/start"), json=body, timeout=self.cfg.request_timeout_s
        )
        return self._envelope(r, "POST /api/simulation/start")

    def run_status(self, simulation_id: str) -> Dict[str, Any]:
        r = self.session.get(
            self._url(f"/api/simulation/{simulation_id}/run-status"),
            timeout=self.cfg.request_timeout_s,
        )
        return self._envelope(r, f"GET /api/simulation/{simulation_id}/run-status")

    def wait_run(self, simulation_id: str) -> Dict[str, Any]:
        def check() -> Dict[str, Any]:
            payload = self.run_status(simulation_id)
            data = payload.get("data") or {}
            status = (data.get("runner_status") or data.get("status") or "").lower()
            if status in {"completed", "finished", "complete"}:
                return {"_done": True, **data}
            if status in {"failed", "error"}:
                raise MirosharkAPIError(f"run failed: {data}")
            return {"round": data.get("current_round"), "status": status}
        return self._poll(check, label=f"run {simulation_id}")

    def export(self, simulation_id: str, fmt: str = "json") -> Any:
        r = self.session.get(
            self._url(f"/api/simulation/{simulation_id}/export"),
            params={"format": fmt},
            timeout=self.cfg.request_timeout_s * 2,
        )
        if r.status_code >= 400:
            raise MirosharkAPIError(f"export → {r.status_code}: {r.text[:200]}")
        return r.json() if fmt == "json" else r.text

    def stop(self, simulation_id: str) -> Dict[str, Any]:
        r = self.session.post(
            self._url("/api/simulation/stop"),
            json={"simulation_id": simulation_id},
            timeout=self.cfg.request_timeout_s,
        )
        return self._envelope(r, "POST /api/simulation/stop")

    def _poll(
        self,
        check: Callable[[], Dict[str, Any]],
        label: str = "task",
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + self.cfg.poll_timeout_s
        last: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            got: Dict[str, Any] = check()
            if got and got.get("_done"):
                return got
            if got:
                last = got
                logger.debug("[%s] still working: %s", label, got)
            time.sleep(self.cfg.poll_interval_s)
        raise MirosharkAPIError(f"{label}: poll timeout after {self.cfg.poll_timeout_s}s; last={last}")
