"""Minimal MCP stdio client for the Semantica server.

Speaks newline-delimited JSON-RPC 2.0 to ``semantica-mcp`` over a subprocess
pipe using only the standard library — semantica's multi-gigabyte dependency
tree (torch, transformers, opencv, ...) never enters SWARM's environment.
"""

import json
import logging
import select
import subprocess
import time
from typing import Any, Dict, Optional

from swarm.bridges.semantica.config import SemanticaConfig

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"


class SemanticaMCPError(RuntimeError):
    pass


def encode_request(request_id: Optional[int], method: str, params: Dict[str, Any]) -> bytes:
    """Frame one JSON-RPC message (notification when request_id is None)."""
    msg: Dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
    if request_id is not None:
        msg["id"] = request_id
    return (json.dumps(msg) + "\n").encode("utf-8")


def decode_tool_result(response: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap a tools/call response into the tool's dict payload.

    MCP servers return content blocks; Semantica puts a JSON object in the
    first text block. Fall back to the raw result for structured variants.
    """
    if "error" in response:
        raise SemanticaMCPError(f"MCP error: {response['error']}")
    result = response.get("result")
    if not isinstance(result, dict):
        raise SemanticaMCPError(f"malformed tools/call result: {result!r}")
    if result.get("isError"):
        raise SemanticaMCPError(f"tool reported error: {result}")
    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                try:
                    parsed = json.loads(block.get("text", ""))
                except (TypeError, ValueError):
                    return {"text": block.get("text", "")}
                return parsed if isinstance(parsed, dict) else {"value": parsed}
    return result


class SemanticaMCPClient:
    """Synchronous client around one semantica-mcp subprocess."""

    def __init__(self, cfg: Optional[SemanticaConfig] = None):
        self.cfg = cfg or SemanticaConfig()
        self._proc: Optional[subprocess.Popen] = None
        self._next_id = 1

    def __enter__(self) -> "SemanticaMCPClient":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def start(self) -> None:
        if self._proc is not None:
            return
        try:
            self._proc = subprocess.Popen(
                list(self.cfg.mcp_command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            raise SemanticaMCPError(
                f"could not start {self.cfg.mcp_command!r}: {e}. "
                "Install semantica in some environment and point "
                "SemanticaConfig.mcp_command at its semantica-mcp binary."
            ) from e
        self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": self.cfg.client_name, "version": "1.0"},
            },
        )
        self._notify("notifications/initialized", {})

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    def record_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Push one mapper-produced decision; strips non-MCP keys."""
        args = {
            k: v
            for k, v in decision.items()
            if k in self.cfg.mcp_arg_keys and v is not None
        }
        return self.call_tool("record_decision", args)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        response = self._request(
            "tools/call", {"name": name, "arguments": arguments}
        )
        return decode_tool_result(response)

    # ------------------------------------------------------------------

    def _stdio(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            raise SemanticaMCPError("semantica-mcp process is not running")
        return self._proc

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        proc = self._stdio()
        assert proc.stdin is not None
        proc.stdin.write(encode_request(None, method, params))
        proc.stdin.flush()

    def _request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        proc = self._stdio()
        assert proc.stdin is not None and proc.stdout is not None
        request_id = self._next_id
        self._next_id += 1
        proc.stdin.write(encode_request(request_id, method, params))
        proc.stdin.flush()

        deadline = time.monotonic() + self.cfg.request_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SemanticaMCPError(f"{method}: timeout waiting for response")
            ready, _, _ = select.select([proc.stdout], [], [], remaining)
            if not ready:
                continue
            line = proc.stdout.readline()
            if not line:
                raise SemanticaMCPError(f"{method}: server closed stdout")
            try:
                msg = json.loads(line)
            except ValueError:
                logger.debug("skipping non-JSON stdout line: %r", line[:120])
                continue
            if msg.get("id") == request_id:
                return dict(msg)
            # Server-initiated notification/log — ignore and keep waiting.
            logger.debug("skipping message id=%s awaiting id=%s", msg.get("id"), request_id)
