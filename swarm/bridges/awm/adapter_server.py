"""AWM Adapter Server — bridges AWM-1K FastAPI apps to SWARM's MCP endpoints.

Loads an AWM scenario's ``full_code`` from ``gen_envs.jsonl``, ``exec()``s it to
obtain the upstream FastAPI ``app``, and re-exposes it through the endpoints that
``AWMMCPSyncClient`` expects (``/tools``, ``/tools/call``, ``/reset``, ``/verify``,
``/task``, ``/transaction/*``).

Usage::

    python -m swarm.bridges.awm.adapter_server \
        --scenario content_platform_1 \
        --envs-jsonl data/awm-1k/gen_envs.jsonl \
        --db-dir external/awm-envs/outputs/databases \
        --host 127.0.0.1 --port 9100 \
        --data-path data/awm-1k
"""

import argparse
import inspect
import json
import logging
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Conservative character whitelists for SSRF-safe path dispatch
_ALLOWED_PATH_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "/._-~"
)
_ALLOWED_QUERY_CHARS = _ALLOWED_PATH_CHARS | frozenset("=&+%@!$'()*,;:")

# Built-in FastAPI routes to skip during introspection
_SKIP_BUILTIN_ROUTES = frozenset({"openapi", "swagger_ui_html", "swagger_ui_redirect", "redoc_html"})


def _sanitize_dispatch_path(normalized: str, sep: str, query: str) -> Optional[str]:
    """Validate and rebuild the dispatch path from whitelisted characters.

    Returns the sanitized path string, or ``None`` if any character fails
    the whitelist.  Each character is validated and appended to a fresh list
    inside the same loop, so the output is built entirely from vetted
    primitives — breaking any taint chain from user-supplied input.
    """
    # Validate and rebuild the path component from allowed characters only.
    safe_path_chars: list[str] = []
    for ch in normalized:
        if ch not in _ALLOWED_PATH_CHARS:
            return None
        safe_path_chars.append(ch)

    # Validate and rebuild the query component (if any) from allowed characters only.
    safe_query_chars: list[str] = []
    if query:
        for ch in query:
            if ch not in _ALLOWED_QUERY_CHARS:
                return None
            safe_query_chars.append(ch)

    safe_path = "".join(safe_path_chars)
    safe_query = "".join(safe_query_chars)
    return safe_path + sep + safe_query


def _is_safe_dispatch_path(dispatch_path: str) -> bool:
    """Final structural validation for the dispatch path before HTTP requests.

    Runs after character-level sanitization to enforce that the path is a
    simple absolute path under the current service — not a full/scheme-relative
    URL, and free of traversal or ambiguous segments.
    """
    if not dispatch_path.startswith("/"):
        return False
    if dispatch_path.startswith("//"):
        return False
    if "://" in dispatch_path:
        return False
    if ".." in dispatch_path:
        return False
    if "//" in dispatch_path or "/./" in dispatch_path:
        return False
    # Belt-and-suspenders: urlparse must see no scheme or netloc
    parsed = urlparse(dispatch_path)
    if parsed.scheme or parsed.netloc:
        return False
    # Enforce that the path component matches the whitelist structurally
    path_only, _, _ = dispatch_path.partition("?")
    if not re.fullmatch(r"/[A-Za-z0-9/_.\-~]*", path_only):
        return False
    return True


def _validate_tool_base_path(raw_path: str) -> str:
    """Validate and sanitize a tool's configured base path at startup.

    Raises ``ValueError`` if the path is structurally invalid so that
    ``build_adapter`` can fail fast rather than serving a broken tool.
    Returns the sanitized, normalized base path string on success.
    """
    base_path_only, base_sep, base_query = raw_path.partition("?")
    base_normalized = posixpath.normpath(base_path_only)
    if base_path_only.endswith("/") and base_normalized != "/":
        base_normalized += "/"
    if not base_normalized.startswith("/"):
        raise ValueError(
            f"Configured tool path {raw_path!r} must start at the application root."
        )
    sanitized = _sanitize_dispatch_path(base_normalized, base_sep, base_query)
    if sanitized is None:
        raise ValueError(
            f"Configured tool path {raw_path!r} contains invalid characters."
        )
    return sanitized


def _load_scenario_code(envs_jsonl: Path, scenario: str) -> str:
    """Return the ``full_code`` string for *scenario* from *envs_jsonl*."""
    with open(envs_jsonl) as fh:
        for line in fh:
            entry = json.loads(line)
            if entry["scenario"] == scenario:
                return str(entry["full_code"])
    raise ValueError(f"Scenario {scenario!r} not found in {envs_jsonl}")


def _load_tasks(tasks_jsonl: Path, scenario: str) -> List[str]:
    """Return task descriptions for *scenario* from ``gen_tasks.jsonl``."""
    with open(tasks_jsonl) as fh:
        for line in fh:
            entry = json.loads(line)
            if entry["scenario"] == scenario:
                return list(entry["tasks"])
    return []


def _load_verifier_code(verifier_jsonl: Path, scenario: str, task_idx: int) -> Optional[str]:
    """Return the verifier ``code`` string for *scenario* / *task_idx*."""
    with open(verifier_jsonl) as fh:
        for line in fh:
            entry = json.loads(line)
            if entry["scenario"] == scenario and entry.get("task_idx") == task_idx:
                return str(entry["verification"]["code"])
    return None


_AWM_CODE_BLOCKLIST = re.compile(
    r"""
    \bsubprocess\b
    | \bos\.system\b
    | \bos\.popen\b
    | \bos\.exec
    | \bctypes\b
    | \bsocket\b
    | \bshutil\.rmtree\b
    | \bos\.remove\b
    | \bos\.unlink\b
    | \beval\s*\(
    | \bcompile\s*\(
    """,
    re.VERBOSE,
)

# Safe subset of builtins for exec'd AWM code.  Excludes dangerous
# introspection/execution primitives while keeping everything needed
# for typical FastAPI app code.
_RESTRICTED_BUILTINS: Dict[str, Any] = {
    # types & constructors
    "bool": bool, "int": int, "float": float, "complex": complex,
    "str": str, "bytes": bytes, "bytearray": bytearray,
    "list": list, "tuple": tuple, "dict": dict, "set": set, "frozenset": frozenset,
    "type": type, "object": object,
    # iteration & functional
    "range": range, "enumerate": enumerate, "zip": zip, "map": map,
    "filter": filter, "reversed": reversed, "sorted": sorted,
    "iter": iter, "next": next,
    # math & comparisons
    "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
    "pow": pow, "divmod": divmod,
    # string & repr
    "repr": repr, "ascii": ascii, "chr": chr, "ord": ord,
    "format": format, "hash": hash, "id": id,
    # type checks
    "isinstance": isinstance, "issubclass": issubclass, "callable": callable,
    "len": len,
    # attribute access
    "getattr": getattr, "setattr": setattr, "delattr": delattr, "hasattr": hasattr,
    # collections helpers
    "any": any, "all": all,
    # I/O — needed for SQLite and file access within AWM apps
    "open": open, "print": print, "input": input,
    # exceptions
    "Exception": Exception, "TypeError": TypeError, "ValueError": ValueError,
    "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
    "RuntimeError": RuntimeError, "StopIteration": StopIteration,
    "FileNotFoundError": FileNotFoundError, "IOError": IOError,
    "OSError": OSError, "ImportError": ImportError, "NotImplementedError": NotImplementedError,
    # misc
    "__import__": __import__,
    "__name__": "__main__",
    "__build_class__": __build_class__,
    "property": property, "staticmethod": staticmethod, "classmethod": classmethod,
    "super": super,
    "vars": vars, "dir": dir,
    "None": None, "True": True, "False": False,
    "Ellipsis": Ellipsis, "NotImplemented": NotImplemented,
    "slice": slice, "memoryview": memoryview,
}


def _validate_awm_code(code: str) -> None:
    """Static blocklist scan — raises ``ValueError`` on dangerous patterns."""
    match = _AWM_CODE_BLOCKLIST.search(code)
    if match:
        raise ValueError(
            f"AWM code contains blocked pattern: {match.group(0)!r}"
        )


def _exec_awm_app(code: str, db_url: str) -> Any:
    """``exec()`` the AWM full_code and return the FastAPI ``app`` object.

    We override ``DATABASE_PATH`` env-var so the AWM code's
    ``os.getenv("DATABASE_PATH", ...)`` picks up our working DB.

    A synthetic module is created so that Pydantic v2 can resolve
    forward-reference type annotations (``List``, ``Optional``, etc.)
    inside the exec'd namespace.

    Two defense layers are applied before exec:
    1. Static blocklist scan (``_validate_awm_code``)
    2. Restricted ``__builtins__`` dict (no ``exec``, ``eval``, ``compile``,
       ``globals``, ``locals``, ``breakpoint``, ``exit``, ``quit``)
    """
    import types

    _validate_awm_code(code)

    os.environ["DATABASE_PATH"] = db_url

    # Create a real module so Pydantic's model_rebuild() can locate types
    mod = types.ModuleType(f"_awm_env_{id(code)}")
    mod.__file__ = "<awm-env>"
    sys.modules[mod.__name__] = mod

    # Restrict builtins available inside the exec'd code
    mod.__dict__["__builtins__"] = dict(_RESTRICTED_BUILTINS)

    exec(code, mod.__dict__)  # noqa: S102 — AWM-1K code with restricted builtins

    app = getattr(mod, "app", None)
    if app is None:
        raise RuntimeError("AWM full_code did not define an `app` variable")

    # Rebuild any Pydantic models so forward refs resolve against the module
    from pydantic import BaseModel as _BM

    for obj in mod.__dict__.values():
        if isinstance(obj, type) and issubclass(obj, _BM) and obj is not _BM:
            try:
                obj.model_rebuild(_types_namespace=mod.__dict__)  # type: ignore[attr-defined]
            except Exception:
                pass

    return app


def _run_verifier_subprocess(
    code: str, initial_db: str, working_db: str, *, timeout: int = 10
) -> Dict[str, Any]:
    """Run verifier code in an isolated subprocess and return its result dict.

    The child process ``exec()``s the verifier code, calls
    ``verify_task_completion(initial_db, working_db)``, and writes the
    result as JSON to stdout.  If the process times out or crashes, an
    ``{"error": ...}`` dict is returned.
    """
    wrapper = textwrap.dedent("""\
        import json, sys
        _payload = json.loads(sys.stdin.read())
        _code = _payload["code"]
        _initial = _payload["initial_db"]
        _working = _payload["working_db"]
        _ns = {}
        exec(_code, _ns)
        _fn = _ns.get("verify_task_completion")
        if _fn is None:
            json.dump({"error": "verify_task_completion not found"}, sys.stdout)
        else:
            try:
                _result = _fn(_initial, _working)
                json.dump({"result": _result}, sys.stdout)
            except Exception as _exc:
                json.dump({"error": str(_exc)}, sys.stdout)
    """)

    payload = json.dumps({
        "code": code,
        "initial_db": initial_db,
        "working_db": working_db,
    })

    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapper],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": "verifier timed out"}

    if proc.returncode != 0:
        stderr_snippet = (proc.stderr or "")[:500]
        return {"error": f"verifier process failed (rc={proc.returncode}): {stderr_snippet}"}

    try:
        parsed: Dict[str, Any] = json.loads(proc.stdout)
        return parsed
    except (json.JSONDecodeError, ValueError):
        return {"error": f"verifier produced invalid JSON: {(proc.stdout or '')[:200]}"}


def _extract_tools_from_app(awm_app: Any) -> List[Dict[str, Any]]:
    """Introspect an AWM FastAPI app's routes and return MCP-style tool dicts."""
    tools: List[Dict[str, Any]] = []
    for route in awm_app.routes:
        if not hasattr(route, "endpoint"):
            continue
        endpoint = route.endpoint
        name = getattr(route, "operation_id", None) or endpoint.__name__
        if name in _SKIP_BUILTIN_ROUTES:
            continue
        description = getattr(route, "description", "") or ""
        summary = getattr(route, "summary", "") or ""

        # Build a minimal inputSchema from the endpoint's signature
        sig = inspect.signature(endpoint)
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for pname, param in sig.parameters.items():
            if pname in ("self", "db"):
                continue
            prop: Dict[str, Any] = {"type": "string"}
            if param.annotation is not inspect.Parameter.empty:
                # Try to get a more specific type
                ann = param.annotation
                if ann is int:
                    prop = {"type": "integer"}
                elif ann is float:
                    prop = {"type": "number"}
                elif ann is bool:
                    prop = {"type": "boolean"}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
            properties[pname] = prop

        methods = list(getattr(route, "methods", {"GET"}))
        tools.append(
            {
                "name": name,
                "description": summary or description,
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
                "method": methods[0] if methods else "GET",
                "path": getattr(route, "path", "/"),
            }
        )
    return tools


def build_adapter(
    *,
    scenario: str,
    envs_jsonl: Path,
    db_dir: Path,
    data_path: Path,
    task_idx: int = 0,
) -> Any:
    """Build and return the adapter FastAPI application."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    # ── Resolve file paths ───────────────────────────────────────────
    tasks_jsonl = data_path / "gen_tasks.jsonl"
    verifier_jsonl = data_path / "gen_verifier.pure_code.jsonl"

    source_db = db_dir / f"{scenario}.db"
    if not source_db.exists():
        raise FileNotFoundError(f"Database not found: {source_db}")

    # Working copy so /reset can restore from the original
    work_dir = Path(tempfile.mkdtemp(prefix=f"awm-{scenario}-"))
    initial_db = work_dir / "initial.db"
    working_db = work_dir / "working.db"
    shutil.copy2(source_db, initial_db)
    shutil.copy2(source_db, working_db)

    db_url = f"sqlite:///{working_db}"

    # ── Load AWM app ─────────────────────────────────────────────────
    awm_code = _load_scenario_code(envs_jsonl, scenario)
    awm_app = _exec_awm_app(awm_code, db_url)
    tools_list = _extract_tools_from_app(awm_app)

    # Build lookup: tool name → (method, sanitized base path)
    # Paths are validated and sanitized here at startup so that misconfigured
    # tools are caught immediately rather than during each invocation.
    tool_meta: Dict[str, Dict[str, str]] = {}
    for route in awm_app.routes:
        if not hasattr(route, "endpoint"):
            continue
        op_id = getattr(route, "operation_id", None) or route.endpoint.__name__
        if op_id in _SKIP_BUILTIN_ROUTES:
            continue
        raw_path = getattr(route, "path", "/")
        try:
            sanitized_base = _validate_tool_base_path(raw_path)
        except ValueError as exc:
            logger.error(
                "Tool '%s' has an invalid configured path %r — skipping. Reason: %s",
                op_id,
                raw_path,
                exc,
            )
            continue
        methods = list(getattr(route, "methods", {"GET"}))
        tool_meta[op_id] = {
            "method": methods[0] if methods else "GET",
            "path": sanitized_base,
        }

    # Load tasks & current task index
    tasks = _load_tasks(tasks_jsonl, scenario) if tasks_jsonl.exists() else []
    current_task_idx = task_idx

    # ── Adapter app ──────────────────────────────────────────────────
    adapter = FastAPI(title=f"AWM Adapter — {scenario}")

    @adapter.get("/tools")
    async def list_tools() -> JSONResponse:
        return JSONResponse({"tools": tools_list})

    @adapter.post("/tools/call")
    async def call_tool(request: Request) -> JSONResponse:
        import httpx

        body = await request.json()
        tool_name = body.get("name", "")
        arguments = body.get("arguments", {})

        if tool_name not in tool_meta:
            return JSONResponse(
                {"isError": True, "result": f"Unknown tool: {tool_name}"},
                status_code=404,
            )

        meta = tool_meta[tool_name]
        method = meta["method"]
        path = meta["path"]

        def _validate_path_param_value(name: str, value: Any) -> str:
            """Validate a path parameter value to prevent traversal/injection."""
            s = str(value)
            if "/" in s or "\\" in s:
                raise ValueError(f"Invalid path parameter '{name}': unexpected '/' or '\\\\'")
            if "?" in s or "#" in s:
                raise ValueError(
                    f"Invalid path parameter '{name}': unexpected query/fragment delimiter"
                )
            if "\n" in s or "\r" in s or "\t" in s:
                raise ValueError(f"Invalid path parameter '{name}': unexpected whitespace")
            if s.startswith(".") or ".." in s:
                raise ValueError(f"Invalid path parameter '{name}': potentially unsafe value")
            return s

        # Substitute path parameters from arguments
        rendered_path = path
        path_params_used: set = set()
        for match in re.finditer(r"\{(\w+)\}", path):
            param_name = match.group(1)
            if param_name in arguments:
                try:
                    safe_value = _validate_path_param_value(param_name, arguments[param_name])
                except ValueError as exc:
                    logger.warning(
                        "Invalid path parameter value for tool '%s', param '%s': %s",
                        tool_name,
                        param_name,
                        exc,
                    )
                    return JSONResponse(
                        {
                            "isError": True,
                            "result": "Invalid path parameter value.",
                        },
                        status_code=400,
                    )
                rendered_path = rendered_path.replace(f"{{{param_name}}}", safe_value)
                path_params_used.add(param_name)

        # Validate rendered path is a safe relative path (SSRF protection)
        parsed_path = urlparse(rendered_path)
        if parsed_path.scheme or parsed_path.netloc:
            return JSONResponse(
                {"isError": True, "result": "Absolute URLs are not allowed in tool paths."},
                status_code=400,
            )
        if rendered_path.startswith("//"):
            return JSONResponse(
                {"isError": True, "result": "Scheme-relative URLs are not allowed."},
                status_code=400,
            )
        if not rendered_path.startswith("/"):
            return JSONResponse(
                {"isError": True, "result": "Tool paths must start with '/'."},
                status_code=400,
            )
        if "/../" in rendered_path or rendered_path.endswith("/.."):
            return JSONResponse(
                {"isError": True, "result": "Path traversal is not allowed."},
                status_code=400,
            )

        # Normalize and strictly validate the path component to prevent SSRF/traversal
        path_only, sep, query = rendered_path.partition("?")
        normalized = posixpath.normpath(path_only)
        # Preserve trailing slash (normpath strips it, but FastAPI routes may require it)
        if path_only.endswith("/") and normalized != "/":
            normalized += "/"
        if not normalized.startswith("/"):
            return JSONResponse(
                {"isError": True, "result": "Normalized tool path must stay within root."},
                status_code=400,
            )
        # Validate and rebuild the dispatch path from whitelisted characters.
        # _sanitize_dispatch_path reconstructs the string character-by-character
        # so the result is free of any user-supplied taint.
        dispatch_path = _sanitize_dispatch_path(normalized, sep, query)
        if dispatch_path is None:
            return JSONResponse(
                {
                    "isError": True,
                    "result": "Tool path or query string contains invalid characters.",
                },
                status_code=400,
            )
        # Defensively re-parse to ensure no scheme/netloc slipped through.
        parsed_dispatch = urlparse(dispatch_path)
        if parsed_dispatch.scheme or parsed_dispatch.netloc:
            return JSONResponse(
                {
                    "isError": True,
                    "result": "Tool path must be a relative URL without scheme or host.",
                },
                status_code=400,
            )
        dispatch_relative_path = dispatch_path
        # Final structural validation: ensure the dispatch path is a safe,
        # absolute path under the current service with no SSRF/traversal risk.
        if not _is_safe_dispatch_path(dispatch_relative_path):
            return JSONResponse(
                {
                    "isError": True,
                    "result": "Tool path is not structurally valid.",
                },
                status_code=400,
            )

        # Ensure the final dispatch path stays under the trusted base path.
        # This prevents user input from changing the effective base path used for dispatch.
        base_path_only, base_sep, base_query = path.partition("?")
        # Strip FastAPI-style path template placeholders ({param_name}) before
        # sanitizing, so _sanitize_dispatch_path is not given raw '{' / '}'
        # characters.  For a template like "/users/{user_id}" we anchor against
        # the static prefix "/users/" that precedes the first placeholder.
        base_static = re.split(r"\{[^}]+\}", base_path_only)[0]
        base_normalized = posixpath.normpath(base_static) if base_static else "/"
        if base_static.endswith("/") and base_normalized != "/":
            base_normalized += "/"
        if not base_normalized.startswith("/"):
            return JSONResponse(
                {
                    "isError": True,
                    "result": "Configured tool path must start at the application root.",
                },
                status_code=500,
            )
        base_dispatch_sanitized = _sanitize_dispatch_path(base_normalized, base_sep, base_query)
        if base_dispatch_sanitized is None:
            return JSONResponse(
                {
                    "isError": True,
                    "result": "Configured tool path contains invalid characters.",
                },
                status_code=500,
            )
        base_dispatch_path = base_dispatch_sanitized
        # Require that the runtime dispatch path is either exactly the base path
        # or a strict sub-path of it (path-prefix with '/' boundary).
        # The '/' boundary prevents false prefix matches: "/api/v1" should not match "/api/v1.5/users".
        # For parameterized templates the static prefix is used as the anchor.
        if dispatch_relative_path != base_dispatch_path:
            prefix = base_dispatch_path
            if not prefix.endswith("/"):
                prefix = prefix + "/"
            if not dispatch_relative_path.startswith(prefix):
                return JSONResponse(
                    {
                        "isError": True,
                        "result": "Tool path must remain within its configured base path.",
                    },
                    status_code=400,
                )

        remaining = {k: v for k, v in arguments.items() if k not in path_params_used}

        # Dispatch via httpx ASGITransport directly to the AWM app.
        # The HTTP client is configured with a fixed base_url ("http://awm")
        # routed to the in-process ASGI app; we only ever pass a sanitized,
        # relative path so callers cannot control the destination host or scheme.
        transport = httpx.ASGITransport(app=awm_app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(transport=transport, base_url="http://awm") as client:
            try:
                if method.upper() in ("POST", "PUT", "PATCH"):
                    if method.upper() == "PUT":
                        resp = await client.put(dispatch_relative_path, json=remaining)
                    elif method.upper() == "PATCH":
                        resp = await client.patch(dispatch_relative_path, json=remaining)
                    else:
                        resp = await client.post(dispatch_relative_path, json=remaining)
                elif method.upper() == "DELETE":
                    resp = await client.delete(dispatch_relative_path, params=remaining)
                else:
                    resp = await client.get(dispatch_relative_path, params=remaining)
            except Exception as exc:
                logger.exception("Error while dispatching request to AWM app", exc_info=exc)
                return JSONResponse(
                    {
                        "isError": True,
                        "result": "An internal error occurred while dispatching the tool.",
                    },
                    status_code=500,
                )

        if resp.status_code >= 400:
            return JSONResponse(
                {"isError": True, "result": resp.text},
                status_code=resp.status_code,
            )

        try:
            result = resp.json()
        except Exception:
            result = resp.text

        return JSONResponse({"result": result})

    @adapter.post("/reset")
    async def reset_environment() -> JSONResponse:
        nonlocal awm_app
        shutil.copy2(initial_db, working_db)
        # Re-exec the AWM app so SQLAlchemy picks up the fresh DB
        awm_app = _exec_awm_app(awm_code, db_url)
        # Rebuild tool metadata (should be identical); re-validate paths.
        nonlocal tool_meta
        tool_meta.clear()
        for route in awm_app.routes:
            if not hasattr(route, "endpoint"):
                continue
            op_id = getattr(route, "operation_id", None) or route.endpoint.__name__
            if op_id in _SKIP_BUILTIN_ROUTES:
                continue
            raw_path = getattr(route, "path", "/")
            try:
                sanitized_base = _validate_tool_base_path(raw_path)
            except ValueError as exc:
                logger.error(
                    "Tool '%s' has an invalid configured path %r after reset — skipping. Reason: %s",
                    op_id,
                    raw_path,
                    exc,
                )
                continue
            methods = list(getattr(route, "methods", {"GET"}))
            tool_meta[op_id] = {
                "method": methods[0] if methods else "GET",
                "path": sanitized_base,
            }
        return JSONResponse({"status": "reset"})

    @adapter.post("/verify")
    async def verify_task() -> JSONResponse:
        nonlocal current_task_idx
        if not verifier_jsonl.exists():
            return JSONResponse({"passed": False, "error": "no verifier data"})

        code = _load_verifier_code(verifier_jsonl, scenario, current_task_idx)
        if code is None:
            return JSONResponse({"passed": False, "error": "verifier not found for task"})

        result = _run_verifier_subprocess(code, str(initial_db), str(working_db))

        # Subprocess returned only an error (no "result" key)
        if "error" in result and "result" not in result:
            return JSONResponse({"passed": False, "error": result["error"]})

        inner = result.get("result", {})
        passed = inner.get("result") == "complete" if isinstance(inner, dict) else False
        return JSONResponse({"passed": passed, "confidence": 0.8, "details": inner})

    @adapter.get("/task")
    async def get_task() -> JSONResponse:
        nonlocal current_task_idx
        if not tasks or current_task_idx >= len(tasks):
            return JSONResponse(
                {"task_id": scenario, "description": f"Complete tasks for {scenario}"}
            )
        return JSONResponse(
            {
                "task_id": f"{scenario}_{current_task_idx}",
                "description": tasks[current_task_idx],
            }
        )

    @adapter.post("/transaction/begin")
    async def transaction_begin() -> JSONResponse:
        # SQLite WAL-mode transactions are implicit; this is a no-op stub
        return JSONResponse({"status": "begun"})

    @adapter.post("/transaction/end")
    async def transaction_end(request: Request) -> JSONResponse:
        # Accept {"commit": true/false} but SQLite auto-commits
        return JSONResponse({"status": "ended"})

    return adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="AWM Adapter Server")
    parser.add_argument("--scenario", required=True, help="AWM scenario name")
    parser.add_argument(
        "--envs-jsonl",
        type=Path,
        default=Path("data/awm-1k/gen_envs.jsonl"),
        help="Path to gen_envs.jsonl",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=Path("external/awm-envs/outputs/databases"),
        help="Directory containing .db files",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/awm-1k"),
        help="Path to AWM-1K data directory",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--task-idx", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    app = build_adapter(
        scenario=args.scenario,
        envs_jsonl=args.envs_jsonl,
        db_dir=args.db_dir,
        data_path=args.data_path,
        task_idx=args.task_idx,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
