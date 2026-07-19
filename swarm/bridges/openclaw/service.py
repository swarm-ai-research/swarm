"""FastAPI service for the OpenClaw bridge.

Gracefully handles missing FastAPI dependency.
"""

import logging
from typing import Any, Optional

from swarm.bridges.openclaw.config import ServiceConfig
from swarm.bridges.openclaw.job_queue import JobQueue, JobState, SimulationFn
from swarm.bridges.openclaw.schemas import RunRequest

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


def create_app(
    config: Optional[ServiceConfig] = None,
    simulation_fn: Optional[SimulationFn] = None,
) -> Any:
    """Create a FastAPI app for the OpenClaw service.

    Raises ImportError if FastAPI is not installed.
    """
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the OpenClaw service. "
            "Install with: pip install 'swarm-safety[api]'"
        )

    config = config or ServiceConfig()
    queue = JobQueue(config=config, simulation_fn=simulation_fn)

    app = FastAPI(title="OpenClaw SWARM Service", version="1.0.0")

    # Store queue on app for access in tests
    app.state.queue = queue  # type: ignore[attr-defined]

    @app.post("/runs")
    def submit_run(request: RunRequest) -> JSONResponse:
        job = queue.submit(request.model_dump(exclude_none=True))
        return JSONResponse(
            content={"job_id": job.job_id, "status": job.state.value},
            status_code=202,
        )

    @app.get("/runs/{job_id}")
    def get_run_status(job_id: str) -> JSONResponse:
        job = queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse(content=job.to_status_dict())

    @app.get("/runs/{job_id}/metrics")
    def get_run_metrics(job_id: str) -> JSONResponse:
        job = queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.state != JobState.COMPLETED:
            raise HTTPException(
                status_code=404, detail="Metrics not available (job not completed)"
            )
        return JSONResponse(
            content={
                "job_id": job.job_id,
                "toxicity_rate": job.metrics.get("toxicity_rate", 0.0),
                "quality_gap": job.metrics.get("quality_gap", 0.0),
                "total_welfare": job.metrics.get("total_welfare", 0.0),
                "interactions_count": job.metrics.get("interactions_count", 0),
                "epochs_completed": job.epochs_completed,
                "raw_metrics": job.metrics,
            }
        )

    @app.get("/health")
    def health_check() -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    return app


def start_service(
    config: Optional[ServiceConfig] = None,
    simulation_fn: Optional[SimulationFn] = None,
) -> None:
    """Start the OpenClaw service with uvicorn."""
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the OpenClaw service. "
            "Install with: pip install 'swarm-safety[api]'"
        )

    try:
        import uvicorn
    except ImportError as err:
        raise ImportError(
            "uvicorn is required for the OpenClaw service. "
            "Install with: pip install 'swarm-safety[api]'"
        ) from err

    config = config or ServiceConfig()
    app = create_app(config, simulation_fn=simulation_fn)
    uvicorn.run(app, host=config.host, port=config.port)
