"""Local AgentRxiv server management.

This module provides utilities for starting and managing a local
AgentRxiv server instance for development and testing.
"""

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class AgentRxivServerError(Exception):
    """Error related to AgentRxiv server operations."""

    pass


class AgentRxivServer:
    """Manage a local AgentRxiv server instance.

    This class wraps the Flask-based AgentRxiv server from the
    Agent Laboratory project, providing lifecycle management.

    Example:
        >>> with AgentRxivServer(port=5000) as server:
        ...     # Server is running
        ...     client = AgentRxivClient()
        ...     results = client.search("query")
        >>> # Server is stopped

    Attributes:
        port: The port the server is running on.
        uploads_dir: Directory where PDFs are stored.
        is_running: Whether the server is currently running.
    """

    DEFAULT_AGENT_LAB_PATH = "./external/AgentLaboratory"

    def __init__(
        self,
        port: int = 5000,
        uploads_dir: str | Path = "./agentrxiv_papers",
        agent_lab_path: str | Path | None = None,
        auto_install_deps: bool = False,
    ):
        """Initialize the server manager.

        Args:
            port: Port to run the server on.
            uploads_dir: Directory for PDF storage.
            agent_lab_path: Path to Agent Laboratory installation.
            auto_install_deps: Whether to auto-install missing dependencies.
        """
        self.port = port
        self.uploads_dir = Path(uploads_dir)
        lab_path = (
            agent_lab_path
            if agent_lab_path is not None
            else os.environ.get("AGENT_LABORATORY_PATH", self.DEFAULT_AGENT_LAB_PATH)
        )
        self.agent_lab_path = Path(lab_path)
        self.auto_install_deps = auto_install_deps

        self._process: subprocess.Popen | None = None

    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    @property
    def base_url(self) -> str:
        """Get the server's base URL."""
        return f"http://127.0.0.1:{self.port}"

    def _check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        required = ["flask", "sentence_transformers"]
        missing = []

        for package in required:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append(package)

        if missing:
            if self.auto_install_deps:
                logger.info(f"Installing missing dependencies: {missing}")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + missing,
                    capture_output=True,
                )
                return True
            else:
                logger.error(f"Missing dependencies: {missing}")
                return False

        return True

    def _check_agent_lab(self) -> bool:
        """Check if Agent Laboratory is installed."""
        app_py = self.agent_lab_path / "app.py"
        if not app_py.exists():
            logger.error(
                f"Agent Laboratory not found at {self.agent_lab_path}. "
                "Run: git clone https://github.com/SamuelSchmidgall/AgentLaboratory external/AgentLaboratory"
            )
            return False
        return True

    def start(self, wait: bool = True, timeout: int = 30) -> None:
        """Start the AgentRxiv server.

        Args:
            wait: Whether to wait for the server to be ready.
            timeout: Maximum seconds to wait for server startup.

        Raises:
            AgentRxivServerError: If server fails to start.
        """
        if self.is_running:
            logger.warning("AgentRxiv server is already running")
            return

        if not self._check_agent_lab():
            raise AgentRxivServerError("Agent Laboratory not installed")

        if not self._check_dependencies():
            raise AgentRxivServerError("Missing dependencies")

        # Create uploads directory
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        # Set up environment
        env = os.environ.copy()
        env["FLASK_APP"] = "app.py"
        env["FLASK_RUN_PORT"] = str(self.port)

        # Start the Flask server
        logger.info(f"Starting AgentRxiv server on port {self.port}")

        self._process = subprocess.Popen(
            [sys.executable, "-m", "flask", "run", "--port", str(self.port)],
            cwd=self.agent_lab_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if wait:
            self._wait_for_ready(timeout)

        logger.info(f"AgentRxiv server started at {self.base_url}")

    def _wait_for_ready(self, timeout: int = 30) -> None:
        """Wait for the server to be ready to accept requests.

        Args:
            timeout: Maximum seconds to wait.

        Raises:
            AgentRxivServerError: If server doesn't start in time.
        """
        start = time.time()
        last_error = None

        while time.time() - start < timeout:
            # Check if process died
            if self._process and self._process.poll() is not None:
                stdout = self._process.stdout.read().decode() if self._process.stdout else ""
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                raise AgentRxivServerError(
                    f"Server process died. stdout: {stdout}, stderr: {stderr}"
                )

            try:
                response = requests.get(f"{self.base_url}/", timeout=1)
                if response.status_code == 200:
                    return
            except requests.RequestException as e:
                last_error = e

            time.sleep(0.5)

        raise AgentRxivServerError(
            f"Server failed to start within {timeout}s. Last error: {last_error}"
        )

    def stop(self, timeout: int = 5) -> None:
        """Stop the AgentRxiv server.

        Args:
            timeout: Maximum seconds to wait for graceful shutdown.
        """
        if not self._process:
            return

        logger.info("Stopping AgentRxiv server")

        # Try graceful termination first
        self._process.terminate()

        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Force kill if necessary
            logger.warning("Server did not stop gracefully, killing")
            self._process.kill()
            self._process.wait()

        self._process = None
        logger.info("AgentRxiv server stopped")

    def restart(self) -> None:
        """Restart the server."""
        self.stop()
        self.start()

    def seed_papers(self, paper_dir: str | Path) -> int:
        """Copy papers from a directory to the uploads folder.

        Args:
            paper_dir: Directory containing PDF files.

        Returns:
            Number of papers copied.
        """
        paper_dir = Path(paper_dir)
        if not paper_dir.exists():
            logger.warning(f"Paper directory not found: {paper_dir}")
            return 0

        count = 0
        for pdf in paper_dir.glob("*.pdf"):
            dest = self.uploads_dir / pdf.name
            if not dest.exists():
                shutil.copy(pdf, dest)
                count += 1
                logger.debug(f"Copied {pdf.name} to uploads")

        if count > 0:
            logger.info(f"Seeded {count} papers to AgentRxiv")

        return count

    def __enter__(self) -> "AgentRxivServer":
        """Context manager entry - start server."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - stop server."""
        self.stop()


def ensure_agent_laboratory(path: str | Path = "./external/AgentLaboratory") -> Path:
    """Ensure Agent Laboratory is installed.

    Args:
        path: Installation path.

    Returns:
        Path to Agent Laboratory.

    Raises:
        AgentRxivServerError: If installation fails.
    """
    path = Path(path)

    if (path / "app.py").exists():
        return path

    logger.info(f"Cloning Agent Laboratory to {path}")

    result = subprocess.run(
        [
            "git", "clone",
            "https://github.com/SamuelSchmidgall/AgentLaboratory",
            str(path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise AgentRxivServerError(f"Failed to clone: {result.stderr}")

    return path
