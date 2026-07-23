"""Manages AWM FastAPI server instances for agent environments.

Each agent gets its own server instance (separate port, private DB copy).
Servers start lazily on first AWM_EXECUTE_TASK action, reset DB between
epochs, and stop on orchestrator shutdown.

NOTE: In Phase 1, servers are simulated (no actual subprocess).
Phase 2 will add real server lifecycle management.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, Optional

from swarm.bridges.awm.config import AWMConfig

logger = logging.getLogger(__name__)


class AWMServerInstance:
    """Represents a single AWM server instance for one agent."""

    def __init__(
        self,
        agent_id: str,
        port: int,
        environment_id: str,
        envs_path: str,
        *,
        live_mode: bool = False,
        server_command_template: str = "",
        host: str = "127.0.0.1",
        startup_timeout: float = 30.0,
        health_check_interval: float = 0.5,
        data_path: str = "data/awm-1k",
    ) -> None:
        self.agent_id = agent_id
        self.port = port
        self.environment_id = environment_id
        self.envs_path = envs_path
        self.live_mode = live_mode
        self.server_command_template = server_command_template
        self.host = host
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval
        self.data_path = data_path
        self.running = False
        self._process: Any = None  # subprocess.Popen when live_mode

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> bool:
        """Start the AWM server.

        When live_mode is False: no-op (simulation).
        When live_mode is True: spawn subprocess + poll health check.
        """
        if not self.live_mode:
            logger.info(
                "AWM server start (simulated) for agent=%s port=%d env=%s",
                self.agent_id,
                self.port,
                self.environment_id,
            )
            self.running = True
            return True

        cmd = self.server_command_template.format(
            python=sys.executable,
            host=self.host,
            port=self.port,
            env_path=self.envs_path,
            env_id=self.environment_id,
            data_path=self.data_path,
        )
        cmd_args = shlex.split(cmd)
        logger.info(
            "Starting AWM server for agent=%s: %s", self.agent_id, cmd_args
        )

        self._process = subprocess.Popen(
            cmd_args,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Poll health check until server responds or timeout
        from swarm.bridges.awm.mcp_client import AWMMCPSyncClient

        client = AWMMCPSyncClient(
            base_url=self.base_url, timeout=self.health_check_interval
        )
        deadline = time.monotonic() + self.startup_timeout
        try:
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    logger.error(
                        "AWM server process exited early for agent=%s (rc=%d)",
                        self.agent_id,
                        self._process.returncode,
                    )
                    self._process = None
                    return False
                if client.health_check():
                    self.running = True
                    logger.info(
                        "AWM server healthy for agent=%s port=%d",
                        self.agent_id,
                        self.port,
                    )
                    return True
                time.sleep(self.health_check_interval)
        finally:
            client.close()

        logger.error(
            "AWM server startup timeout for agent=%s after %.1fs",
            self.agent_id,
            self.startup_timeout,
        )
        # Kill the timed-out process
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None
        return False

    async def stop(self) -> None:
        """Stop the AWM server."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        self.running = False
        logger.info(
            "AWM server stopped for agent=%s port=%d",
            self.agent_id,
            self.port,
        )

    async def reset_db(self) -> bool:
        """Reset the database to initial state.

        When live_mode is False: no-op.
        When live_mode is True: POST /reset via sync client.
        """
        if not self.live_mode:
            logger.debug("AWM DB reset (simulated) for agent=%s", self.agent_id)
            return True

        from swarm.bridges.awm.mcp_client import AWMMCPSyncClient

        client = AWMMCPSyncClient(base_url=self.base_url)
        try:
            result: bool = client.reset_environment()
            return result
        finally:
            client.close()


class AWMServerManager:
    """Manages AWM server instances for all agents in a scenario.

    Lifecycle:
    - start_server(agent_id): Lazy start on first task
    - reset_all(): Called between epochs
    - shutdown(): Called on orchestrator shutdown
    """

    def __init__(self, config: AWMConfig) -> None:
        self.config = config
        self._servers: Dict[str, AWMServerInstance] = {}
        self._next_port = config.base_port
        self._shared_server: Optional[AWMServerInstance] = None

    def _allocate_port(self) -> int:
        """Allocate the next available port."""
        port = self._next_port
        self._next_port += 1
        return port

    def _create_server_instance(self, agent_id: str, port: int) -> AWMServerInstance:
        """Create an AWMServerInstance with configuration from self.config."""
        return AWMServerInstance(
            agent_id=agent_id,
            port=port,
            environment_id=self.config.environment_id,
            envs_path=str(self.config.envs_path),
            live_mode=self.config.live_mode,
            server_command_template=self.config.server_command_template,
            host=self.config.host,
            startup_timeout=self.config.server_startup_timeout,
            health_check_interval=self.config.health_check_interval,
            data_path=str(self.config.data_path),
        )

    async def start_server(self, agent_id: str) -> Optional[AWMServerInstance]:
        """Start (or return existing) server for an agent.

        In shared_database mode, all agents share one server instance.
        Returns None if max_concurrent_servers would be exceeded.
        """
        if self.config.shared_database:
            return await self._start_shared_server(agent_id)
        return await self._start_dedicated_server(agent_id)

    async def _start_shared_server(
        self, agent_id: str
    ) -> Optional[AWMServerInstance]:
        """Return the single shared server, creating it on first call."""
        if self._shared_server is not None:
            # Track agent → server mapping for lookup
            self._servers[agent_id] = self._shared_server
            if not self._shared_server.running:
                await self._shared_server.start()
            return self._shared_server

        port = self._allocate_port()
        server = self._create_server_instance("shared", port)
        await server.start()
        self._shared_server = server
        self._servers[agent_id] = server
        return server

    async def _start_dedicated_server(
        self, agent_id: str
    ) -> Optional[AWMServerInstance]:
        """Start a dedicated server for one agent (original behavior)."""
        if agent_id in self._servers:
            server = self._servers[agent_id]
            if not server.running:
                await server.start()
            return server

        if len(self._servers) >= self.config.max_concurrent_servers:
            logger.warning(
                "Cannot start AWM server for %s: max_concurrent_servers=%d reached",
                agent_id,
                self.config.max_concurrent_servers,
            )
            return None

        port = self._allocate_port()
        server = self._create_server_instance(agent_id, port)
        await server.start()
        self._servers[agent_id] = server
        return server

    def get_server(self, agent_id: str) -> Optional[AWMServerInstance]:
        """Get an existing server instance for an agent."""
        return self._servers.get(agent_id)

    async def reset_all(self) -> None:
        """Reset all servers' databases (called between epochs).

        In shared mode, reset the shared server once instead of per-agent.
        """
        if self._shared_server is not None:
            if self._shared_server.running:
                await self._shared_server.reset_db()
            return

        for server in self._servers.values():
            if server.running:
                await server.reset_db()

    async def shutdown(self) -> None:
        """Stop all servers and clean up."""
        if self._shared_server is not None:
            await self._shared_server.stop()
            self._shared_server = None
            self._servers.clear()
        else:
            for server in list(self._servers.values()):
                await server.stop()
            self._servers.clear()
        self._next_port = self.config.base_port

    @property
    def active_count(self) -> int:
        """Number of currently running servers."""
        return sum(1 for s in self._servers.values() if s.running)
