"""Remote Terminal and TmuxSession implementations using HTTP client.

These classes provide the same interface as Terminal and TmuxSession,
but communicate with a dev node via HTTP instead of direct Docker API calls.
"""

import logging
from pathlib import Path
from typing import Optional

from .remote_docker_client import RemoteDockerClient

logger = logging.getLogger(__name__)


class RemoteTerminal:
    """Remote Terminal that communicates with dev node via HTTP.

    This class provides the same interface as Terminal but uses HTTP
    to communicate with a dev node that manages Docker containers.
    """

    def __init__(
        self,
        client_container_name: str,
        client_image_name: str,
        docker_compose_path: Path,
        docker_image_name_prefix: Optional[str] = None,
        sessions_logs_path: Optional[Path] = None,
        agent_logs_path: Optional[Path] = None,
        commands_path: Optional[Path] = None,
        no_rebuild: bool = False,
        cleanup: bool = False,
        disable_recording: bool = False,
        dev_node_url: str = "http://localhost:8080",
    ):
        """Initialize remote terminal.

        Args:
            client_container_name: Name for the container
            client_image_name: Docker image name
            docker_compose_path: Path to docker-compose.yaml (on dev node)
            docker_image_name_prefix: Prefix for Docker image names
            sessions_logs_path: Path for session logs (on dev node)
            agent_logs_path: Path for agent logs (on dev node)
            commands_path: Path to save command history (not used in remote mode)
            no_rebuild: Skip rebuilding if image exists
            cleanup: Clean up resources on stop
            disable_recording: Disable recording (always True in remote mode)
            dev_node_url: Base URL of dev node service
        """
        self._client_container_name = client_container_name
        self._client_image_name = client_image_name
        self._docker_compose_path = docker_compose_path
        self._docker_image_name_prefix = docker_image_name_prefix
        self._sessions_logs_path = sessions_logs_path
        self._agent_logs_path = agent_logs_path
        self._no_rebuild = no_rebuild
        self._cleanup = cleanup

        self._logger = logger
        self._sessions: dict[str, "RemoteTmuxSession"] = {}

        # Initialize HTTP client
        self._client = RemoteDockerClient(dev_node_url=dev_node_url)
        self._container_id: Optional[str] = None

    def start(self) -> None:
        """Start the Docker container on dev node."""
        self._logger.info(
            f"Starting container: name={self._client_container_name}, "
            f"image={self._client_image_name}"
        )

        response = self._client.start_container(
            container_name=self._client_container_name,
            image_name=self._client_image_name,
            docker_compose_path=str(self._docker_compose_path),
            docker_image_name_prefix=self._docker_image_name_prefix,
            no_rebuild=self._no_rebuild,
            cleanup=self._cleanup,
            sessions_logs_path=(
                str(self._sessions_logs_path) if self._sessions_logs_path else None
            ),
            agent_logs_path=(
                str(self._agent_logs_path) if self._agent_logs_path else None
            ),
        )

        self._container_id = response["container_id"]
        self._logger.info(
            f"Started container {self._container_id} on dev node "
            f"(name={self._client_container_name})"
        )

    def stop(self) -> None:
        """Stop all sessions and the Docker container."""
        if self._container_id:
            # Stop all sessions first
            for session in list(self._sessions.values()):
                try:
                    session.stop()
                except Exception as e:
                    self._logger.warning(f"Error stopping session: {e}")

            # Stop container
            try:
                self._client.stop_container(self._container_id)
                self._logger.info(f"Stopped container {self._container_id}")
            except Exception as e:
                self._logger.warning(f"Error stopping container: {e}")

            self._sessions.clear()
            self._container_id = None

    def create_session(
        self, session_name: str, as_configured_user: bool = True
    ) -> "RemoteTmuxSession":
        """Create a new tmux session.

        Args:
            session_name: Name of the tmux session
            as_configured_user: Whether to use configured user or root

        Returns:
            The created RemoteTmuxSession instance
        """
        if self._container_id is None:
            raise ValueError("Container not started. Run start() first.")

        if session_name in self._sessions:
            raise ValueError(f"Session {session_name} already exists")

        self._logger.debug(
            f"Creating session '{session_name}' in container {self._container_id}"
        )

        response = self._client.create_session(
            container_id=self._container_id,
            session_name=session_name,
            as_configured_user=as_configured_user,
        )

        session_id = response["session_id"]
        self._logger.info(
            f"Created session '{session_name}' (id={session_id}) "
            f"in container {self._container_id}"
        )

        session = RemoteTmuxSession(
            session_name=session_name,
            session_id=session_id,
            container_id=self._container_id,
            client=self._client,
        )

        self._sessions[session_name] = session
        return session

    def get_session(self, session_name: str) -> "RemoteTmuxSession":
        """Get an existing tmux session by name.

        Args:
            session_name: Name of the session to get

        Returns:
            The RemoteTmuxSession instance
        """
        if session_name not in self._sessions:
            raise ValueError(f"Session {session_name} does not exist")
        return self._sessions[session_name]

    def copy_to_container(
        self,
        paths: list[Path] | Path,
        container_dir: Optional[str] = None,
        container_filename: Optional[str] = None,
    ) -> None:
        """Copy files or directories to the container.

        In remote mode, this method sends paths to dev node, which then copies
        files from its local filesystem to the container. This avoids HTTP file
        transfer, similar to how docker-compose.yaml path is handled.

        Args:
            paths: File or directory paths to copy (paths should exist on dev node)
            container_dir: Directory in container to copy to
            container_filename: Filename to use in container (for single files)
        """
        if self._container_id is None:
            raise ValueError("Container not started")

        if isinstance(paths, Path):
            paths = [paths]

        # Convert Path objects to absolute path strings
        # These paths should exist on dev node (dev node and local are synced)
        path_strings = [str(path.resolve().absolute()) for path in paths]

        # Use the copy_from_path endpoint which copies from dev node's local filesystem
        self._client.copy_to_container_from_path(
            container_id=self._container_id,
            paths=path_strings,
            container_dir=container_dir,
            container_filename=container_filename,
        )

    @property
    def container(self):
        """Return None for remote mode (container is on dev node)."""
        return None

    @property
    def container_id(self) -> Optional[str]:
        """Return the container ID."""
        return self._container_id


class RemoteTmuxSession:
    """Remote TmuxSession that communicates with dev node via HTTP.

    This class provides the same interface as TmuxSession but uses HTTP
    to communicate with a dev node that manages tmux sessions.
    """

    def __init__(
        self,
        session_name: str,
        session_id: str,
        container_id: str,
        client: RemoteDockerClient,
    ):
        """Initialize remote tmux session.

        Args:
            session_name: Name of the tmux session
            session_id: ID of the session (same as session_name typically)
            container_id: ID of the container
            client: RemoteDockerClient instance
        """
        self._session_name = session_name
        self._session_id = session_id
        self._container_id = container_id
        self._client = client
        self._logger = logger

    def send_keys(
        self,
        keys: str | list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ) -> None:
        """Execute a command in the tmux session.

        Args:
            keys: The keys to send to the tmux session
            block: Whether to wait for the command to complete
            min_timeout_sec: Minimum time in seconds to wait after executing
            max_timeout_sec: Maximum time in seconds to wait for blocking commands
        """
        if isinstance(keys, str):
            keys = [keys]

        # Validate container_id and session_id are set
        if not self._container_id or not self._session_id:
            raise ValueError(
                f"Container ID or Session ID not set. "
                f"container_id={self._container_id}, session_id={self._session_id}"
            )

        self._logger.debug(
            f"Sending keys to container={self._container_id}, "
            f"session={self._session_id}, keys={keys}"
        )

        try:
            self._client.send_keys(
                container_id=self._container_id,
                session_id=self._session_id,
                keys=keys,
                block=block,
                min_timeout_sec=min_timeout_sec,
                max_timeout_sec=max_timeout_sec,
            )
        except Exception as e:
            self._logger.error(
                f"Failed to send keys to container={self._container_id}, "
                f"session={self._session_id}: {e}"
            )
            raise

    def get_incremental_output(self) -> str:
        """Get incremental output from the tmux session.

        Returns:
            Formatted output string
        """
        # Validate container_id and session_id are set
        if not self._container_id or not self._session_id:
            raise ValueError(
                f"Container ID or Session ID not set. "
                f"container_id={self._container_id}, session_id={self._session_id}"
            )

        self._logger.debug(
            f"Getting output from container={self._container_id}, "
            f"session={self._session_id}"
        )

        try:
            response = self._client.get_incremental_output(
                container_id=self._container_id,
                session_id=self._session_id,
            )
            return response.get("output", "")
        except Exception as e:
            self._logger.error(
                f"Failed to get output from container={self._container_id}, "
                f"session={self._session_id}: {e}"
            )
            raise

    def stop(self) -> None:
        """Stop the tmux session (no-op in remote mode, handled by dev node)."""
        # Session cleanup is handled by dev node when container stops
        pass

    def capture_pane(self, capture_entire: bool = False) -> str:
        """Capture the tmux pane content.

        Note: This is a simplified implementation. You may want to add
        a dedicated endpoint on dev node for this.
        """
        # For now, use get_incremental_output
        # In production, add a /capture-pane endpoint
        return self.get_incremental_output()

    def is_session_alive(self) -> bool:
        """Check if the tmux session is still alive.

        Note: This requires a health check endpoint on dev node.
        """
        # In production, add a /sessions/{session_id}/health endpoint
        try:
            self.get_incremental_output()
            return True
        except Exception:
            return False
