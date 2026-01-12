"""Remote Docker client that communicates with dev node via HTTP.

This module provides a client-side wrapper that replaces direct Docker API calls
with HTTP requests to a dev node that manages Docker containers.
"""

import logging
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class RemoteDockerClient:
    """HTTP client for communicating with dev node Docker service."""

    def __init__(
        self,
        dev_node_url: str,
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize remote Docker client.

        Args:
            dev_node_url: Base URL of the dev node HTTP service (e.g., "http://dev-node:8080")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.dev_node_url = dev_node_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self._logger = logger

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make HTTP request to dev node.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint (e.g., "/containers/start")
            json_data: JSON payload for request body
            params: Query parameters

        Returns:
            Response JSON as dictionary

        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.dev_node_url}{endpoint}"
        self._logger.debug(f"{method} {url} - {json_data}")

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=json_data,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            self._logger.error(f"Request failed: {method} {url} - {e}")
            raise

        if response.status_code >= 400:
            body = response.text
            if len(body) > 2000:
                body = f"{body[:2000]}...(truncated)"
            self._logger.error(
                "Request failed: %s %s - HTTP %s response: %s",
                method,
                url,
                response.status_code,
                body,
            )
            response.raise_for_status()

        return response.json()

    def start_container(
        self,
        container_name: str,
        image_name: str,
        docker_compose_path: str,
        docker_image_name_prefix: Optional[str] = None,
        no_rebuild: bool = False,
        cleanup: bool = False,
        sessions_logs_path: Optional[str] = None,
        agent_logs_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Start a Docker container on dev node.

        Args:
            container_name: Name for the container
            image_name: Docker image name
            docker_compose_path: Path to docker-compose.yaml (on dev node)
            docker_image_name_prefix: Prefix for Docker image names
            no_rebuild: Skip rebuilding if image exists
            cleanup: Clean up resources on stop
            sessions_logs_path: Path for session logs (on dev node)
            agent_logs_path: Path for agent logs (on dev node)

        Returns:
            Response containing container_id and status
        """
        return self._request(
            method="POST",
            endpoint="/containers/start",
            json_data={
                "container_name": container_name,
                "image_name": image_name,
                "docker_compose_path": docker_compose_path,
                "docker_image_name_prefix": docker_image_name_prefix,
                "no_rebuild": no_rebuild,
                "cleanup": cleanup,
                "sessions_logs_path": sessions_logs_path,
                "agent_logs_path": agent_logs_path,
            },
        )

    def stop_container(self, container_id: str) -> dict[str, Any]:
        """Stop and remove a container.

        Args:
            container_id: ID of the container to stop

        Returns:
            Response with status
        """
        return self._request(
            method="POST",
            endpoint=f"/containers/{container_id}/stop",
        )

    def create_session(
        self,
        container_id: str,
        session_name: str,
        as_configured_user: bool = True,
    ) -> dict[str, Any]:
        """Create a tmux session in a container.

        Args:
            container_id: ID of the container
            session_name: Name for the tmux session
            as_configured_user: Whether to use configured user or root

        Returns:
            Response containing session_id
        """
        return self._request(
            method="POST",
            endpoint=f"/containers/{container_id}/sessions",
            json_data={
                "session_name": session_name,
                "as_configured_user": as_configured_user,
            },
        )

    def send_keys(
        self,
        container_id: str,
        session_id: str,
        keys: list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ) -> dict[str, Any]:
        """Send keys/commands to a tmux session.

        Args:
            container_id: ID of the container
            session_id: ID of the tmux session
            keys: List of keys/commands to send
            block: Whether to wait for command completion
            min_timeout_sec: Minimum wait time for non-blocking
            max_timeout_sec: Maximum wait time for blocking

        Returns:
            Response with status
        """
        return self._request(
            method="POST",
            endpoint=f"/containers/{container_id}/sessions/{session_id}/send-keys",
            json_data={
                "keys": keys,
                "block": block,
                "min_timeout_sec": min_timeout_sec,
                "max_timeout_sec": max_timeout_sec,
            },
        )

    def get_incremental_output(
        self,
        container_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Get incremental output from a tmux session.

        Args:
            container_id: ID of the container
            session_id: ID of the tmux session

        Returns:
            Response containing observation/output text
        """
        return self._request(
            method="GET",
            endpoint=f"/containers/{container_id}/sessions/{session_id}/output",
        )

    def copy_to_container_from_path(
        self,
        container_id: str,
        paths: list[str],
        container_dir: Optional[str] = None,
        container_filename: Optional[str] = None,
    ) -> dict[str, Any]:
        """Copy files from dev node local filesystem to container.

        This method tells dev node to copy files from its local filesystem
        directly to the container, avoiding HTTP file transfer.

        Args:
            container_id: ID of the container
            paths: List of file/directory paths on dev node
            container_dir: Destination directory in container
            container_filename: Filename to use in container (for single files)

        Returns:
            Response with status
        """
        return self._request(
            method="POST",
            endpoint=f"/containers/{container_id}/copy-from-path",
            json_data={
                "paths": paths,
                "container_dir": container_dir,
                "container_filename": container_filename,
            },
        )

    def health_check(self) -> dict[str, Any]:
        """Check if dev node service is healthy.

        Returns:
            Health status
        """
        return self._request(method="GET", endpoint="/health")
