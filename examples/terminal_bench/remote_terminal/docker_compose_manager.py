"""Docker Compose manager for managing Docker containers.

This code is adapted from terminal-bench (https://github.com/terminal-bench/terminal-bench)
"""

import io
import logging
import signal
import subprocess
import tarfile
import time
import weakref
from pathlib import Path
from typing import Optional

import docker
import docker.errors
from docker.models.containers import Container

logger = logging.getLogger(__name__)

# Global registry to track all active Docker containers for cleanup on signal
_active_containers: weakref.WeakSet = weakref.WeakSet()


def _cleanup_all_containers():
    """Clean up all tracked Docker containers. Called on signal."""
    logger.info("Cleaning up all tracked Docker containers...")
    for container_manager in list(_active_containers):
        try:
            container_manager.stop()
        except Exception as e:
            logger.warning(
                f"Error cleaning up container {container_manager._client_container_name}: {e}"
            )


def _setup_signal_handlers():
    """Setup signal handlers for cleanup on interrupt."""

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, cleaning up Docker containers...")
        _cleanup_all_containers()
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except Exception:
        # Signal handling may fail in some environments (e.g., Windows threads)
        pass


# Setup signal handlers on module import
_setup_signal_handlers()


class DockerComposeManager:
    """Manages Docker Compose lifecycle for containers."""

    CONTAINER_SESSION_LOGS_PATH = "/logs"
    CONTAINER_AGENT_LOGS_PATH = "/agent-logs"
    CONTAINER_TEST_DIR = Path("/tests")

    def __init__(
        self,
        client_container_name: str,
        client_image_name: str,
        docker_compose_path: Path,
        docker_image_name_prefix: Optional[str] = None,
        no_rebuild: bool = False,
        cleanup: bool = False,
        sessions_logs_path: Optional[Path] = None,
        agent_logs_path: Optional[Path] = None,
    ):
        try:
            self._client = docker.from_env()
        except docker.errors.DockerException as e:
            raise RuntimeError(
                f"Error creating docker client: {e}. "
                "Please ensure that Docker is installed and running."
            )

        self._client_container_name = client_container_name
        self._client_image_name = client_image_name
        self._docker_name_prefix = docker_image_name_prefix
        self._docker_compose_path = docker_compose_path
        self._no_rebuild = no_rebuild
        self._cleanup = cleanup
        self._client_container: Optional[Container] = None
        self._sessions_logs_path = sessions_logs_path
        self._agent_logs_path = agent_logs_path
        self._logger = logger

        # Register this instance for cleanup on signal
        _active_containers.add(self)

        # Build environment variables
        self.env = self._build_env_dict()

    def _build_env_dict(self) -> dict:
        """Build environment variables for docker compose."""
        import os

        env = os.environ.copy()
        # T-Bench docker-compose.yaml expects T_BENCH_ prefixed variables
        env["T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME"] = self._client_container_name
        env["T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME"] = self._client_image_name
        if self._docker_name_prefix:
            env["T_BENCH_TASK_DOCKER_NAME_PREFIX"] = self._docker_name_prefix
        env["T_BENCH_CONTAINER_LOGS_PATH"] = self.CONTAINER_SESSION_LOGS_PATH
        env["T_BENCH_CONTAINER_AGENT_LOGS_PATH"] = self.CONTAINER_AGENT_LOGS_PATH
        env["T_BENCH_TEST_DIR"] = str(self.CONTAINER_TEST_DIR)
        # sessions_logs_path and agent_logs_path must be provided by the caller
        # to avoid invalid docker volume specs (:/logs)
        if self._sessions_logs_path:
            env["T_BENCH_TASK_LOGS_PATH"] = str(self._sessions_logs_path.absolute())
        if self._agent_logs_path:
            env["T_BENCH_TASK_AGENT_LOGS_PATH"] = str(self._agent_logs_path.absolute())
        return env

    def get_docker_compose_command(self, command: list[str]) -> list[str]:
        """Get the full docker compose command."""
        return [
            "docker",
            "compose",
            "-p",
            self._client_container_name,
            "-f",
            str(self._docker_compose_path.resolve().absolute()),
            *command,
        ]

    def _run_docker_compose_command(
        self, command: list[str]
    ) -> subprocess.CompletedProcess:
        """Run a docker compose command."""
        full_command = self.get_docker_compose_command(command)
        self._logger.debug(f"Running docker compose command: {' '.join(full_command)}")

        try:
            result = subprocess.run(
                full_command,
                env=self.env,
                check=True,
                capture_output=True,
                text=True,
            )
            return result
        except subprocess.CalledProcessError as e:
            self._logger.warning(
                f"Docker compose command failed with exit code {e.returncode}"
            )
            self._logger.warning(f"Command: {' '.join(full_command)}")
            if e.stdout:
                self._logger.warning(f"STDOUT: {e.stdout}")
            if e.stderr:
                self._logger.warning(f"STDERR: {e.stderr}")
            raise

    def _cleanup_all_containers_by_project(self) -> None:
        """Clean up all containers associated with this docker compose project."""
        import time as _time
        try:
            if self._client_container_name:
                project_name = self._client_container_name

                list_start = _time.time()
                all_containers = self._client.containers.list(all=True)
                list_time = _time.time() - list_start
                if list_time > 1.0:
                    self._logger.warning(
                        f"[Docker] containers.list() took {list_time:.2f}s "
                        f"({len(all_containers)} containers)"
                    )

                containers_to_remove = []
                for container in all_containers:
                    container_name = container.name
                    container_id = container.id[:12]
                    compose_project = container.labels.get(
                        "com.docker.compose.project"
                    )

                    belongs_to_project = (
                        container_name == project_name
                        or container_name.startswith(project_name + "_")
                        or compose_project == project_name
                    )

                    if belongs_to_project:
                        containers_to_remove.append(
                            (container, container_name, container_id)
                        )

                if containers_to_remove:
                    self._logger.debug(
                        f"Found {len(containers_to_remove)} container(s) for project '{project_name}'"
                    )
                    try:
                        down_start = _time.time()
                        self._run_docker_compose_command(["down", "--remove-orphans"])
                        down_time = _time.time() - down_start
                        if down_time > 2.0:
                            self._logger.warning(
                                f"[Docker] 'docker compose down' took {down_time:.2f}s"
                            )
                    except Exception:
                        pass

                    for container, container_name, container_id in containers_to_remove:
                        try:
                            if container.status == "running":
                                container.stop(timeout=5)
                            container.remove(force=True)
                            self._logger.debug(
                                f"Removed container: {container_name} (ID: {container_id})"
                            )
                        except docker.errors.NotFound:
                            pass
                        except Exception as e:
                            self._logger.warning(
                                f"Error removing container {container_name}: {e}"
                            )
        except Exception as e:
            self._logger.warning(f"Error cleaning up containers by project: {e}")

    def start(self, max_retries: int = 3, retry_delay: float = 5.0) -> Container:
        """Start the Docker container with retry mechanism."""
        import time as _time
        start_time = _time.time()
        last_exception = None
        for attempt in range(max_retries):
            try:
                self._logger.info(
                    f"[Docker] Starting container {self._client_container_name} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

                cleanup_start = _time.time()
                self._cleanup_all_containers_by_project()
                self._logger.debug(
                    f"[Docker] Cleanup took {_time.time() - cleanup_start:.2f}s"
                )

                if not self._no_rebuild:
                    image_exists = False
                    try:
                        self._client.images.get(self._client_image_name)
                        image_exists = True
                        self._logger.debug(
                            f"Image {self._client_image_name} already exists"
                        )
                    except docker.errors.ImageNotFound:
                        self._logger.info(
                            f"[Docker] Image {self._client_image_name} not found, building..."
                        )

                    if not image_exists:
                        build_start = _time.time()
                        self._run_docker_compose_command(["build"])
                        self._logger.info(
                            f"[Docker] Build took {_time.time() - build_start:.2f}s"
                        )

                up_start = _time.time()
                self._run_docker_compose_command(["up", "-d"])
                self._logger.debug(
                    f"[Docker] 'up -d' took {_time.time() - up_start:.2f}s"
                )

                self._client_container = self._client.containers.get(
                    self._client_container_name
                )

                if self._client_container.status != "running":
                    raise RuntimeError(
                        f"Container {self._client_container_name} is not running"
                    )

                total_time = _time.time() - start_time
                self._logger.info(
                    f"[Docker] Container {self._client_container_name} started "
                    f"in {total_time:.2f}s"
                )
                return self._client_container
            except Exception as e:
                last_exception = e
                self._logger.warning(
                    f"Failed to start Docker container (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    self._cleanup_all_containers_by_project()
                    time.sleep(retry_delay)

        self._cleanup_all_containers_by_project()
        raise RuntimeError(
            f"Failed to start Docker container after {max_retries} attempts"
        ) from last_exception

    def stop(self) -> None:
        """Stop and remove the docker compose services."""
        try:
            self._cleanup_all_containers_by_project()
            self._remove_networks()

            if self._cleanup:
                try:
                    # Don't use --rmi all as images are shared across containers
                    # and attempting to remove them causes "Resource is still in use" errors
                    self._run_docker_compose_command(
                        ["down", "--volumes", "--remove-orphans"]
                    )
                except Exception as e:
                    self._logger.warning(f"Error during docker compose cleanup: {e}")

                self._remove_networks()
        except Exception as e:
            self._logger.error(f"Error cleaning up docker compose services: {e}")
        finally:
            _active_containers.discard(self)

    def _remove_networks(self) -> None:
        """Remove Docker networks associated with this compose project."""
        try:
            if not self._client_container_name:
                return

            all_networks = self._client.networks.list()
            project_name = self._client_container_name

            for network in all_networks:
                network_name = network.name
                if network_name.startswith(
                    project_name + "_"
                ) or network_name == project_name:
                    try:
                        if network.containers:
                            for container in network.containers:
                                try:
                                    network.disconnect(container, force=True)
                                except Exception:
                                    pass
                        network.remove()
                        self._logger.debug(f"Removed network: {network_name}")
                    except docker.errors.NotFound:
                        pass
                    except Exception as e:
                        self._logger.debug(f"Could not remove network {network_name}: {e}")
        except Exception as e:
            self._logger.warning(f"Error removing networks: {e}")

    @staticmethod
    def _create_tar_archive(
        paths: list[Path], container_filename: Optional[str]
    ) -> io.BytesIO:
        """Create a tar archive from paths."""
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for path in paths:
                if path.is_file():
                    arcname = container_filename if container_filename else path.name
                    tar.add(path, arcname=arcname)
                elif path.is_dir():
                    for item in path.rglob("*"):
                        tar.add(item, arcname=item.relative_to(path))
                else:
                    raise ValueError(f"Path {path} is neither a file nor directory")

        tar_stream.seek(0)
        return tar_stream

    @staticmethod
    def copy_to_container(
        container: Container,
        paths: list[Path] | Path,
        container_dir: Optional[str] = None,
        container_filename: Optional[str] = None,
    ) -> None:
        """Copy files or directories to a running docker container."""
        container_dir = container_dir or (
            container.attrs.get("Config", {}).get("WorkingDir")
            if container.attrs
            else None
        )

        if container_dir is None:
            raise ValueError("Container working directory not found")

        if isinstance(paths, Path):
            paths = [paths]

        container.exec_run(f"mkdir -p {container_dir}")

        tar_stream = DockerComposeManager._create_tar_archive(paths, container_filename)
        container.put_archive(container_dir, tar_stream.read())
        tar_stream.close()

    def copy_to_client_container(
        self,
        paths: list[Path] | Path,
        container_dir: Optional[str] = None,
        container_filename: Optional[str] = None,
    ) -> None:
        """Copy files or directories to the client container."""
        if self._client_container is None:
            raise ValueError("Client container not started")

        return self.copy_to_container(
            container=self._client_container,
            paths=paths,
            container_dir=container_dir,
            container_filename=container_filename,
        )
