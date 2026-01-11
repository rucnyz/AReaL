"""HTTP server for dev node that manages Docker containers.

This server runs on the dev node and handles Docker container management
requests from K8s training nodes via HTTP API.

Usage:
    python -m examples.terminal_bench.remote_terminal.dev_node_server --host 0.0.0.0 --port 8080
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import from AReaL's local_terminal module
from .local_terminal import Terminal, TmuxSession

app = Flask(__name__)
CORS(app)  # Enable CORS for k8s requests

# Global state: track active containers and sessions
# In production, use a proper database or Redis
_active_containers: Dict[str, Terminal] = {}
_active_sessions: Dict[str, Dict[str, TmuxSession]] = {}  # {container_id: {session_id: session}}


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "active_containers": len(_active_containers),
    })


@app.route('/containers/start', methods=['POST'])
def start_container():
    """Start a Docker container.

    Request body:
        {
            "container_name": str,
            "image_name": str,
            "docker_compose_path": str,
            "docker_image_name_prefix": Optional[str],
            "no_rebuild": bool,
            "cleanup": bool,
            "sessions_logs_path": Optional[str],
            "agent_logs_path": Optional[str]
        }

    Response:
        {
            "container_id": str,
            "status": "started"
        }
    """
    try:
        data = request.json

        # Create Terminal instance (reuses your existing code)
        terminal = Terminal(
            client_container_name=data['container_name'],
            client_image_name=data['image_name'],
            docker_compose_path=Path(data['docker_compose_path']),
            docker_image_name_prefix=data.get('docker_image_name_prefix'),
            sessions_logs_path=Path(data['sessions_logs_path']) if data.get('sessions_logs_path') else None,
            agent_logs_path=Path(data['agent_logs_path']) if data.get('agent_logs_path') else None,
            no_rebuild=data.get('no_rebuild', False),
            cleanup=data.get('cleanup', False),
            disable_recording=True,
        )

        # Start container
        terminal.start()

        # Use container name as ID (or generate UUID)
        container_id = data['container_name']
        _active_containers[container_id] = terminal
        _active_sessions[container_id] = {}

        return jsonify({
            "container_id": container_id,
            "status": "started",
        }), 200

    except Exception as e:
        logging.error(f"Failed to start container: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "failed",
        }), 500


@app.route('/containers/<container_id>/stop', methods=['POST'])
def stop_container(container_id: str):
    """Stop and remove a container.

    Response:
        {
            "status": "stopped"
        }
    """
    try:
        if container_id not in _active_containers:
            return jsonify({
                "error": f"Container {container_id} not found",
            }), 404

        terminal = _active_containers[container_id]
        terminal.stop()

        # Clean up state
        del _active_containers[container_id]
        if container_id in _active_sessions:
            del _active_sessions[container_id]

        return jsonify({
            "status": "stopped",
        }), 200

    except Exception as e:
        logging.error(f"Failed to stop container {container_id}: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "failed",
        }), 500


@app.route('/containers/<container_id>/sessions', methods=['POST'])
def create_session(container_id: str):
    """Create a tmux session in a container.

    Request body:
        {
            "session_name": str,
            "as_configured_user": bool
        }

    Response:
        {
            "session_id": str,
            "status": "created"
        }
    """
    try:
        if container_id not in _active_containers:
            return jsonify({
                "error": f"Container {container_id} not found",
            }), 404

        data = request.json
        terminal = _active_containers[container_id]

        session = terminal.create_session(
            session_name=data['session_name'],
            as_configured_user=data.get('as_configured_user', True),
        )

        session_id = data['session_name']
        _active_sessions[container_id][session_id] = session

        return jsonify({
            "session_id": session_id,
            "status": "created",
        }), 200

    except Exception as e:
        logging.error(f"Failed to create session: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "failed",
        }), 500


@app.route('/containers/<container_id>/sessions/<session_id>/send-keys', methods=['POST'])
def send_keys(container_id: str, session_id: str):
    """Send keys/commands to a tmux session.

    Request body:
        {
            "keys": list[str],
            "block": bool,
            "min_timeout_sec": float,
            "max_timeout_sec": float
        }

    Response:
        {
            "status": "sent"
        }
    """
    try:
        # Validate container exists
        if container_id not in _active_containers:
            logging.warning(
                f"Container {container_id} not found. "
                f"Active containers: {list(_active_containers.keys())}"
            )
            return jsonify({
                "error": f"Container {container_id} not found",
                "active_containers": list(_active_containers.keys()),
            }), 404

        # Validate session exists in container
        if container_id not in _active_sessions or session_id not in _active_sessions[container_id]:
            available_sessions = (
                list(_active_sessions[container_id].keys())
                if container_id in _active_sessions
                else []
            )
            logging.warning(
                f"Session {session_id} not found in container {container_id}. "
                f"Available sessions: {available_sessions}"
            )
            return jsonify({
                "error": f"Session {session_id} not found in container {container_id}",
                "available_sessions": available_sessions,
            }), 404

        data = request.json
        session = _active_sessions[container_id][session_id]

        logging.debug(
            f"Sending keys to container={container_id}, session={session_id}, "
            f"keys={data.get('keys', [])}"
        )

        session.send_keys(
            keys=data['keys'],
            block=data.get('block', False),
            min_timeout_sec=data.get('min_timeout_sec', 0.0),
            max_timeout_sec=data.get('max_timeout_sec', 180.0),
        )

        return jsonify({
            "status": "sent",
            "container_id": container_id,
            "session_id": session_id,
        }), 200

    except TimeoutError as e:
        logging.warning(f"Command timed out: {e}")
        return jsonify({
            "error": "Command timed out",
            "status": "timeout",
        }), 408
    except Exception as e:
        logging.error(f"Failed to send keys: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "failed",
        }), 500


@app.route('/containers/<container_id>/sessions/<session_id>/output', methods=['GET'])
def get_output(container_id: str, session_id: str):
    """Get incremental output from a tmux session.

    Response:
        {
            "output": str,
            "status": "success"
        }
    """
    try:
        # Validate container exists
        if container_id not in _active_containers:
            logging.warning(
                f"Container {container_id} not found. "
                f"Active containers: {list(_active_containers.keys())}"
            )
            return jsonify({
                "error": f"Container {container_id} not found",
                "active_containers": list(_active_containers.keys()),
            }), 404

        # Validate session exists in container
        if container_id not in _active_sessions or session_id not in _active_sessions[container_id]:
            available_sessions = (
                list(_active_sessions[container_id].keys())
                if container_id in _active_sessions
                else []
            )
            logging.warning(
                f"Session {session_id} not found in container {container_id}. "
                f"Available sessions: {available_sessions}"
            )
            return jsonify({
                "error": f"Session {session_id} not found in container {container_id}",
                "available_sessions": available_sessions,
            }), 404

        session = _active_sessions[container_id][session_id]

        logging.debug(
            f"Getting output from container={container_id}, session={session_id}"
        )

        output = session.get_incremental_output()

        return jsonify({
            "output": output,
            "status": "success",
            "container_id": container_id,
            "session_id": session_id,
        }), 200

    except Exception as e:
        logging.error(f"Failed to get output: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "failed",
        }), 500


@app.route('/containers/<container_id>/copy-from-path', methods=['POST'])
def copy_from_path_to_container(container_id: str):
    """Copy files from dev node local filesystem to container.

    This endpoint allows copying files/directories from dev node's local filesystem
    directly to the container, similar to how docker-compose.yaml path is used.

    Request body:
        {
            "paths": list[str],  # List of file/directory paths on dev node
            "container_dir": str,  # Destination directory in container
            "container_filename": Optional[str]  # Filename to use in container (for single files)
        }

    Response:
        {
            "status": "copied"
        }
    """
    try:
        if container_id not in _active_containers:
            return jsonify({
                "error": f"Container {container_id} not found",
            }), 404

        data = request.json
        paths = data.get('paths', [])
        container_dir = data.get('container_dir')
        container_filename = data.get('container_filename')

        if not paths:
            return jsonify({
                "error": "No paths provided",
            }), 400

        terminal = _active_containers[container_id]

        # Convert string paths to Path objects
        path_objects = [Path(p) for p in paths]

        # Verify all paths exist on dev node
        for path in path_objects:
            if not path.exists():
                return jsonify({
                    "error": f"Path not found on dev node: {path}",
                }), 404

        # Use terminal's copy_to_container method which works with local paths
        terminal.copy_to_container(
            paths=path_objects,
            container_dir=container_dir,
            container_filename=container_filename,
        )

        return jsonify({
            "status": "copied",
            "paths": paths,
        }), 200

    except Exception as e:
        logging.error(f"Failed to copy from path: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "failed",
        }), 500


def main():
    """Run the dev node server."""
    parser = argparse.ArgumentParser(description='Dev node Docker service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    setup_logging()

    logging.info(f"Starting dev node server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
