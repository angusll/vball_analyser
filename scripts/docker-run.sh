#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

USE_GPU=false
FORCE_BUILD=false
DETACH=false
PORT="${VBALL_PORT:-8081}"
VIDEOS_DIR=""
DEVICE_OVERRIDE="${VBALL_DEVICE:-}"

usage() {
    cat <<'EOF'
Usage: ./scripts/docker-run.sh [OPTIONS]

Build and run VBall Analyser in Docker (CPU by default, NVIDIA GPU with --gpu).

Options:
  --gpu                 Enable NVIDIA GPU support (--build-arg GPU=true and --gpus all)
  --build               Force rebuilding the Docker image before running
  -d, --detach          Run container in detached background mode
  -p, --port PORT       Host port to expose the studio on (default: 8081)
  -v, --videos-dir DIR  Host directory to mount read-only at /videos for local loading
  --device DEVICE       Override VBALL_DEVICE inside the container (e.g. cuda:0, cpu)
  -h, --help            Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            USE_GPU=true
            shift
            ;;
        --build)
            FORCE_BUILD=true
            shift
            ;;
        -d|--detach)
            DETACH=true
            shift
            ;;
        -p|--port)
            PORT="${2:?Missing value for --port}"
            shift 2
            ;;
        -v|--videos-dir)
            VIDEOS_DIR="${2:?Missing value for --videos-dir}"
            shift 2
            ;;
        --device)
            DEVICE_OVERRIDE="${2:?Missing value for --device}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "$USE_GPU" == "true" ]]; then
    IMAGE_TAG="vball-analyser:gpu"
else
    IMAGE_TAG="vball-analyser:cpu"
fi

mkdir -p "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/models"

if [[ "$FORCE_BUILD" == "true" ]] || ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "Building ${IMAGE_TAG} (GPU=${USE_GPU})..."
    docker build \
        --build-arg "GPU=${USE_GPU}" \
        -t "${IMAGE_TAG}" \
        "${PROJECT_ROOT}"
fi

RUN_ARGS=(
    --rm
    --name vball-analyser
    -p "${PORT}:8081"
    -v "${PROJECT_ROOT}/data:/app/data"
    -v "${PROJECT_ROOT}/models:/app/models"
)

if [[ "$DETACH" == "true" ]]; then
    RUN_ARGS+=(-d)
fi

if [[ "$USE_GPU" == "true" ]]; then
    RUN_ARGS+=(--gpus all)
fi

if [[ -n "$DEVICE_OVERRIDE" ]]; then
    RUN_ARGS+=(-e "VBALL_DEVICE=${DEVICE_OVERRIDE}")
fi

if [[ -n "$VIDEOS_DIR" ]]; then
    ABS_VIDEOS_DIR="$(cd "${VIDEOS_DIR}" && pwd)"
    RUN_ARGS+=(-v "${ABS_VIDEOS_DIR}:/videos:ro")
fi

echo "Starting ${IMAGE_TAG} on http://127.0.0.1:${PORT} (GPU=${USE_GPU})..."
exec docker run "${RUN_ARGS[@]}" "${IMAGE_TAG}"
