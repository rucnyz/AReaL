#!/bin/bash
# Docker cleanup script for terminal bench
# Usage: ./cleanup_docker.sh [--all] [--networks]

set -e

# Count before cleanup
TB_COUNT=$(docker ps -aq --filter "name=tb__" 2>/dev/null | wc -l)
ALL_COUNT=$(docker ps -aq 2>/dev/null | wc -l)

echo "=== Docker Cleanup ==="
echo "tb__ containers: $TB_COUNT"
echo "All containers: $ALL_COUNT"
echo ""

# Parse arguments
CLEAN_ALL=false
CLEAN_NETWORKS=false
for arg in "$@"; do
    case $arg in
        --all) CLEAN_ALL=true ;;
        --networks) CLEAN_NETWORKS=true ;;
    esac
done

# Clean tb__ containers
if [ "$TB_COUNT" -gt 0 ]; then
    echo "Removing tb__ containers..."
    docker ps -aq --filter "name=tb__" | xargs -r docker rm -f 2>/dev/null
    echo "Done: removed $TB_COUNT tb__ containers"
else
    echo "No tb__ containers to remove"
fi

# Clean all containers if --all flag
if [ "$CLEAN_ALL" = true ]; then
    OTHER_COUNT=$((ALL_COUNT - TB_COUNT))
    if [ "$OTHER_COUNT" -gt 0 ]; then
        echo ""
        echo "Removing all other containers..."
        docker ps -aq | xargs -r docker rm -f 2>/dev/null
        echo "Done: removed all containers"
    fi
fi

# Clean networks if --networks flag
if [ "$CLEAN_NETWORKS" = true ]; then
    echo ""
    echo "Removing task-related networks (keeping system & shared)..."
    docker network ls --format "{{.Name}}" | grep -vE "^(bridge|host|none|my_default|shared_rl_tasks)$" | xargs -r docker network rm 2>/dev/null || true
    echo "Done"
fi

# Summary
echo ""
echo "=== After Cleanup ==="
echo "Remaining containers: $(docker ps -aq 2>/dev/null | wc -l)"
echo "Remaining networks: $(docker network ls -q 2>/dev/null | wc -l)"
