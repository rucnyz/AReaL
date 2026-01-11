#!/bin/bash
# 批量修改 docker-compose.yaml 使用共享网络

NETWORK_NAME="shared_rl_tasks"
TASKS_DIR="./rl_tasks_t100"

# 1. 创建共享网络（如果不存在）
echo "Creating shared network: $NETWORK_NAME"
docker network create $NETWORK_NAME 2>/dev/null || echo "Network already exists"

# 2. 批量修改所有 docker-compose.yaml
echo "Modifying docker-compose.yaml files..."

count=0
skipped=0
find $TASKS_DIR -name "docker-compose.yaml" | while read file; do
    # 检查是否已经修改过
    if grep -q "networks:" "$file"; then
        echo "Skipping: $file"
        continue
    fi

    # 添加网络配置
    cat >> "$file" << EOF
    networks:
      - shared

networks:
  shared:
    external: true
    name: $NETWORK_NAME
EOF

    echo "Modified: $file"
done

echo ""
echo "Done! Checking results..."
modified=$(grep -l "shared_rl_tasks" $TASKS_DIR/*/docker-compose.yaml 2>/dev/null | wc -l)
echo "Files with shared network: $modified"