#!/bin/bash

##############################################################################
# 配置单节点训练环境 - 分发数据
##############################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config_single.sh"
source "$SCRIPT_DIR/cluster_info_single.sh"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}配置单节点训练环境${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# SSH 配置
SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

# 检查 SSH key
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}错误: SSH key 不存在: $SSH_KEY${NC}"
    exit 1
fi

echo -e "${YELLOW}[1/3] 检查实例连接性...${NC}"
if ssh $SSH_OPTS ubuntu@$INSTANCE_IP "echo 'OK'" 2>/dev/null; then
    echo -e "${GREEN}✓ 连接成功${NC}"
else
    echo -e "${RED}✗ 连接失败${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}[2/3] 分发 ${DATASET_SIZE} 数据集...${NC}"
EASYREC_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
DATA_DIR="$EASYREC_ROOT/examples/data/ali_ccp"

rsync -avz --progress -e "ssh $SSH_OPTS" \
    "$DATA_DIR/ali_ccp_train_${DATASET_SIZE}.csv" \
    "$DATA_DIR/ali_ccp_test_${DATASET_SIZE}.csv" \
    ubuntu@$INSTANCE_IP:/home/ubuntu/easyrec_data/data/

echo -e "${GREEN}✓ 数据分发完成${NC}"

echo ""
echo -e "${YELLOW}[3/3] 生成训练配置文件...${NC}"

# 生成 TF_CONFIG 配置 (所有服务在 localhost)
WORKER_LIST=""
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    PORT=$((WORKER_BASE_PORT + i))
    if [ $i -gt 0 ]; then
        WORKER_LIST+=","
    fi
    WORKER_LIST+="\"localhost:$PORT\""
done

# 保存配置
cat > "$SCRIPT_DIR/tf_configs_single.sh" <<EOF
#!/bin/bash
export PS_TF_CONFIG='{"cluster":{"ps":["localhost:${PS_PORT}"],"chief":["localhost:${CHIEF_PORT}"],"worker":[${WORKER_LIST}]},"task":{"type":"ps","index":0}}'
export CHIEF_TF_CONFIG='{"cluster":{"ps":["localhost:${PS_PORT}"],"chief":["localhost:${CHIEF_PORT}"],"worker":[${WORKER_LIST}]},"task":{"type":"chief","index":0}}'
EOF

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "export WORKER_${i}_TF_CONFIG='{\"cluster\":{\"ps\":[\"localhost:${PS_PORT}\"],\"chief\":[\"localhost:${CHIEF_PORT}\"],\"worker\":[${WORKER_LIST}]},\"task\":{\"type\":\"worker\",\"index\":$i}}'" >> "$SCRIPT_DIR/tf_configs_single.sh"
done

echo -e "${GREEN}✓ 配置文件已生成${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}配置完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}下一步:${NC}"
echo "  bash 03_start_single_training.sh"
