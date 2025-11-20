#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/single_instance_info.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}配置单节点训练环境${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${YELLOW}[1/2] 上传数据...${NC}"
EASYREC_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
DATA_DIR="$EASYREC_ROOT/examples/data/ali_ccp"

rsync -avz --progress -e "ssh $SSH_OPTS" \
    "$DATA_DIR/ali_ccp_train_${DATASET_SIZE}.csv" \
    "$DATA_DIR/ali_ccp_test_${DATASET_SIZE}.csv" \
    ubuntu@$SINGLE_INSTANCE_IP:/home/ubuntu/easyrec_data/data/

echo -e "${GREEN}✓ 数据上传完成${NC}"

echo ""
echo -e "${YELLOW}[2/2] 生成 TF_CONFIG...${NC}"

WORKER_LIST=""
for i in {0..3}; do
    PORT=$((WORKER_BASE_PORT + i))
    [ $i -gt 0 ] && WORKER_LIST+=","
    WORKER_LIST+="\"localhost:$PORT\""
done

cat > "$SCRIPT_DIR/single_tf_configs.sh" <<EOF
export PS_TF_CONFIG='{"cluster":{"ps":["localhost:$PS_PORT"],"chief":["localhost:$CHIEF_PORT"],"worker":[$WORKER_LIST]},"task":{"type":"ps","index":0}}'
export CHIEF_TF_CONFIG='{"cluster":{"ps":["localhost:$PS_PORT"],"chief":["localhost:$CHIEF_PORT"],"worker":[$WORKER_LIST]},"task":{"type":"chief","index":0}}'
export WORKER_0_TF_CONFIG='{"cluster":{"ps":["localhost:$PS_PORT"],"chief":["localhost:$CHIEF_PORT"],"worker":[$WORKER_LIST]},"task":{"type":"worker","index":0}}'
export WORKER_1_TF_CONFIG='{"cluster":{"ps":["localhost:$PS_PORT"],"chief":["localhost:$CHIEF_PORT"],"worker":[$WORKER_LIST]},"task":{"type":"worker","index":1}}'
export WORKER_2_TF_CONFIG='{"cluster":{"ps":["localhost:$PS_PORT"],"chief":["localhost:$CHIEF_PORT"],"worker":[$WORKER_LIST]},"task":{"type":"worker","index":2}}'
export WORKER_3_TF_CONFIG='{"cluster":{"ps":["localhost:$PS_PORT"],"chief":["localhost:$CHIEF_PORT"],"worker":[$WORKER_LIST]},"task":{"type":"worker","index":3}}'
EOF

echo -e "${GREEN}✓ TF_CONFIG 已生成${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}配置完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}下一步:${NC}"
echo "  bash single_03_start_training.sh"
