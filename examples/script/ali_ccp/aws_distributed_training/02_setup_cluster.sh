#!/bin/bash

##############################################################################
# 配置 EC2 集群 - 分发代码和数据
##############################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/cluster_info.sh"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}配置 EC2 分布式训练集群${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# SSH 配置
SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

# 检查 SSH key
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}错误: SSH key 不存在: $SSH_KEY${NC}"
    echo -e "${YELLOW}请确保 SSH key 文件存在并设置正确的权限: chmod 400 $SSH_KEY${NC}"
    exit 1
fi

# 所有节点 IP
ALL_IPS=("$PS_IP" "$CHIEF_IP" "${WORKER_IPS[@]}")

echo -e "${YELLOW}[1/5] 检查实例连接性...${NC}"
for ip in "${ALL_IPS[@]}"; do
    echo "  测试连接: $ip"
    if ssh $SSH_OPTS ubuntu@$ip "echo 'OK'" 2>/dev/null; then
        echo -e "    ${GREEN}✓ $ip 连接成功${NC}"
    else
        echo -e "    ${RED}✗ $ip 连接失败${NC}"
        echo -e "${YELLOW}    提示: 可能实例还在初始化，请稍等片刻后重试${NC}"
        exit 1
    fi
done

echo ""
echo -e "${YELLOW}[2/5] 创建最小化工作目录...${NC}"
EASYREC_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

for ip in "${ALL_IPS[@]}"; do
    echo "  创建目录结构: $ip"
    ssh $SSH_OPTS ubuntu@$ip "mkdir -p /home/ubuntu/easyrec_data/configs /home/ubuntu/easyrec_data/data /home/ubuntu/easyrec_data/ckpt"
    echo -e "    ${GREEN}✓ 完成${NC}"
done

echo ""
echo -e "${YELLOW}[3/5] 分发 ${DATASET_SIZE} 数据集...${NC}"
DATA_DIR="$EASYREC_ROOT/examples/data/ali_ccp"

for ip in "${ALL_IPS[@]}"; do
    echo "  分发到: $ip"
    rsync -avz --progress -e "ssh $SSH_OPTS" \
        "$DATA_DIR/ali_ccp_train_${DATASET_SIZE}.csv" \
        "$DATA_DIR/ali_ccp_test_${DATASET_SIZE}.csv" \
        ubuntu@$ip:/home/ubuntu/easyrec_data/data/
    echo -e "    ${GREEN}✓ 完成${NC}"
done

echo ""
echo -e "${YELLOW}[4/5] 生成训练配置文件...${NC}"

# 生成 TF_CONFIG 配置
WORKER_LIST=""
for i in "${!WORKER_IPS[@]}"; do
    PORT=$((WORKER_BASE_PORT + i))
    if [ $i -gt 0 ]; then
        WORKER_LIST+=","
    fi
    WORKER_LIST+="\"${WORKER_IPS[$i]}:$PORT\""
done

# 为每个节点生成配置
# PS Server
PS_TF_CONFIG="{\"cluster\":{\"ps\":[\"${PS_IP}:${PS_PORT}\"],\"chief\":[\"${CHIEF_IP}:${CHIEF_PORT}\"],\"worker\":[${WORKER_LIST}]},\"task\":{\"type\":\"ps\",\"index\":0}}"

# Chief
CHIEF_TF_CONFIG="{\"cluster\":{\"ps\":[\"${PS_IP}:${PS_PORT}\"],\"chief\":[\"${CHIEF_IP}:${CHIEF_PORT}\"],\"worker\":[${WORKER_LIST}]},\"task\":{\"type\":\"chief\",\"index\":0}}"

# Workers
WORKER_TF_CONFIGS=()
for i in "${!WORKER_IPS[@]}"; do
    WORKER_TF_CONFIGS+=("{\"cluster\":{\"ps\":[\"${PS_IP}:${PS_PORT}\"],\"chief\":[\"${CHIEF_IP}:${CHIEF_PORT}\"],\"worker\":[${WORKER_LIST}]},\"task\":{\"type\":\"worker\",\"index\":$i}}")
done

# 保存配置
cat > "$SCRIPT_DIR/tf_configs.sh" << EOF
#!/bin/bash
# TF_CONFIG 配置

export PS_TF_CONFIG='$PS_TF_CONFIG'
export CHIEF_TF_CONFIG='$CHIEF_TF_CONFIG'
export WORKER_TF_CONFIGS=($(printf "'%s' " "${WORKER_TF_CONFIGS[@]}"))
EOF

echo -e "${GREEN}✓ TF_CONFIG 配置已生成${NC}"

echo ""
echo -e "${YELLOW}[5/5] 创建模型输出目录...${NC}"
ssh $SSH_OPTS ubuntu@$CHIEF_IP "mkdir -p /home/ubuntu/easyrec_data/ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps"
echo -e "${GREEN}✓ 模型目录已创建: /home/ubuntu/easyrec_data/ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}集群配置完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}下一步: bash 03_start_training.sh${NC}"
