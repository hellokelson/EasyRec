#!/bin/bash

##############################################################################
# 同步 Checkpoint 文件
#
# 在 PS 模式下，模型权重保存在 PS Server 上，需要复制到 Chief 节点才能评估
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
echo -e "${BLUE}同步 Checkpoint 文件${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# SSH 配置
SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

# 使用当前实验名称
if [ -f "$SCRIPT_DIR/current_experiment.sh" ]; then
    source "$SCRIPT_DIR/current_experiment.sh"
    CKPT_DIR=$(basename "$CURRENT_MODEL_DIR")
else
    echo -e "${RED}✗ 未找到当前实验信息${NC}"
    exit 1
fi

echo -e "${BLUE}实验: $CKPT_DIR${NC}"
echo ""

echo -e "${YELLOW}[1/3] 检查 PS Server checkpoint 文件...${NC}"

# 检查 PS 节点是否有 checkpoint 文件
PS_CKPTS=$(ssh $SSH_OPTS ubuntu@$PS_IP \
  "ls /home/ubuntu/easyrec_data/ckpt/${CKPT_DIR}/model.ckpt-*.data-* 2>/dev/null | wc -l" || echo "0")

if [ "$PS_CKPTS" -eq 0 ]; then
    echo -e "${RED}✗ PS Server 没有 checkpoint 文件${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 找到 $PS_CKPTS 个 checkpoint 数据文件${NC}"

echo ""
echo -e "${YELLOW}[2/3] 复制 checkpoint 文件到本地...${NC}"

# 创建临时目录
TMP_DIR="/tmp/ckpt_${CURRENT_EXPERIMENT}_$$"
mkdir -p "$TMP_DIR"

# 从 PS 复制到本地 (使用 sudo 处理权限)
ssh $SSH_OPTS ubuntu@$PS_IP "sudo chmod -R 755 /home/ubuntu/easyrec_data/ckpt/${CKPT_DIR}"
scp $SSH_OPTS \
  "ubuntu@$PS_IP:/home/ubuntu/easyrec_data/ckpt/${CKPT_DIR}/model.ckpt-*.data-*" \
  "ubuntu@$PS_IP:/home/ubuntu/easyrec_data/ckpt/${CKPT_DIR}/model.ckpt-*.index" \
  "$TMP_DIR/" 2>/dev/null

COPIED_FILES=$(ls "$TMP_DIR" | wc -l)
echo -e "${GREEN}✓ 已复制 $COPIED_FILES 个文件到本地${NC}"

echo ""
echo -e "${YELLOW}[3/3] 上传 checkpoint 文件到 Chief...${NC}"

# 从本地复制到 Chief
ssh $SSH_OPTS ubuntu@$CHIEF_IP "sudo mkdir -p /home/ubuntu/easyrec_data/ckpt/${CKPT_DIR} && sudo chmod 777 /home/ubuntu/easyrec_data/ckpt/${CKPT_DIR}"
scp $SSH_OPTS "$TMP_DIR"/* \
  ubuntu@$CHIEF_IP:/home/ubuntu/easyrec_data/ckpt/${CKPT_DIR}/

# 清理临时目录
rm -rf "$TMP_DIR"

echo -e "${GREEN}✓ Checkpoint 文件已同步到 Chief${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}同步完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}下一步: bash 08_evaluate_model.sh (评估模型)${NC}"
