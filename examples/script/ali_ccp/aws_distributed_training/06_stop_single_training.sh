#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config_single.sh"
source "$SCRIPT_DIR/cluster_info_single.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}停止单节点训练${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${YELLOW}停止训练容器...${NC}"
ssh $SSH_OPTS ubuntu@$INSTANCE_IP "docker stop \$(docker ps -q --filter 'name=easyrec') 2>/dev/null && docker rm \$(docker ps -aq --filter 'name=easyrec') 2>/dev/null" || true

echo -e "${GREEN}✓ 训练已停止${NC}"
echo ""
echo -e "${YELLOW}注意: TensorBoard 容器仍在运行${NC}"
echo -e "${YELLOW}如需停止 TensorBoard: docker stop tensorboard${NC}"
