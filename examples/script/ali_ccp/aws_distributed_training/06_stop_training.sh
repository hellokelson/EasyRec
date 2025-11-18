#!/bin/bash

##############################################################################
# 停止分布式训练
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
echo -e "${BLUE}停止分布式训练${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# SSH 配置
SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

echo -e "${YELLOW}[1/3] 停止 Workers...${NC}"
for i in "${!WORKER_IPS[@]}"; do
    ip="${WORKER_IPS[$i]}"
    echo "  停止 Worker-$i: $ip"
    ssh $SSH_OPTS ubuntu@$ip "docker stop easyrec_worker_$i && docker rm easyrec_worker_$i" || true
done
echo -e "${GREEN}✓ Workers 已停止${NC}"

echo ""
echo -e "${YELLOW}[2/3] 停止 Chief Worker...${NC}"
ssh $SSH_OPTS ubuntu@$CHIEF_IP "docker stop easyrec_chief && docker rm easyrec_chief" || true
echo -e "${GREEN}✓ Chief Worker 已停止${NC}"

echo ""
echo -e "${YELLOW}[3/3] 停止 PS Server...${NC}"
ssh $SSH_OPTS ubuntu@$PS_IP "docker stop easyrec_ps_0 && docker rm easyrec_ps_0" || true
echo -e "${GREEN}✓ PS Server 已停止${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}训练已停止!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}注意: EC2 实例仍在运行，如需终止实例请运行:${NC}"
echo "  ${GREEN}bash 07_terminate_instances.sh${NC}"
