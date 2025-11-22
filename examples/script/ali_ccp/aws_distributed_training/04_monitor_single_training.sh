#!/bin/bash

##############################################################################
# 监控单节点训练状态
##############################################################################

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
echo -e "${BLUE}单节点训练监控${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查训练状态
FINISHED=$(ssh $SSH_OPTS ubuntu@$INSTANCE_IP "docker logs easyrec_chief 2>&1 | grep -c 'Train and evaluate finish' || echo 0" 2>/dev/null)
FINISHED=${FINISHED:-0}

if [ "$FINISHED" -gt 0 ] 2>/dev/null; then
    echo -e "${GREEN}训练状态: ✅ FINISHED${NC}"
else
    echo -e "${YELLOW}训练状态: ⏳ RUNNING${NC}"
fi

echo ""
echo -e "${YELLOW}[容器状态]${NC}"
ssh $SSH_OPTS ubuntu@$INSTANCE_IP "docker ps --format 'table {{.Names}}\t{{.Status}}'"

echo ""
echo -e "${YELLOW}[最近日志]${NC}"
ssh $SSH_OPTS ubuntu@$INSTANCE_IP "docker logs easyrec_chief 2>&1 | tail -20"

echo ""
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}监控命令:${NC}"
echo "  查看 Chief 实时日志:"
echo "    ssh -i $SSH_KEY ubuntu@$INSTANCE_IP 'docker logs -f easyrec_chief'"
echo ""
echo "  刷新监控:"
echo "    bash 04_monitor_single_training.sh"
