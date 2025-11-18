#!/bin/bash

##############################################################################
# 终止 EC2 实例
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
echo -e "${BLUE}终止 EC2 实例${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${RED}警告: 此操作将终止所有训练实例!${NC}"
echo -e "${YELLOW}实例列表:${NC}"
echo "  PS Server:    $PS_INSTANCE_ID ($PS_IP)"
echo "  Chief Worker: $CHIEF_INSTANCE_ID ($CHIEF_IP)"
for i in "${!WORKER_INSTANCE_IDS[@]}"; do
    echo "  Worker-$i:     ${WORKER_INSTANCE_IDS[$i]} (${WORKER_IPS[$i]})"
done
echo ""

read -p "确认终止所有实例? (yes/no): " -r
if [[ ! $REPLY =~ ^yes$ ]]; then
    echo "操作已取消"
    exit 0
fi

echo ""
echo -e "${YELLOW}终止实例...${NC}"

aws ec2 terminate-instances \
    --region $AWS_REGION \
    --instance-ids $ALL_INSTANCE_IDS

echo ""
echo -e "${GREEN}✓ 实例终止命令已发送${NC}"
echo -e "${YELLOW}等待实例终止...${NC}"

aws ec2 wait instance-terminated \
    --region $AWS_REGION \
    --instance-ids $ALL_INSTANCE_IDS

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有实例已终止!${NC}"
echo -e "${GREEN}========================================${NC}"
