#!/bin/bash

##############################################################################
# 终止单节点训练实例
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
echo -e "${BLUE}终止单节点训练实例${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${RED}警告: 此操作将终止训练实例!${NC}"
echo -e "${YELLOW}实例信息:${NC}"
echo "  实例 ID: $INSTANCE_ID"
echo "  内网 IP: $INSTANCE_IP"
echo ""

read -p "确认终止实例? (yes/no): " -r
if [[ ! $REPLY =~ ^yes$ ]]; then
    echo "操作已取消"
    exit 0
fi

echo ""
echo -e "${YELLOW}终止实例...${NC}"

aws ec2 terminate-instances \
    --region $AWS_REGION \
    --instance-ids $INSTANCE_ID

echo ""
echo -e "${GREEN}✓ 实例终止命令已发送${NC}"
echo -e "${YELLOW}等待实例终止...${NC}"

aws ec2 wait instance-terminated \
    --region $AWS_REGION \
    --instance-ids $INSTANCE_ID

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}实例已终止!${NC}"
echo -e "${GREEN}========================================${NC}"

echo ""
echo -e "${YELLOW}清理配置文件...${NC}"
rm -f "$SCRIPT_DIR/cluster_info_single.sh"
rm -f "$SCRIPT_DIR/current_experiment_single.sh"
echo -e "${GREEN}✓ 清理完成${NC}"
