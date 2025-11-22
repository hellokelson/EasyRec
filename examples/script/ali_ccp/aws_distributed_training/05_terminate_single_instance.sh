#!/bin/bash

##############################################################################
# 终止单节点训练实例
##############################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/cluster_info_single.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}终止单节点训练实例${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${YELLOW}终止实例: $INSTANCE_ID${NC}"
aws ec2 terminate-instances --region ap-northeast-1 --instance-ids $INSTANCE_ID --output text

echo ""
echo -e "${GREEN}✓ 实例已终止${NC}"
echo ""
echo -e "${YELLOW}清理配置文件...${NC}"
rm -f "$SCRIPT_DIR/cluster_info_single.sh" "$SCRIPT_DIR/tf_configs_single.sh" "$SCRIPT_DIR/current_experiment_single.sh"

echo -e "${GREEN}✓ 清理完成${NC}"
