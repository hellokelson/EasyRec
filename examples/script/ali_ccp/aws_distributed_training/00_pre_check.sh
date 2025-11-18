#!/bin/bash

##############################################################################
# 启动前环境检查
##############################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}EasyRec AWS 分布式训练 - 环境检查${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查 AWS CLI
echo -e "${YELLOW}[1/8] 检查 AWS CLI...${NC}"
if command -v aws &> /dev/null; then
    AWS_VERSION=$(aws --version 2>&1 | cut -d' ' -f1)
    echo -e "  ${GREEN}✓ AWS CLI 已安装: $AWS_VERSION${NC}"
    ((PASS++))
else
    echo -e "  ${RED}✗ AWS CLI 未安装${NC}"
    echo -e "  ${YELLOW}安装: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html${NC}"
    ((FAIL++))
fi

# 检查 AWS 凭证
echo ""
echo -e "${YELLOW}[2/8] 检查 AWS 凭证...${NC}"
if aws sts get-caller-identity &> /dev/null; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    echo -e "  ${GREEN}✓ AWS 凭证有效${NC}"
    echo -e "  ${BLUE}Account ID: $ACCOUNT_ID${NC}"
    ((PASS++))
else
    echo -e "  ${RED}✗ AWS 凭证无效或未配置${NC}"
    echo -e "  ${YELLOW}配置: aws configure${NC}"
    ((FAIL++))
fi

# 检查 AWS 区域
echo ""
echo -e "${YELLOW}[3/8] 检查 AWS 区域...${NC}"
CURRENT_REGION=$(aws configure get region)
echo -e "  ${BLUE}当前区域: $CURRENT_REGION${NC}"
echo -e "  ${BLUE}配置区域: $AWS_REGION${NC}"
if [ "$CURRENT_REGION" != "$AWS_REGION" ]; then
    echo -e "  ${YELLOW}⚠ 区域不匹配，将使用配置的区域: $AWS_REGION${NC}"
fi
((PASS++))

# 检查 SSH Key
echo ""
echo -e "${YELLOW}[4/8] 检查 SSH Key...${NC}"
SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
if [ -f "$SSH_KEY" ]; then
    echo -e "  ${GREEN}✓ SSH Key 存在: $SSH_KEY${NC}"
    KEY_PERM=$(stat -c %a "$SSH_KEY" 2>/dev/null || stat -f %A "$SSH_KEY")
    if [ "$KEY_PERM" == "400" ] || [ "$KEY_PERM" == "600" ]; then
        echo -e "  ${GREEN}✓ 权限正确: $KEY_PERM${NC}"
        ((PASS++))
    else
        echo -e "  ${YELLOW}⚠ 权限不正确: $KEY_PERM (应该是 400)${NC}"
        echo -e "  ${YELLOW}修复: chmod 400 $SSH_KEY${NC}"
        ((FAIL++))
    fi
else
    echo -e "  ${RED}✗ SSH Key 不存在: $SSH_KEY${NC}"
    echo -e "  ${YELLOW}创建 Key:${NC}"
    echo -e "  ${GREEN}aws ec2 create-key-pair --region $AWS_REGION --key-name $KEY_NAME --query 'KeyMaterial' --output text > $SSH_KEY${NC}"
    echo -e "  ${GREEN}chmod 400 $SSH_KEY${NC}"
    ((FAIL++))
fi

# 检查数据文件
echo ""
echo -e "${YELLOW}[5/8] 检查数据文件...${NC}"
EASYREC_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
TRAIN_FILE="$EASYREC_ROOT/examples/data/ali_ccp/ali_ccp_train_${DATASET_SIZE}.csv"
TEST_FILE="$EASYREC_ROOT/examples/data/ali_ccp/ali_ccp_test_${DATASET_SIZE}.csv"

if [ -f "$TRAIN_FILE" ]; then
    TRAIN_SIZE=$(du -h "$TRAIN_FILE" | cut -f1)
    echo -e "  ${GREEN}✓ 训练数据: $TRAIN_SIZE${NC}"
    ((PASS++))
else
    echo -e "  ${RED}✗ 训练数据不存在: $TRAIN_FILE${NC}"
    ((FAIL++))
fi

if [ -f "$TEST_FILE" ]; then
    TEST_SIZE=$(du -h "$TEST_FILE" | cut -f1)
    echo -e "  ${GREEN}✓ 测试数据: $TEST_SIZE${NC}"
    ((PASS++))
else
    echo -e "  ${RED}✗ 测试数据不存在: $TEST_FILE${NC}"
    ((FAIL++))
fi

# 检查 Docker
echo ""
echo -e "${YELLOW}[6/8] 检查 Docker (本地 TensorBoard 需要)...${NC}"
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | tr -d ',')
    echo -e "  ${GREEN}✓ Docker 已安装: $DOCKER_VERSION${NC}"
    ((PASS++))
else
    echo -e "  ${YELLOW}⚠ Docker 未安装 (本地 TensorBoard 需要)${NC}"
    echo -e "  ${YELLOW}安装: https://docs.docker.com/get-docker/${NC}"
fi

# 检查 VPC/Subnet
echo ""
echo -e "${YELLOW}[7/8] 检查 VPC 和 Subnet...${NC}"
echo -e "  ${BLUE}VPC ID: $VPC_ID${NC}"
echo -e "  ${BLUE}Subnets: ${SUBNET_IDS[@]}${NC}"
echo -e "  ${BLUE}Security Group: $SECURITY_GROUP${NC}"
echo -e "  ${YELLOW}提示: 将在启动实例时验证${NC}"
((PASS++))

# 检查配置
echo ""
echo -e "${YELLOW}[8/8] 检查训练配置...${NC}"
echo -e "  ${BLUE}数据集大小: $DATASET_SIZE${NC}"
echo -e "  ${BLUE}Worker 数量: $NUM_WORKERS${NC}"
echo -e "  ${BLUE}训练步数: $NUM_STEPS${NC}"
echo -e "  ${BLUE}Batch Size: $BATCH_SIZE${NC}"
echo -e "  ${BLUE}全局 Batch Size: $((BATCH_SIZE * (NUM_WORKERS + 1)))${NC}"
echo -e "  ${BLUE}PS 实例类型: $PS_INSTANCE_TYPE${NC}"
echo -e "  ${BLUE}Chief 实例类型: $CHIEF_INSTANCE_TYPE${NC}"
echo -e "  ${BLUE}Worker 实例类型: $WORKER_INSTANCE_TYPE${NC}"
((PASS++))

# 总结
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}检查完成${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}通过: $PASS 项${NC}"
if [ $FAIL -gt 0 ]; then
    echo -e "${RED}失败: $FAIL 项${NC}"
    echo ""
    echo -e "${YELLOW}请修复失败项后再启动训练${NC}"
    exit 1
else
    echo -e "${GREEN}失败: 0 项${NC}"
    echo ""
    echo -e "${GREEN}✓ 所有检查通过，可以开始训练!${NC}"
    echo ""
    echo -e "${BLUE}下一步:${NC}"
    echo "  1. 修改 config.sh (如需调整配置)"
    echo "  2. bash 01_launch_ec2_instances.sh"
    echo "  3. 等待 5 分钟实例初始化"
    echo "  4. bash 02_setup_cluster.sh"
    echo "  5. bash 03_start_training.sh"
    echo "  6. bash 05_setup_local_tensorboard.sh"
    echo ""
    echo -e "${YELLOW}详细使用说明请查看: README.md${NC}"
fi
