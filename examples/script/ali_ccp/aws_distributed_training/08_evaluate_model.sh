#!/bin/bash

##############################################################################
# 模型评估脚本 - 在训练完成后评估模型性能
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
echo -e "${BLUE}模型评估${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# SSH 配置
SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

# 配置文件路径
CONFIG_NAME="deepfm_on_ali_ccp_${DATASET_SIZE}_ps.config"
REMOTE_CONFIG_PATH="/home/ubuntu/easyrec_data/configs/${CONFIG_NAME}"
REMOTE_MODEL_DIR="/home/ubuntu/easyrec_data/ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps"
EVAL_RESULT_PATH="/home/ubuntu/easyrec_data/eval_result_${DATASET_SIZE}.txt"

echo -e "${YELLOW}[1/4] 检查训练是否完成...${NC}"

# 检查训练是否完成
FINISHED=$(ssh $SSH_OPTS ubuntu@$CHIEF_IP \
  "docker logs easyrec_chief 2>&1 | grep -c 'Train and evaluate finish'" 2>/dev/null || echo "0")

if [ "$FINISHED" -eq 0 ]; then
    echo -e "${RED}✗ 训练尚未完成，无法评估${NC}"
    echo ""
    echo "请先等待训练完成，可以使用以下命令检查："
    echo "  bash check_training_status.sh"
    exit 1
fi

echo -e "${GREEN}✓ 训练已完成${NC}"
echo ""

echo -e "${YELLOW}[2/4] 检查模型文件...${NC}"

# 检查checkpoint文件是否存在
CKPT_EXISTS=$(ssh $SSH_OPTS ubuntu@$CHIEF_IP \
  "ls ${REMOTE_MODEL_DIR}/checkpoint 2>/dev/null" || echo "")

if [ -z "$CKPT_EXISTS" ]; then
    echo -e "${RED}✗ 未找到checkpoint文件${NC}"
    exit 1
fi

# 获取最新的checkpoint
LATEST_CKPT=$(ssh $SSH_OPTS ubuntu@$CHIEF_IP \
  "cat ${REMOTE_MODEL_DIR}/checkpoint | grep 'model_checkpoint_path' | head -1 | cut -d'\"' -f2")

echo -e "${GREEN}✓ 找到模型: $LATEST_CKPT${NC}"
echo ""

echo -e "${YELLOW}[3/4] 运行模型评估...${NC}"
echo ""

# 在Chief节点运行评估
ssh $SSH_OPTS ubuntu@$CHIEF_IP << EVALEOF
echo "Running evaluation on Chief node..."
echo "Config: $REMOTE_CONFIG_PATH"
echo "Model: $LATEST_CKPT"
echo ""

# 使用Docker运行评估
docker run --rm \
  --network host \
  -v /home/ubuntu/easyrec_data:/workspace \
  ${DOCKER_IMAGE} \
  python3 -m easy_rec.python.eval \
  --pipeline_config_path=/workspace/configs/${CONFIG_NAME} \
  --model_dir=/workspace/ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps \
  --eval_result_path=/workspace/eval_result_${DATASET_SIZE}.txt

echo ""
echo "Evaluation completed!"
EVALEOF

EVAL_STATUS=$?

echo ""
if [ $EVAL_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ 评估完成${NC}"
else
    echo -e "${RED}✗ 评估失败，错误码: $EVAL_STATUS${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}[4/4] 获取评估结果...${NC}"
echo ""

# 下载评估结果
LOCAL_EVAL_RESULT="${SCRIPT_DIR}/eval_result_${DATASET_SIZE}.txt"
scp $SSH_OPTS ubuntu@$CHIEF_IP:${EVAL_RESULT_PATH} ${LOCAL_EVAL_RESULT} 2>/dev/null

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}评估结果${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 显示评估结果
if [ -f "${LOCAL_EVAL_RESULT}" ]; then
    cat ${LOCAL_EVAL_RESULT}
    echo ""
    echo -e "${GREEN}评估结果已保存到: ${LOCAL_EVAL_RESULT}${NC}"
else
    echo -e "${YELLOW}获取评估结果失败，请手动查看:${NC}"
    echo "  ssh -i $SSH_KEY ubuntu@$CHIEF_IP 'cat ${EVAL_RESULT_PATH}'"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}模型评估完成!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 显示模型信息
echo -e "${BLUE}模型信息:${NC}"
echo "  配置文件: ${CONFIG_NAME}"
echo "  模型目录: ${REMOTE_MODEL_DIR}"
echo "  Checkpoint: ${LATEST_CKPT}"
echo ""

# 显示关键指标（如果有）
if [ -f "${LOCAL_EVAL_RESULT}" ]; then
    echo -e "${BLUE}关键指标:${NC}"
    grep -E "(auc|gauc|accuracy|precision|recall|f1)" ${LOCAL_EVAL_RESULT} 2>/dev/null || echo "  请查看评估结果文件"
fi
