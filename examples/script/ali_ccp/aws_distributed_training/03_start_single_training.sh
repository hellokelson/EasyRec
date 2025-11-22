#!/bin/bash

##############################################################################
# 启动单节点分布式训练
##############################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config_single.sh"
source "$SCRIPT_DIR/cluster_info_single.sh"
source "$SCRIPT_DIR/tf_configs_single.sh"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

EXPERIMENT_NAME="deepfm_ali_ccp_${DATASET_SIZE}_single_$(date +%Y%m%d_%H%M%S)"
CONFIG_FILE="${EXPERIMENT_NAME}.config"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}启动单节点分布式训练${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}实验名称: $EXPERIMENT_NAME${NC}"
echo ""

# Download working config from remote if exists, otherwise use local
WORKING_CONFIG_REMOTE="/home/ubuntu/easyrec_data/ckpt/deepfm_ali_ccp_full_single_20251120_141350/pipeline.config"
WORKING_CONFIG_LOCAL="${HOME}/tensorboard_logs_single/deepfm_ali_ccp_full_single_20251120_105747/pipeline.config"

echo -e "${YELLOW}生成配置文件...${NC}"

# Try to get config from remote first
if ssh $SSH_OPTS ubuntu@$INSTANCE_IP "test -f $WORKING_CONFIG_REMOTE" 2>/dev/null; then
    scp $SSH_OPTS ubuntu@$INSTANCE_IP:$WORKING_CONFIG_REMOTE /tmp/template_config.config > /dev/null 2>&1
elif [ -f "$WORKING_CONFIG_LOCAL" ]; then
    cp "$WORKING_CONFIG_LOCAL" /tmp/template_config.config
else
    echo -e "${RED}错误: 找不到工作配置模板${NC}"
    exit 1
fi

# Update model_dir in config
sed "s|/workspace/ckpt/deepfm_ali_ccp_full_single_[0-9_]*|/workspace/ckpt/${EXPERIMENT_NAME}|g" /tmp/template_config.config > /tmp/$CONFIG_FILE

# Upload config
scp $SSH_OPTS /tmp/$CONFIG_FILE ubuntu@$INSTANCE_IP:/home/ubuntu/easyrec_data/configs/ > /dev/null 2>&1

echo -e "${GREEN}✓ 配置文件已生成${NC}"

echo ""
echo -e "${YELLOW}启动 PS Server...${NC}"
ssh $SSH_OPTS ubuntu@$INSTANCE_IP "docker run -d --name easyrec_ps --network host -v /home/ubuntu/easyrec_data:/workspace -e TF_CONFIG='$PS_TF_CONFIG' $DOCKER_IMAGE python -m easy_rec.python.train_eval --pipeline_config_path /workspace/configs/$CONFIG_FILE"

echo -e "${YELLOW}启动 Chief Worker...${NC}"
ssh $SSH_OPTS ubuntu@$INSTANCE_IP "docker run -d --name easyrec_chief --network host -v /home/ubuntu/easyrec_data:/workspace -e TF_CONFIG='$CHIEF_TF_CONFIG' $DOCKER_IMAGE python -m easy_rec.python.train_eval --pipeline_config_path /workspace/configs/$CONFIG_FILE"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo -e "${YELLOW}启动 Worker-$i...${NC}"
    WORKER_TF_CONFIG_VAR="WORKER_${i}_TF_CONFIG"
    ssh $SSH_OPTS ubuntu@$INSTANCE_IP "docker run -d --name easyrec_worker_$i --network host -v /home/ubuntu/easyrec_data:/workspace -e TF_CONFIG='${!WORKER_TF_CONFIG_VAR}' $DOCKER_IMAGE python -m easy_rec.python.train_eval --pipeline_config_path /workspace/configs/$CONFIG_FILE"
done

# 保存当前实验信息
echo "export CURRENT_EXPERIMENT='$EXPERIMENT_NAME'" > "$SCRIPT_DIR/current_experiment_single.sh"
echo "export CURRENT_MODEL_DIR='/workspace/ckpt/${EXPERIMENT_NAME}'" >> "$SCRIPT_DIR/current_experiment_single.sh"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}训练已启动!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}监控命令:${NC}"
echo "  bash 04_monitor_single_training.sh"
echo ""
echo -e "${BLUE}查看日志:${NC}"
echo "  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP 'docker logs -f easyrec_chief'"
