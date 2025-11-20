#!/bin/bash

##############################################################################
# 监控分布式训练状态
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

# SSH 配置
SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

clear

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}分布式训练监控${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 显示当前实验
if [ -f "$SCRIPT_DIR/current_experiment.sh" ]; then
    source "$SCRIPT_DIR/current_experiment.sh"
    echo -e "${BLUE}当前实验: $CURRENT_EXPERIMENT${NC}"
else
    echo -e "${YELLOW}未找到当前实验信息${NC}"
fi

# 检查训练是否完成
FINISHED=$(ssh $SSH_OPTS ubuntu@$CHIEF_IP \
  "docker logs easyrec_chief 2>&1 | grep -c 'Train and evaluate finish' || echo 0" 2>/dev/null)

# 确保 FINISHED 是数字
FINISHED=${FINISHED:-0}

if [ "$FINISHED" -gt 0 ] 2>/dev/null; then
    echo -e "${GREEN}训练状态: ✅ FINISHED${NC}"
    
    # 检查 DONE marker
    if [ -f "$SCRIPT_DIR/current_experiment.sh" ]; then
        DONE_FILE=$(ssh $SSH_OPTS ubuntu@$CHIEF_IP \
          "ls /home/ubuntu/easyrec_data/ckpt/${CURRENT_EXPERIMENT}/ESTIMATOR_TRAIN_DONE 2>/dev/null" || echo "")
        if [ -n "$DONE_FILE" ]; then
            echo -e "${GREEN}DONE Marker: ✅ 存在${NC}"
        fi
    fi
    
    echo ""
    echo -e "${YELLOW}下一步操作:${NC}"
    echo "  1. bash 10_sync_checkpoints.sh  # 同步 checkpoint"
    echo "  2. bash 08_evaluate_model.sh    # 评估模型"
else
    echo -e "${YELLOW}训练状态: ⏳ RUNNING${NC}"
fi

echo ""

# 检查容器状态
echo -e "${YELLOW}[容器状态]${NC}"
echo ""

echo -e "${BLUE}PS Server ($PS_IP):${NC}"
ssh $SSH_OPTS ubuntu@$PS_IP "docker ps --filter name=easyrec_ps_0 --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
echo ""

echo -e "${BLUE}Chief Worker ($CHIEF_IP):${NC}"
ssh $SSH_OPTS ubuntu@$CHIEF_IP "docker ps --filter name=easyrec_chief --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
echo ""

for i in "${!WORKER_IPS[@]}"; do
    ip="${WORKER_IPS[$i]}"
    echo -e "${BLUE}Worker-$i ($ip):${NC}"
    ssh $SSH_OPTS ubuntu@$ip "docker ps --filter name=easyrec_worker_$i --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
    echo ""
done

# 获取训练进度
echo -e "${YELLOW}[训练进度 - Chief]${NC}"
echo ""

RECENT_LOGS=$(ssh $SSH_OPTS ubuntu@$CHIEF_IP "docker logs easyrec_chief 2>&1 | grep -E '(global step|loss|auc)' | tail -20" 2>/dev/null)

if [ -n "$RECENT_LOGS" ]; then
    echo "$RECENT_LOGS"
    
    # 提取最新步数和进度
    LATEST_STEP=$(echo "$RECENT_LOGS" | grep -oP 'global step \K[0-9]+' | tail -1)
    if [ -n "$LATEST_STEP" ] && [ "$NUM_STEPS" -gt 0 ]; then
        PROGRESS=$((LATEST_STEP * 100 / NUM_STEPS))
        echo ""
        echo -e "${BLUE}进度: ${LATEST_STEP}/${NUM_STEPS} (${PROGRESS}%)${NC}"
    fi
else
    echo "暂无训练日志"
fi

echo ""

# 资源使用
echo -e "${YELLOW}[资源使用]${NC}"
echo ""

echo -e "${BLUE}PS Server:${NC}"
ssh $SSH_OPTS ubuntu@$PS_IP "docker stats easyrec_ps_0 --no-stream --format 'CPU: {{.CPUPerc}}\tMemory: {{.MemUsage}}'"
echo ""

echo -e "${BLUE}Chief Worker:${NC}"
ssh $SSH_OPTS ubuntu@$CHIEF_IP "docker stats easyrec_chief --no-stream --format 'CPU: {{.CPUPerc}}\tMemory: {{.MemUsage}}'"
echo ""

for i in "${!WORKER_IPS[@]}"; do
    ip="${WORKER_IPS[$i]}"
    echo -e "${BLUE}Worker-$i:${NC}"
    ssh $SSH_OPTS ubuntu@$ip "docker stats easyrec_worker_$i --no-stream --format 'CPU: {{.CPUPerc}}\tMemory: {{.MemUsage}}'"
    echo ""
done

echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}监控命令:${NC}"
echo "  查看 Chief 实时日志:"
echo "    ${GREEN}ssh -i $SSH_KEY ubuntu@$CHIEF_IP 'docker logs -f easyrec_chief'${NC}"
echo ""
echo "  查看训练指标:"
echo "    ${GREEN}ssh -i $SSH_KEY ubuntu@$CHIEF_IP 'docker logs easyrec_chief 2>&1 | grep -E \"(global step|loss|auc)\"'${NC}"
echo ""
echo "  刷新监控:"
echo "    ${GREEN}bash 04_monitor_training.sh${NC}"
echo ""
echo "  停止训练:"
echo "    ${GREEN}bash 06_stop_training.sh${NC}"
