#!/bin/bash

##############################################################################
# 自动等待训练完成并评估模型
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

# 获取当前实验
if [ -f "$SCRIPT_DIR/current_experiment.sh" ]; then
    source "$SCRIPT_DIR/current_experiment.sh"
else
    echo -e "${RED}✗ 未找到当前实验信息${NC}"
    exit 1
fi

# 配置
CHECK_INTERVAL=60  # 每60秒检查一次
MAX_WAIT_TIME=14400  # 最多等待4小时 (14400秒)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}自动等待训练完成并评估${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}当前实验: $CURRENT_EXPERIMENT${NC}"
echo "检查间隔: ${CHECK_INTERVAL}秒"
echo "最长等待: $((MAX_WAIT_TIME / 3600))小时"
echo ""

ELAPSED_TIME=0
LAST_STEP=0

while [ $ELAPSED_TIME -lt $MAX_WAIT_TIME ]; do
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] 检查训练状态...${NC}"

    # 检查训练是否完成
    FINISHED=$(ssh $SSH_OPTS ubuntu@$CHIEF_IP \
      "docker logs easyrec_chief 2>&1 | grep -c 'Train and evaluate finish' || echo 0" 2>/dev/null)
    
    # 确保 FINISHED 是数字
    FINISHED=${FINISHED:-0}

    if [ "$FINISHED" -gt 0 ] 2>/dev/null; then
        echo -e "${GREEN}✅ 训练已完成!${NC}"
        echo ""

        # 获取最终步数
        FINAL_STEP=$(ssh $SSH_OPTS ubuntu@$CHIEF_IP \
          "docker logs easyrec_chief 2>&1 | grep 'global step' | tail -1" 2>/dev/null)

        if [ -n "$FINAL_STEP" ]; then
            echo "最终训练信息:"
            echo "$FINAL_STEP"
            echo ""
        fi

        # 等待5秒确保所有文件写入完成
        echo "等待5秒确保文件写入..."
        sleep 5

        # 同步 checkpoint 文件
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}同步 Checkpoint 文件${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo ""
        
        bash "${SCRIPT_DIR}/10_sync_checkpoints.sh"
        
        echo ""

        # 开始评估
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}开始模型评估${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo ""

        bash "${SCRIPT_DIR}/08_evaluate_model.sh"

        exit 0
    fi

    # 获取当前训练进度
    CURRENT_STEP=$(ssh $SSH_OPTS ubuntu@$CHIEF_IP \
      "docker logs easyrec_chief 2>&1 | grep 'global step' | tail -1" 2>/dev/null || echo "")

    if [ -n "$CURRENT_STEP" ]; then
        # 提取步数
        STEP_NUM=$(echo "$CURRENT_STEP" | grep -oP 'global step \K[0-9]+' || echo "0")

        if [ "$STEP_NUM" != "$LAST_STEP" ] && [ "$STEP_NUM" != "0" ]; then
            echo "训练进度: step $STEP_NUM / $NUM_STEPS"

            # 计算预计剩余时间（简单估算）
            if [ "$STEP_NUM" -gt 0 ]; then
                PROGRESS_PCT=$((STEP_NUM * 100 / NUM_STEPS))
                echo "完成度: ${PROGRESS_PCT}%"
            fi

            LAST_STEP=$STEP_NUM
        else
            echo "训练运行中..."
        fi
    else
        echo "等待训练开始..."
    fi

    echo ""

    # 等待指定间隔
    sleep $CHECK_INTERVAL
    ELAPSED_TIME=$((ELAPSED_TIME + CHECK_INTERVAL))
done

# 超时
echo -e "${RED}========================================${NC}"
echo -e "${RED}等待超时!${NC}"
echo -e "${RED}========================================${NC}"
echo ""
echo "训练时间超过最大等待时间 ($((MAX_WAIT_TIME / 3600))小时)"
echo "请手动检查训练状态:"
echo "  bash 04_monitor_training.sh"
echo ""
echo "如果训练已完成，可以手动运行评估:"
echo "  bash 10_sync_checkpoints.sh"
echo "  bash 08_evaluate_model.sh"

exit 1
