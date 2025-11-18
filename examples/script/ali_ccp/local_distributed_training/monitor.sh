#!/bin/bash

##############################################################################
# EasyRec 分布式训练监控脚本
#
# 实时监控训练进度、资源使用情况和日志
##############################################################################

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 使用新版或旧版docker compose命令
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# 清屏
clear

# 显示标题
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   EasyRec 分布式训练监控面板${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查容器是否运行
check_containers() {
    echo -e "${YELLOW}[容器状态]${NC}"

    containers=("easyrec_ps_0" "easyrec_chief" "easyrec_worker_0" "easyrec_worker_1")
    all_running=true

    for container in "${containers[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
            status=$(docker inspect --format='{{.State.Status}}' $container 2>/dev/null)
            if [ "$status" == "running" ]; then
                echo -e "  ${GREEN}✓${NC} $container: ${GREEN}Running${NC}"
            else
                echo -e "  ${RED}✗${NC} $container: ${RED}$status${NC}"
                all_running=false
            fi
        else
            echo -e "  ${RED}✗${NC} $container: ${RED}Not Found${NC}"
            all_running=false
        fi
    done

    echo ""

    if [ "$all_running" = false ]; then
        echo -e "${RED}警告: 部分容器未运行${NC}"
        echo -e "${YELLOW}请运行: bash start_training.sh${NC}"
        echo ""
        exit 1
    fi
}

# 显示资源使用情况
show_resources() {
    echo -e "${YELLOW}[资源使用情况]${NC}"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" \
        easyrec_ps_0 easyrec_chief easyrec_worker_0 easyrec_worker_1 2>/dev/null | \
        awk 'NR==1 {print "  "$0} NR>1 {print "  "$0}'
    echo ""
}

# 提取训练指标
extract_metrics() {
    local container=$1
    local logfile="logs/${container}.log"

    # 保存日志到文件
    docker logs $container 2>&1 | tail -100 > "$logfile" 2>/dev/null

    # 提取最新的step和loss
    local step=$(grep -oP "global_step.*?=\s*\K\d+" "$logfile" | tail -1)
    local loss=$(grep -oP "loss\s*=\s*\K[\d.]+" "$logfile" | tail -1)
    local auc=$(grep -oP "auc\s*=\s*\K[\d.]+" "$logfile" | tail -1)

    echo "$step|$loss|$auc"
}

# 显示训练进度
show_training_progress() {
    echo -e "${YELLOW}[训练进度]${NC}"

    # 从chief获取训练进度
    metrics=$(extract_metrics "easyrec_chief")
    IFS='|' read -r step loss auc <<< "$metrics"

    if [ -n "$step" ]; then
        echo -e "  ${CYAN}Global Step:${NC} $step"
    else
        echo -e "  ${CYAN}Global Step:${NC} ${RED}N/A${NC}"
    fi

    if [ -n "$loss" ]; then
        echo -e "  ${CYAN}Loss:${NC} $loss"
    fi

    if [ -n "$auc" ]; then
        echo -e "  ${CYAN}AUC:${NC} $auc"
    fi

    echo ""
}

# 显示最新日志
show_recent_logs() {
    echo -e "${YELLOW}[最新日志 - Chief]${NC}"
    docker logs easyrec_chief 2>&1 | grep -E "(INFO|WARNING|ERROR|global_step|auc)" | tail -5 | \
        sed 's/^/  /'
    echo ""
}

# 显示检查点信息
show_checkpoint_info() {
    echo -e "${YELLOW}[模型检查点]${NC}"

    if [ -d "model_dir" ]; then
        ckpt_count=$(find model_dir -name "*.index" 2>/dev/null | wc -l)
        latest_ckpt=$(find model_dir -name "checkpoint" -exec cat {} \; 2>/dev/null | grep "model_checkpoint_path" | head -1)

        echo -e "  ${CYAN}检查点数量:${NC} $ckpt_count"
        if [ -n "$latest_ckpt" ]; then
            echo -e "  ${CYAN}最新检查点:${NC} ${latest_ckpt#*: }"
        fi

        # 显示模型目录大小
        model_size=$(du -sh model_dir 2>/dev/null | awk '{print $1}')
        echo -e "  ${CYAN}模型目录大小:${NC} $model_size"
    else
        echo -e "  ${RED}模型目录不存在${NC}"
    fi

    echo ""
}

# 主循环
main() {
    # 创建日志目录
    mkdir -p logs

    # 检查容器
    check_containers

    # 显示静态信息
    show_resources
    show_training_progress
    show_checkpoint_info
    show_recent_logs

    # 显示菜单
    echo -e "${BLUE}========================================${NC}"
    echo -e "${YELLOW}[监控选项]${NC}"
    echo "  1) 实时监控 (自动刷新)"
    echo "  2) 查看Chief完整日志"
    echo "  3) 查看Worker-0完整日志"
    echo "  4) 查看Worker-1完整日志"
    echo "  5) 查看PS-0完整日志"
    echo "  6) 查看所有容器日志"
    echo "  7) 显示详细资源使用"
    echo "  8) 导出日志到文件"
    echo "  9) 停止训练"
    echo "  0) 退出"
    echo ""
    read -p "请选择 (0-9): " choice

    case $choice in
        1)
            echo -e "${GREEN}启动实时监控 (Ctrl+C退出)...${NC}"
            sleep 2
            watch -n 5 -c "bash $0 --refresh"
            ;;
        2)
            docker logs -f easyrec_chief
            ;;
        3)
            docker logs -f easyrec_worker_0
            ;;
        4)
            docker logs -f easyrec_worker_1
            ;;
        5)
            docker logs -f easyrec_ps_0
            ;;
        6)
            $DOCKER_COMPOSE logs -f
            ;;
        7)
            watch -n 2 "docker stats easyrec_ps_0 easyrec_chief easyrec_worker_0 easyrec_worker_1"
            ;;
        8)
            echo -e "${YELLOW}导出日志到文件...${NC}"
            timestamp=$(date +%Y%m%d_%H%M%S)
            docker logs easyrec_chief > "logs/chief_${timestamp}.log" 2>&1
            docker logs easyrec_worker_0 > "logs/worker0_${timestamp}.log" 2>&1
            docker logs easyrec_worker_1 > "logs/worker1_${timestamp}.log" 2>&1
            docker logs easyrec_ps_0 > "logs/ps0_${timestamp}.log" 2>&1
            echo -e "${GREEN}✓ 日志已导出到 logs/ 目录${NC}"
            ;;
        9)
            read -p "确认停止训练? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}停止训练...${NC}"
                $DOCKER_COMPOSE down
                echo -e "${GREEN}✓ 训练已停止${NC}"
            fi
            ;;
        0)
            echo -e "${GREEN}退出监控${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}无效选项${NC}"
            ;;
    esac
}

# 如果是刷新模式，只显示信息
if [ "$1" == "--refresh" ]; then
    check_containers
    show_resources
    show_training_progress
    show_checkpoint_info
    show_recent_logs
else
    main
fi
