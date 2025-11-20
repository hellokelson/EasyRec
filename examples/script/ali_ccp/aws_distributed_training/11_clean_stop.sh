#!/bin/bash

##############################################################################
# 选择性清理并重启训练
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
echo -e "${BLUE}选择性清理训练实验${NC}"
echo -e "${BLUE}========================================${NC}"

SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

# 获取所有实验列表
echo -e "${YELLOW}正在获取实验列表...${NC}"
EXPERIMENTS=$(ssh $SSH_OPTS ubuntu@$CHIEF_IP 'ls -1 /home/ubuntu/easyrec_data/ckpt/ 2>/dev/null | grep "deepfm_ali_ccp.*_ps_[0-9]" | sort -r' || echo "")

if [ -z "$EXPERIMENTS" ]; then
    echo -e "${GREEN}没有找到任何实验目录${NC}"
    exit 0
fi

echo -e "${BLUE}找到以下实验:${NC}"
echo "$EXPERIMENTS" | nl -w2 -s') '
echo ""
echo -e "${YELLOW}选项:${NC}"
echo "  输入实验编号 (1-$(echo "$EXPERIMENTS" | wc -l)): 删除指定实验"
echo "  输入 'all': 删除所有实验"
echo "  输入 'cancel': 取消操作"
echo ""

read -p "请选择: " choice

case $choice in
    all)
        echo -e "${RED}⚠️  将删除所有实验数据!${NC}"
        echo -e "${YELLOW}实验列表:${NC}"
        echo "$EXPERIMENTS"
        echo ""
        read -p "确认删除所有实验? (yes/no): " confirm
        
        if [ "$confirm" = "yes" ]; then
            echo -e "${YELLOW}[1/4] 停止所有训练容器...${NC}"
            bash "$SCRIPT_DIR/06_stop_training.sh" 2>/dev/null || true
            
            echo -e "${YELLOW}[2/4] 删除所有实验数据...${NC}"
            for exp in $EXPERIMENTS; do
                echo "  删除实验: $exp"
                # 清理主机目录
                ssh $SSH_OPTS ubuntu@$CHIEF_IP "rm -rf /home/ubuntu/EasyRec/examples/ckpt/$exp" 2>/dev/null || true
                ssh $SSH_OPTS ubuntu@$PS_IP "rm -rf /home/ubuntu/EasyRec/examples/ckpt/$exp" 2>/dev/null || true
                # 清理Docker workspace
                ssh $SSH_OPTS ubuntu@$CHIEF_IP "sudo rm -rf /home/ubuntu/easyrec_data/ckpt/$exp" 2>/dev/null || true
                ssh $SSH_OPTS ubuntu@$PS_IP "sudo rm -rf /home/ubuntu/easyrec_data/ckpt/$exp" 2>/dev/null || true
                # 清理配置文件
                ssh $SSH_OPTS ubuntu@$CHIEF_IP "rm -f /home/ubuntu/easyrec_data/configs/$exp.config" 2>/dev/null || true
            done
            
            echo -e "${YELLOW}[3/4] 清理实验记录...${NC}"
            rm -f "$SCRIPT_DIR/current_experiment.sh"
            
            echo -e "${GREEN}✓ 所有实验已删除${NC}"
            
        else
            echo -e "${GREEN}操作已取消${NC}"
            exit 0
        fi
        ;;
    cancel)
        echo -e "${GREEN}操作已取消${NC}"
        exit 0
        ;;
    [0-9]*)
        if [ "$choice" -ge 1 ] && [ "$choice" -le $(echo "$EXPERIMENTS" | wc -l) ]; then
            SELECTED_EXP=$(echo "$EXPERIMENTS" | sed -n "${choice}p")
            echo -e "${YELLOW}选择的实验: $SELECTED_EXP${NC}"
            
            read -p "确认删除实验 '$SELECTED_EXP'? (yes/no): " confirm
            
            if [ "$confirm" = "yes" ]; then
                echo -e "${YELLOW}[1/4] 停止训练容器...${NC}"
                bash "$SCRIPT_DIR/06_stop_training.sh" 2>/dev/null || true
                
                echo -e "${YELLOW}[2/4] 删除实验: $SELECTED_EXP${NC}"
                # 清理主机目录
                ssh $SSH_OPTS ubuntu@$CHIEF_IP "rm -rf /home/ubuntu/EasyRec/examples/ckpt/$SELECTED_EXP" 2>/dev/null || true
                ssh $SSH_OPTS ubuntu@$PS_IP "rm -rf /home/ubuntu/EasyRec/examples/ckpt/$SELECTED_EXP" 2>/dev/null || true
                # 清理Docker workspace
                ssh $SSH_OPTS ubuntu@$CHIEF_IP "sudo rm -rf /home/ubuntu/easyrec_data/ckpt/$SELECTED_EXP" 2>/dev/null || true
                ssh $SSH_OPTS ubuntu@$PS_IP "sudo rm -rf /home/ubuntu/easyrec_data/ckpt/$SELECTED_EXP" 2>/dev/null || true
                # 清理配置文件
                ssh $SSH_OPTS ubuntu@$CHIEF_IP "rm -f /home/ubuntu/easyrec_data/configs/$SELECTED_EXP.config" 2>/dev/null || true
                
                echo -e "${YELLOW}[3/4] 更新实验记录...${NC}"
                # 如果删除的是当前实验，清理记录
                if [ -f "$SCRIPT_DIR/current_experiment.sh" ]; then
                    source "$SCRIPT_DIR/current_experiment.sh"
                    if [ "$CURRENT_EXPERIMENT" = "$SELECTED_EXP" ]; then
                        rm -f "$SCRIPT_DIR/current_experiment.sh"
                    fi
                fi
                
                echo -e "${GREEN}✓ 实验 '$SELECTED_EXP' 已删除${NC}"
                
            else
                echo -e "${GREEN}操作已取消${NC}"
                exit 0
            fi
        else
            echo -e "${RED}无效的实验编号${NC}"
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac
