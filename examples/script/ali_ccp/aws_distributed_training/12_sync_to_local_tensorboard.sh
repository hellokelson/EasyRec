#!/bin/bash

##############################################################################
# 同步远程训练结果到本地 TensorBoard
##############################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/cluster_info.sh"

SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

LOCAL_TB_DIR="${HOME}/tensorboard_logs"

echo "=========================================="
echo "同步训练结果到本地 TensorBoard"
echo "=========================================="
echo ""
echo "远程: ubuntu@$CHIEF_IP:/home/ubuntu/easyrec_data/ckpt/"
echo "本地: $LOCAL_TB_DIR"
echo ""

# 创建本地目录
mkdir -p "$LOCAL_TB_DIR"

# 同步数据
rsync -avz --progress --delete \
    -e "ssh $SSH_OPTS" \
    ubuntu@$CHIEF_IP:/home/ubuntu/easyrec_data/ckpt/ \
    "$LOCAL_TB_DIR/"

SYNC_STATUS=$?

echo ""
if [ $SYNC_STATUS -eq 0 ]; then
    echo "✅ 同步完成: $(date)"
    echo ""
    echo "已同步的实验:"
    ls -1 "$LOCAL_TB_DIR" | grep "deepfm_ali_ccp_full_ps_2025" | nl
    echo ""
    echo "总大小: $(du -sh "$LOCAL_TB_DIR" | awk '{print $1}')"
    echo ""
    echo "访问 TensorBoard: http://localhost:6007"
else
    echo "❌ 同步失败，错误码: $SYNC_STATUS"
    exit 1
fi
