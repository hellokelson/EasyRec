#!/bin/bash

##############################################################################
# 配置本地 TensorBoard 连接到远程模型目录
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
echo -e "${BLUE}配置本地 TensorBoard${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# SSH 配置
SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

# 本地模型目录 - 使用标准 EasyRec examples/ckpt 目录
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../../../" && pwd)"
LOCAL_MODEL_DIR="${EXAMPLES_DIR}/ckpt"

echo -e "${YELLOW}[1/3] 创建本地模型目录...${NC}"
mkdir -p "$LOCAL_MODEL_DIR"
echo -e "${GREEN}✓ 本地目录: $LOCAL_MODEL_DIR${NC}"

echo ""
echo -e "${YELLOW}[2/3] 配置 SSH 连接...${NC}"

# 创建 SSH 配置
cat > "${SCRIPT_DIR}/ssh_config" << EOF
Host easyrec-chief
    HostName $CHIEF_IP
    User ubuntu
    IdentityFile $SSH_KEY
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
EOF

echo -e "${GREEN}✓ SSH 配置已创建${NC}"

echo ""
echo -e "${YELLOW}[3/3] 启动本地 TensorBoard...${NC}"

# 停止旧的 TensorBoard 容器
docker stop easyrec_tensorboard 2>/dev/null || true
docker rm easyrec_tensorboard 2>/dev/null || true

# 启动新的 TensorBoard 容器，使用 SSHFS 挂载远程目录
echo ""
echo -e "${BLUE}方案 1: 使用 rsync 定期同步 (推荐)${NC}"
echo ""
echo "创建同步脚本..."

cat > "${SCRIPT_DIR}/sync_models.sh" << 'SYNCEOF'
#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/cluster_info.sh"

SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

# Optimized: Use standard EasyRec examples/ckpt directory
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../../../" && pwd)"
LOCAL_MODEL_DIR="${EXAMPLES_DIR}/ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps"

# Create target directory if not exists
mkdir -p "$LOCAL_MODEL_DIR"

echo "════════════════════════════════════════"
echo "Syncing Remote Model to Local"
echo "════════════════════════════════════════"
echo "Remote: ubuntu@$CHIEF_IP:/home/ubuntu/easyrec_data/ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps/"
echo "Local:  $LOCAL_MODEL_DIR"
echo ""

# Optimized rsync with progress and compression
rsync -avz --progress --delete \
    -e "ssh $SSH_OPTS" \
    ubuntu@$CHIEF_IP:/home/ubuntu/easyrec_data/ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps/ \
    $LOCAL_MODEL_DIR/

SYNC_STATUS=$?

echo ""
if [ $SYNC_STATUS -eq 0 ]; then
    echo "✅ 同步完成: $(date)"
    echo ""
    echo "Model Location: $LOCAL_MODEL_DIR"
    echo "Files synced:"
    ls -lh "$LOCAL_MODEL_DIR" | tail -10
else
    echo "❌ 同步失败，错误码: $SYNC_STATUS"
    exit 1
fi
SYNCEOF

chmod +x "${SCRIPT_DIR}/sync_models.sh"

echo -e "${GREEN}✓ 同步脚本已创建: ${SCRIPT_DIR}/sync_models.sh${NC}"

# 执行一次初始同步
echo ""
echo "执行初始同步..."
bash "${SCRIPT_DIR}/sync_models.sh"

# 启动本地 TensorBoard
echo ""
echo "启动 TensorBoard Docker 容器..."

docker run -d \
    --name easyrec_tensorboard \
    -p 6006:6006 \
    -v "${LOCAL_MODEL_DIR}:/logs" \
    --restart unless-stopped \
    tensorflow/tensorflow:2.12.0 \
    tensorboard --logdir=/logs --host=0.0.0.0 --port=6006

echo -e "${GREEN}✓ TensorBoard 已启动${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}本地 TensorBoard 配置完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}访问地址:${NC}"
echo "  ${GREEN}http://localhost:6006${NC}"
echo ""
echo -e "${BLUE}同步远程模型:${NC}"
echo "  手动同步:"
echo "    ${GREEN}bash ${SCRIPT_DIR}/sync_models.sh${NC}"
echo ""
echo "  自动同步 (每5分钟):"
echo "    ${GREEN}watch -n 300 bash ${SCRIPT_DIR}/sync_models.sh${NC}"
echo ""
echo -e "${YELLOW}提示: TensorBoard 会实时显示同步的模型数据${NC}"
echo ""
echo -e "${BLUE}方案 2: 直接通过 SSH 端口转发访问 (备选)${NC}"
echo "  如果你想直接访问远程 TensorBoard:"
echo "  1. 在 Chief 节点启动 TensorBoard:"
echo "     ${GREEN}ssh -i $SSH_KEY ubuntu@$CHIEF_IP 'docker run -d --name tensorboard -p 6006:6006 -v /home/ubuntu/easyrec_data/ckpt:/logs tensorflow/tensorflow:2.12.0 tensorboard --logdir=/logs --host=0.0.0.0'${NC}"
echo "  2. 创建 SSH 隧道:"
echo "     ${GREEN}ssh -i $SSH_KEY -L 6006:localhost:6006 ubuntu@$CHIEF_IP -N${NC}"
echo "  3. 访问 http://localhost:6006"
