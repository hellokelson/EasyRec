#!/bin/bash

##############################################################################
# 在远程 Chief 节点启动 TensorBoard
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
echo -e "${BLUE}在远程启动 TensorBoard${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# SSH 配置
SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

echo -e "${YELLOW}[1/3] 检查 Chief 节点连接...${NC}"
ssh $SSH_OPTS ubuntu@$CHIEF_IP "echo 'Connected'" > /dev/null
echo -e "${GREEN}✓ Chief 节点连接成功: $CHIEF_IP${NC}"

echo ""
echo -e "${YELLOW}[2/3] 在 Chief 节点启动 TensorBoard...${NC}"

# 停止旧的 TensorBoard
ssh $SSH_OPTS ubuntu@$CHIEF_IP "docker stop tensorboard 2>/dev/null || true; docker rm tensorboard 2>/dev/null || true"

# 启动 TensorBoard
ssh $SSH_OPTS ubuntu@$CHIEF_IP << 'REMOTE_CMD'
docker run -d \
    --name tensorboard \
    -p 6006:6006 \
    -v /home/ubuntu/easyrec_data/ckpt:/logs \
    --restart unless-stopped \
    tensorflow/tensorflow:2.12.0 \
    tensorboard --logdir=/logs --host=0.0.0.0 --port=6006

echo "TensorBoard 容器已启动"
docker ps --filter name=tensorboard --format "{{.Names}}: {{.Status}}"
REMOTE_CMD

echo -e "${GREEN}✓ TensorBoard 已在远程启动${NC}"

echo ""
echo -e "${YELLOW}[3/3] 创建 SSH 隧道脚本...${NC}"

# 创建隧道脚本
cat > "${SCRIPT_DIR}/start_tensorboard_tunnel.sh" << EOF
#!/bin/bash
echo "创建 SSH 隧道到远程 TensorBoard..."
echo "本地访问: http://localhost:6006"
echo "按 Ctrl+C 停止隧道"
echo ""
ssh -i $SSH_KEY -L 6006:localhost:6006 ubuntu@$CHIEF_IP -N
EOF

chmod +x "${SCRIPT_DIR}/start_tensorboard_tunnel.sh"

echo -e "${GREEN}✓ 隧道脚本已创建${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}远程 TensorBoard 配置完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}访问方式:${NC}"
echo ""
echo "  方式 1: 通过 SSH 隧道 (推荐)"
echo "    启动隧道:"
echo "      ${GREEN}bash ${SCRIPT_DIR}/start_tensorboard_tunnel.sh${NC}"
echo "    访问地址:"
echo "      ${GREEN}http://localhost:6006${NC}"
echo ""
echo "  方式 2: 直接访问 (需要开放安全组端口 6006)"
echo "    访问地址:"
echo "      ${GREEN}http://$CHIEF_IP:6006${NC}"
echo ""
echo -e "${BLUE}管理命令:${NC}"
echo "  查看 TensorBoard 日志:"
echo "    ${GREEN}ssh -i $SSH_KEY ubuntu@$CHIEF_IP 'docker logs -f tensorboard'${NC}"
echo ""
echo "  停止 TensorBoard:"
echo "    ${GREEN}ssh -i $SSH_KEY ubuntu@$CHIEF_IP 'docker stop tensorboard'${NC}"
echo ""
echo "  重启 TensorBoard:"
echo "    ${GREEN}ssh -i $SSH_KEY ubuntu@$CHIEF_IP 'docker restart tensorboard'${NC}"
