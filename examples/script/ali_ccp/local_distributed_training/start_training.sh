#!/bin/bash

##############################################################################
# EasyRec 分布式训练启动脚本 (PS-Worker模式，Docker版本)
#
# 使用方法:
#   bash start_training.sh [dataset_size]
#
# 参数:
#   dataset_size: small (默认) | medium | large | full
#
# 示例:
#   bash start_training.sh small
##############################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 参数
DATASET_SIZE=${1:-small}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}EasyRec 分布式训练 - PS-Worker模式${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 1. 检查环境
echo -e "${YELLOW}[1/6] 检查环境...${NC}"

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: 未安装Docker${NC}"
    exit 1
fi

# 检查Docker Compose
if ! docker compose version &> /dev/null 2>&1 && ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}错误: 未安装Docker Compose${NC}"
    exit 1
fi

# 使用新版或旧版docker compose命令
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo -e "${GREEN}✓ Docker 环境检查通过${NC}"

# 2. 检查数据文件
echo ""
echo -e "${YELLOW}[2/6] 检查数据文件...${NC}"

DATA_DIR="../../data/ali_ccp"
TRAIN_FILE="${DATA_DIR}/ali_ccp_train_${DATASET_SIZE}.csv"
TEST_FILE="${DATA_DIR}/ali_ccp_test_${DATASET_SIZE}.csv"

if [ ! -f "$TRAIN_FILE" ]; then
    echo -e "${RED}错误: 训练数据文件不存在: $TRAIN_FILE${NC}"
    echo -e "${YELLOW}请先运行数据预处理:${NC}"
    echo "  cd $DATA_DIR"
    echo "  python preprocess.py $DATASET_SIZE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo -e "${RED}错误: 测试数据文件不存在: $TEST_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 数据文件检查通过${NC}"
echo "  训练文件: $TRAIN_FILE"
echo "  测试文件: $TEST_FILE"

# 3. 更新配置文件
echo ""
echo -e "${YELLOW}[3/6] 更新配置文件...${NC}"

CONFIG_FILE="deepfm_on_ali_ccp_ps.config"
BACKUP_FILE="deepfm_on_ali_ccp_ps.config.backup"

# 备份原配置
cp "$CONFIG_FILE" "$BACKUP_FILE"

# 更新数据路径和模型目录
sed -i "s|ali_ccp_train_[a-z]*.csv|ali_ccp_train_${DATASET_SIZE}.csv|g" "$CONFIG_FILE"
sed -i "s|ali_ccp_test_[a-z]*.csv|ali_ccp_test_${DATASET_SIZE}.csv|g" "$CONFIG_FILE"
sed -i "s|model_dir: \"examples/ckpt/deepfm_ali_ccp_[a-z]*_ps\"|model_dir: \"examples/ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps\"|g" "$CONFIG_FILE"

echo -e "${GREEN}✓ 配置文件已更新为 ${DATASET_SIZE} 数据集${NC}"
echo "  模型输出: examples/ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps"

# 4. 清理旧容器和日志
echo ""
echo -e "${YELLOW}[4/6] 清理旧容器和日志...${NC}"

# 停止并移除旧容器
$DOCKER_COMPOSE down 2>/dev/null || true

# 设置模型输出目录
MODEL_DIR="../../ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps"

# 清理模型目录（可选）
if [ -d "$MODEL_DIR" ]; then
    read -p "是否清理旧的模型检查点 ($MODEL_DIR)? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker run --rm -v "$(pwd)/../../":/workspace alpine sh -c "rm -rf /workspace/ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps"
        echo -e "${GREEN}✓ 已清理模型目录${NC}"
    else
        echo -e "${YELLOW}保留旧的模型检查点${NC}"
    fi
else
    echo -e "${GREEN}✓ 模型目录不存在，无需清理${NC}"
fi

# 创建模型输出目录
mkdir -p "$MODEL_DIR"

# 创建日志目录
mkdir -p logs

echo -e "${GREEN}✓ 清理完成${NC}"

# 5. 启动分布式训练
echo ""
echo -e "${YELLOW}[5/6] 启动分布式训练集群...${NC}"
echo ""
echo -e "${BLUE}架构信息:${NC}"
echo "  - 1个 Parameter Server (ps-0): 2 CPUs, 4GB RAM"
echo "  - 1个 Chief Worker (chief): 4 CPUs, 8GB RAM"
echo "  - 2个 Worker (worker-0, worker-1): 各4 CPUs, 8GB RAM"
echo "  - 1个 TensorBoard (tensorboard): 1 CPU, 2GB RAM - 独立可视化服务"
echo ""

$DOCKER_COMPOSE up -d

# 等待容器启动
sleep 5

# 检查容器状态
echo ""
echo -e "${YELLOW}容器状态:${NC}"
$DOCKER_COMPOSE ps

# 6. 显示监控信息
echo ""
echo -e "${YELLOW}[6/6] 训练已启动!${NC}"
echo ""
echo -e "${BLUE}监控命令:${NC}"
echo "  查看所有日志:"
echo "    ${GREEN}docker-compose logs -f${NC}"
echo ""
echo "  查看特定容器日志:"
echo "    ${GREEN}docker compose logs -f chief${NC}"
echo "    ${GREEN}docker compose logs -f worker-0${NC}"
echo "    ${GREEN}docker compose logs -f worker-1${NC}"
echo "    ${GREEN}docker compose logs -f ps-0${NC}"
echo "    ${GREEN}docker compose logs -f tensorboard${NC}"
echo ""
echo "  使用监控脚本:"
echo "    ${GREEN}bash monitor.sh${NC}"
echo ""
echo "  查看TensorBoard (独立服务，训练结束后仍可访问):"
echo "    访问 ${GREEN}http://localhost:6006${NC}"
echo "    查看日志: ${GREEN}docker compose logs -f tensorboard${NC}"
echo ""
echo "  停止所有服务 (包括 TensorBoard):"
echo "    ${GREEN}docker compose down${NC}"
echo ""
echo "  仅停止训练 (保留 TensorBoard):"
echo "    ${GREEN}docker compose stop ps-0 chief worker-0 worker-1${NC}"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}分布式训练启动成功!${NC}"
echo -e "${GREEN}========================================${NC}"
