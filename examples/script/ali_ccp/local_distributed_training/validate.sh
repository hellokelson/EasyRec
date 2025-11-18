#!/bin/bash

##############################################################################
# EasyRec 分布式训练验证脚本
#
# 在启动完整训练前，验证环境配置是否正确
##############################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 统计变量
PASSED=0
FAILED=0

# 检查函数
check_item() {
    local name=$1
    local result=$2

    if [ $result -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} $name"
        ((PASSED++))
    else
        echo -e "  ${RED}✗${NC} $name"
        ((FAILED++))
    fi
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   EasyRec 分布式训练环境验证${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 1. Docker环境检查
echo -e "${YELLOW}[1/7] Docker环境检查${NC}"

# 检查Docker
docker --version &> /dev/null
check_item "Docker已安装" $?

# 检查Docker运行
docker ps &> /dev/null
check_item "Docker服务运行中" $?

# 检查Docker Compose
if docker compose version &> /dev/null 2>&1; then
    check_item "Docker Compose (新版) 已安装" 0
elif docker-compose --version &> /dev/null 2>&1; then
    check_item "Docker Compose (旧版) 已安装" 0
else
    check_item "Docker Compose 已安装" 1
fi

# 检查Docker镜像
docker images | grep -q "easyrec.*py38-tf2.12"
check_item "EasyRec Docker镜像存在" $?

echo ""

# 2. 数据文件检查
echo -e "${YELLOW}[2/7] 数据文件检查${NC}"

DATA_DIR="../../data/ali_ccp"

# 检查原始数据
[ -f "$DATA_DIR/sample_skeleton_train.csv" ]
check_item "原始训练数据存在" $?

[ -f "$DATA_DIR/sample_skeleton_test.csv" ]
check_item "原始测试数据存在" $?

# 检查预处理后的数据
datasets=("small" "medium")
for ds in "${datasets[@]}"; do
    if [ -f "$DATA_DIR/ali_ccp_train_${ds}.csv" ]; then
        check_item "预处理数据 (${ds}) 存在" 0
    else
        check_item "预处理数据 (${ds}) 存在" 1
        echo -e "    ${YELLOW}运行: cd $DATA_DIR && python preprocess.py ${ds}${NC}"
    fi
done

echo ""

# 3. 配置文件检查
echo -e "${YELLOW}[3/7] 配置文件检查${NC}"

[ -f "docker-compose.yml" ]
check_item "docker-compose.yml 存在" $?

[ -f "deepfm_on_ali_ccp_ps.config" ]
check_item "训练配置文件存在" $?

# 检查配置文件语法
if [ -f "deepfm_on_ali_ccp_ps.config" ]; then
    grep -q "train_distribute: PSStrategy" deepfm_on_ali_ccp_ps.config
    check_item "PSStrategy配置正确" $?

    grep -q "sync_replicas:" deepfm_on_ali_ccp_ps.config
    check_item "同步配置存在" $?
fi

echo ""

# 4. 脚本文件检查
echo -e "${YELLOW}[4/7] 脚本文件检查${NC}"

[ -f "start_training.sh" ] && [ -x "start_training.sh" ]
check_item "start_training.sh 可执行" $?

[ -f "monitor.sh" ] && [ -x "monitor.sh" ]
check_item "monitor.sh 可执行" $?

[ -f "validate.sh" ] && [ -x "validate.sh" ]
check_item "validate.sh 可执行" $?

echo ""

# 5. 网络端口检查
echo -e "${YELLOW}[5/7] 网络端口检查${NC}"

ports=(2222 2223 2224 2225 6006)
for port in "${ports[@]}"; do
    if ! netstat -tuln 2>/dev/null | grep -q ":$port " && ! ss -tuln 2>/dev/null | grep -q ":$port "; then
        check_item "端口 $port 可用" 0
    else
        check_item "端口 $port 可用" 1
        echo -e "    ${YELLOW}警告: 端口已被占用，可能导致容器启动失败${NC}"
    fi
done

echo ""

# 6. 系统资源检查
echo -e "${YELLOW}[6/7] 系统资源检查${NC}"

# 检查CPU核心数
cpu_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 0)
if [ $cpu_cores -ge 8 ]; then
    check_item "CPU核心数充足 ($cpu_cores cores)" 0
else
    check_item "CPU核心数充足 ($cpu_cores cores, 建议≥8)" 1
fi

# 检查内存
total_mem=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo 0)
if [ $total_mem -ge 16 ]; then
    check_item "内存充足 (${total_mem}GB)" 0
else
    check_item "内存充足 (${total_mem}GB, 建议≥16GB)" 1
fi

# 检查磁盘空间
disk_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ $disk_space -ge 20 ]; then
    check_item "磁盘空间充足 (${disk_space}GB可用)" 0
else
    check_item "磁盘空间充足 (${disk_space}GB可用, 建议≥20GB)" 1
fi

echo ""

# 7. Docker配置检查
echo -e "${YELLOW}[7/7] Docker Compose配置检查${NC}"

# 检查docker-compose.yml语法
if command -v docker-compose &> /dev/null; then
    docker-compose config &> /dev/null
    check_item "Docker Compose配置语法正确" $?
elif docker compose version &> /dev/null; then
    docker compose config &> /dev/null
    check_item "Docker Compose配置语法正确" $?
fi

# 检查网络配置
grep -q "easyrec-network" docker-compose.yml
check_item "Docker网络配置存在" $?

# 检查服务定义
for service in ps-0 chief worker-0 worker-1; do
    grep -q "$service:" docker-compose.yml
    check_item "服务 $service 已定义" $?
done

echo ""

# 显示总结
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}验证结果总结${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "  ${GREEN}通过: $PASSED${NC}"
echo -e "  ${RED}失败: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ 所有检查通过！环境配置正确。${NC}"
    echo ""
    echo -e "${YELLOW}下一步操作:${NC}"
    echo "  1. 启动训练: ${GREEN}bash start_training.sh small${NC}"
    echo "  2. 监控训练: ${GREEN}bash monitor.sh${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}✗ 发现 $FAILED 个问题，请修复后再启动训练。${NC}"
    echo ""
    echo -e "${YELLOW}常见问题解决:${NC}"
    echo "  1. 数据文件缺失:"
    echo "     cd ../../data/ali_ccp && python preprocess.py small"
    echo ""
    echo "  2. Docker镜像缺失:"
    echo "     docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py38-tf2.12-0.8.5"
    echo ""
    echo "  3. 端口被占用:"
    echo "     netstat -tuln | grep -E ':(2222|2223|2224|2225|6006)'"
    echo "     找到占用进程并停止"
    echo ""
    exit 1
fi
