#!/bin/bash

##############################################################################
# 启动单个 EC2 实例用于单节点分布式训练
##############################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config_single.sh"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}启动单节点训练实例${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查 AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}错误: 未安装 AWS CLI${NC}"
    exit 1
fi

# 用户数据脚本 - 初始化实例
cat > /tmp/user_data_single.sh << 'EOF'
#!/bin/bash
set -e

# 更新系统
apt-get update

# 安装 Docker
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker ubuntu
fi

# 创建工作目录
mkdir -p /home/ubuntu/easyrec_data/{data,configs,ckpt}
chown -R ubuntu:ubuntu /home/ubuntu/easyrec_data

# 拉取 Docker 镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py38-tf2.12-0.8.5

echo "Instance initialized successfully" > /tmp/init_complete
EOF

echo -e "${YELLOW}[1/1] 启动 m6i.4xlarge 实例...${NC}"
INSTANCE_ID=$(aws ec2 run-instances \
    --region $AWS_REGION \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --subnet-id $SUBNET_ID \
    --security-group-ids $SECURITY_GROUP \
    --key-name $KEY_NAME \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
    --user-data file:///tmp/user_data_single.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${TAG_PREFIX}}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${GREEN}✓ 实例已启动: $INSTANCE_ID${NC}"

echo ""
echo -e "${YELLOW}等待实例运行...${NC}"
aws ec2 wait instance-running --region $AWS_REGION --instance-ids $INSTANCE_ID

INSTANCE_IP=$(aws ec2 describe-instances \
    --region $AWS_REGION \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PrivateIpAddress' \
    --output text)

# 保存实例信息
cat > "$SCRIPT_DIR/cluster_info_single.sh" <<EOF
#!/bin/bash
export INSTANCE_ID="$INSTANCE_ID"
export INSTANCE_IP="$INSTANCE_IP"
EOF

echo -e "${GREEN}✓ 实例运行中: $INSTANCE_IP${NC}"

echo ""
echo -e "${YELLOW}等待初始化完成 (90秒)...${NC}"
sleep 300

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}实例启动完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}实例信息:${NC}"
echo "  实例 ID: $INSTANCE_ID"
echo "  内网 IP: $INSTANCE_IP"
echo "  实例类型: $INSTANCE_TYPE"
echo ""
echo -e "${BLUE}下一步:${NC}"
echo "  bash 02_setup_single_cluster.sh"
