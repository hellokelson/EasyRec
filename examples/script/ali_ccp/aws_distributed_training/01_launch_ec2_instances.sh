#!/bin/bash

##############################################################################
# 启动 EC2 实例用于分布式训练
##############################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}启动 EC2 分布式训练集群${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查 AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}错误: 未安装 AWS CLI${NC}"
    exit 1
fi

# 用户数据脚本 - 初始化实例
cat > /tmp/user_data.sh << 'EOF'
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
mkdir -p /home/ubuntu/EasyRec
chown -R ubuntu:ubuntu /home/ubuntu/EasyRec

# 拉取 Docker 镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py38-tf2.12-0.8.5

echo "Instance initialized successfully" > /tmp/init_complete
EOF

echo -e "${YELLOW}[1/6] 启动 PS Server (r6i.2xlarge)...${NC}"
PS_INSTANCE_ID=$(aws ec2 run-instances \
    --region $AWS_REGION \
    --image-id $AMI_ID \
    --instance-type $PS_INSTANCE_TYPE \
    --subnet-id ${SUBNET_IDS[0]} \
    --security-group-ids $SECURITY_GROUP \
    --key-name $KEY_NAME \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3,DeleteOnTermination=true}' \
    --user-data file:///tmp/user_data.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${TAG_PREFIX}-ps-0},{Key=Role,Value=ps},{Key=Index,Value=0}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${GREEN}✓ PS Server 已启动: $PS_INSTANCE_ID${NC}"

echo -e "${YELLOW}[2/6] 启动 Chief Worker (m6i.xlarge)...${NC}"
CHIEF_INSTANCE_ID=$(aws ec2 run-instances \
    --region $AWS_REGION \
    --image-id $AMI_ID \
    --instance-type $CHIEF_INSTANCE_TYPE \
    --subnet-id ${SUBNET_IDS[1]} \
    --security-group-ids $SECURITY_GROUP \
    --key-name $KEY_NAME \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3,DeleteOnTermination=true}' \
    --user-data file:///tmp/user_data.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${TAG_PREFIX}-chief},{Key=Role,Value=chief},{Key=Index,Value=0}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${GREEN}✓ Chief Worker 已启动: $CHIEF_INSTANCE_ID${NC}"

# 启动 Workers
WORKER_INSTANCE_IDS=()
for i in $(seq 0 $((NUM_WORKERS-1))); do
    echo -e "${YELLOW}[$(($i+3))/6] 启动 Worker-$i (m6i.xlarge)...${NC}"

    SUBNET_INDEX=$((i % ${#SUBNET_IDS[@]}))

    WORKER_ID=$(aws ec2 run-instances \
        --region $AWS_REGION \
        --image-id $AMI_ID \
        --instance-type $WORKER_INSTANCE_TYPE \
        --subnet-id ${SUBNET_IDS[$SUBNET_INDEX]} \
        --security-group-ids $SECURITY_GROUP \
        --key-name $KEY_
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3,DeleteOn
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${TAG_PREFIX}-worker-$i},{Key=Role,Value=worker},{Key=Index,Value=$i}]" \
        --query 'Instances[0].InstanceId' \
        --output text)

    WORKER_INSTANCE_IDS+=($WORKER_ID)
    echo -e "${GREEN}✓ Worker-$i 已启动: $WORKER_ID${NC}"
done

echo ""
echo -e "${YELLOW}等待实例启动 (约2分钟)...${NC}"
ALL_INSTANCE_IDS="$PS_INSTANCE_ID $CHIEF_INSTANCE_ID ${WORKER_INSTANCE_IDS[@]}"

aws ec2 wait instance-running \
    --region $AWS_REGION \
    --instance-ids $ALL_INSTANCE_IDS

echo -e "${GREEN}✓ 所有实例已启动${NC}"
echo ""

# 获取私有IP地址
echo -e "${YELLOW}获取实例 IP 地址...${NC}"

PS_IP=$(aws ec2 describe-instances \
    --region $AWS_REGION \
    --instance-ids $PS_INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PrivateIpAddress' \
    --output text)

CHIEF_IP=$(aws ec2 describe-instances \
    --region $AWS_REGION \
    --instance-ids $CHIEF_INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PrivateIpAddress' \
    --output text)

WORKER_IPS=()
for worker_id in "${WORKER_INSTANCE_IDS[@]}"; do
    WORKER_IP=$(aws ec2 describe-instances \
        --region $AWS_REGION \
        --instance-ids $worker_id \
        --query 'Reservations[0].Instances[0].PrivateIpAddress' \
        --output text)
    WORKER_IPS+=($WORKER_IP)
done

# 保存集群信息
cat > "$SCRIPT_DIR/cluster_info.sh" << EOF
#!/bin/bash
# 自动生成的集群信息

export PS_INSTANCE_ID="$PS_INSTANCE_ID"
export PS_IP="$PS_IP"

export CHIEF_INSTANCE_ID="$CHIEF_INSTANCE_ID"
export CHIEF_IP="$CHIEF_IP"

export WORKER_INSTANCE_IDS=(${WORKER_INSTANCE_IDS[@]})
export WORKER_IPS=(${WORKER_IPS[@]})

export ALL_INSTANCE_IDS="$ALL_INSTANCE_IDS"
EOF

chmod +x "$SCRIPT_DIR/cluster_info.sh"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}EC2 集群启动成功!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}集群信息:${NC}"
echo "  PS Server:    $PS_INSTANCE_ID -> $PS_IP"
echo "  Chief Worker: $CHIEF_INSTANCE_ID -> $CHIEF_IP"
for i in "${!WORKER_INSTANCE_IDS[@]}"; do
    echo "  Worker-$i:     ${WORKER_INSTANCE_IDS[$i]} -> ${WORKER_IPS[$i]}"
done
echo ""
echo -e "${YELLOW}注意: 等待约5分钟让实例完成初始化（Docker安装）${NC}"
echo -e "${YELLOW}下一步: bash 02_setup_cluster.sh${NC}"
