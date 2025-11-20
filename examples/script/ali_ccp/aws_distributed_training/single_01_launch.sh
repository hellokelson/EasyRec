#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}启动单节点训练实例 (m6i.4xlarge)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

cat > /tmp/user_data.sh << 'EOF'
#!/bin/bash
set -e
apt-get update
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker ubuntu
fi
mkdir -p /home/ubuntu/easyrec_data/{data,configs,ckpt}
chown -R ubuntu:ubuntu /home/ubuntu/easyrec_data
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py38-tf2.12-0.8.5
echo "Instance initialized" > /tmp/init_complete
EOF

echo -e "${YELLOW}启动 m6i.4xlarge 实例...${NC}"

# Use ap-northeast-1d subnet
SUBNET_ID="subnet-019a7b29090a4a32d"

INSTANCE_ID=$(aws ec2 run-instances \
    --region $AWS_REGION \
    --image-id $AMI_ID \
    --instance-type m6i.4xlarge \
    --subnet-id $SUBNET_ID \
    --security-group-ids $SECURITY_GROUP \
    --key-name $KEY_NAME \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
    --user-data file:///tmp/user_data.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${TAG_PREFIX}-single}]" \
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

echo "export SINGLE_INSTANCE_ID='$INSTANCE_ID'" > single_instance_info.sh
echo "export SINGLE_INSTANCE_IP='$INSTANCE_IP'" >> single_instance_info.sh

echo -e "${GREEN}✓ 实例运行中: $INSTANCE_IP${NC}"
echo ""
echo -e "${YELLOW}等待初始化完成 (60秒)...${NC}"
sleep 60

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}实例启动完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}实例信息:${NC}"
echo "  ID: $INSTANCE_ID"
echo "  IP: $INSTANCE_IP"
echo ""
echo -e "${BLUE}下一步:${NC}"
echo "  bash single_02_setup.sh"
