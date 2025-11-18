#!/bin/bash

##############################################################################
# AWS EC2 分布式训练配置文件
##############################################################################

# AWS 配置
export AWS_REGION="ap-northeast-1"  # 修改为你的 VPC 所在区域
export VPC_ID="vpc-07c7749fc361250c1"
export SUBNET_IDS=("subnet-0c41047741293baa5" "subnet-093897d6c8cdc9908" "subnet-029886a11d150ddf2")
export SECURITY_GROUP="sg-009992f079f745216"
export AMI_ID="ami-0244ef75e95122fd9"
export KEY_NAME="zk-global-admin-tokyo"  # 修改为你的 SSH key 名称

# 实例配置
export PS_INSTANCE_TYPE="r6i.2xlarge"
export CHIEF_INSTANCE_TYPE="m6i.xlarge"
export WORKER_INSTANCE_TYPE="m6i.xlarge"

# 训练配置
export DATASET_SIZE="full"
export NUM_WORKERS=4

# 实例标识
export TAG_PREFIX="easyrec-training"

# Docker 镜像
export DOCKER_IMAGE="mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py38-tf2.12-0.8.5"

# 模型输出目录
export MODEL_DIR="/home/ubuntu/EasyRec/examples/ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps"

# 训练配置
export NUM_STEPS=10000
export BATCH_SIZE=1024

# 网络端口
export PS_PORT=2222
export CHIEF_PORT=2223
export WORKER_BASE_PORT=2224
