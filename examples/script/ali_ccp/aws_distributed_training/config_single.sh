#!/bin/bash

##############################################################################
# 单节点训练配置文件
##############################################################################

# AWS 配置
export AWS_REGION="ap-northeast-1"
export VPC_ID="vpc-07c7749fc361250c1"
export SUBNET_ID="subnet-093897d6c8cdc9908"
export SECURITY_GROUP="sg-009992f079f745216"
export AMI_ID="ami-0244ef75e95122fd9"
export KEY_NAME="zk-global-admin-tokyo"

# 实例配置
export INSTANCE_TYPE="m7i.4xlarge"

# 训练配置
export DATASET_SIZE="full"
export NUM_WORKERS=4

# 实例标识
export TAG_PREFIX="easyrec-single"

# Docker 镜像
export DOCKER_IMAGE="mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py38-tf2.12-0.8.5"

# 训练配置
export NUM_STEPS=10000
export BATCH_SIZE=1024

# 网络端口
export PS_PORT=2222
export CHIEF_PORT=2223
export WORKER_BASE_PORT=2224
