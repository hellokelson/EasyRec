#!/bin/bash

##############################################################################
# 启动分布式训练 - 自动时间戳版本
##############################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/cluster_info.sh"
source "$SCRIPT_DIR/tf_configs.sh"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 生成时间戳版本
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="deepfm_ali_ccp_${DATASET_SIZE}_ps_${TIMESTAMP}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}启动分布式训练 - ${EXPERIMENT_NAME}${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# SSH 配置
SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

# 检查是否有正在运行的训练
echo -e "${YELLOW}[0/5] 检查现有训练状态...${NC}"

# 检查 Chief 容器是否在运行
CHIEF_RUNNING=$(ssh $SSH_OPTS ubuntu@$CHIEF_IP 'docker ps --filter "name=easyrec_chief" --format "{{.Names}}"' 2>/dev/null || echo "")

if [ ! -z "$CHIEF_RUNNING" ]; then
    echo -e "${RED}⚠️  检测到正在运行的训练任务!${NC}"
    echo -e "${BLUE}当前运行的容器: $CHIEF_RUNNING${NC}"
    echo ""
    
    # 获取当前实验信息（如果存在）
    if [ -f "$SCRIPT_DIR/current_experiment.sh" ]; then
        source "$SCRIPT_DIR/current_experiment.sh"
        echo -e "${BLUE}当前实验: $CURRENT_EXPERIMENT${NC}"
    fi
    
    echo -e "${YELLOW}选项:${NC}"
    echo "  1) 停止当前训练并启动新实验"
    echo "  2) 取消操作，保持当前训练"
    echo ""
    read -p "请选择 (1/2): " choice
    
    case $choice in
        1)
            echo -e "${YELLOW}正在停止当前训练...${NC}"
            bash "$SCRIPT_DIR/06_stop_training.sh"
            echo -e "${GREEN}✓ 已停止现有训练${NC}"
            ;;
        2)
            echo -e "${GREEN}操作已取消，当前训练继续运行${NC}"
            echo -e "${BLUE}监控命令: bash 04_monitor_training.sh${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}无效选择，操作已取消${NC}"
            exit 1
            ;;
    esac
else
    echo -e "${GREEN}✓ 没有检测到正在运行的训练${NC}"
fi

# 配置文件路径（容器内）- 使用时间戳
CONFIG_FILE="/workspace/configs/${EXPERIMENT_NAME}.config"
HOST_CONFIG_FILE="/home/ubuntu/easyrec_data/configs/${EXPERIMENT_NAME}.config"

echo -e "${YELLOW}[1/5] 生成训练配置文件...${NC}"

# 在 Chief 节点生成配置文件
ssh $SSH_OPTS ubuntu@$CHIEF_IP << EOFCONFIG
cat > $HOST_CONFIG_FILE << 'CONFIGEND'
train_input_path: "/workspace/data/ali_ccp_train_${DATASET_SIZE}.csv"
eval_input_path: "/workspace/data/ali_ccp_test_${DATASET_SIZE}.csv"
model_dir: "/workspace/ckpt/${EXPERIMENT_NAME}"

train_config {
  log_step_count_steps: 100

  optimizer_config: {
    adam_optimizer: {
      learning_rate: {
        exponential_decay_learning_rate {
          initial_learning_rate: 0.001
          decay_steps: 1000
          decay_factor: 0.9
          min_learning_rate: 0.00001
        }
      }
    }
    use_moving_average: false
  }

  sync_replicas: true
  save_checkpoints_steps: 1000
  keep_checkpoint_max: 5
  num_steps: ${NUM_STEPS}
}

eval_config {
  metrics_set: {
    auc {}
  }
  metrics_set: {
    gauc {
      uid_field: 'user_id'
    }
  }
  # 注意: PS 模式下不会自动评估，需要训练完成后运行 08_evaluate_model.sh
  # TensorBoard 只显示训练 loss，不显示 AUC/GAUC 等评估指标
}

data_config {
  separator: ','
  with_header: true
  input_fields {
    input_name:'label'
    input_type: INT32
  }
  input_fields {
    input_name:'user_id'
    input_type: STRING
  }
  input_fields {
    input_name: 'item_id'
    input_type: STRING
  }
  input_fields {
    input_name: 'f_301'
    input_type: STRING
  }
  input_fields {
    input_name: 'f_205'
    input_type: STRING
  }
  input_fields {
    input_name: 'f_206'
    input_type: STRING
  }
  input_fields {
    input_name: 'f_207'
    input_type: STRING
  }
  input_fields {
    input_name: 'f_210'
    input_type: STRING
  }

  label_fields: 'label'
  batch_size: ${BATCH_SIZE}
  num_epochs: 10
  prefetch_size: 128
  num_parallel_calls: 16
  shuffle: true
}

feature_config: {
  features: {
    input_names: 'user_id'
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 100000
  }
  features: {
    input_names: 'item_id'
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 100000
  }
  features: {
    input_names: 'f_301'
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 10000
  }
  features: {
    input_names: 'f_205'
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 100
  }
  features: {
    input_names: 'f_206'
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 100
  }
  features: {
    input_names: 'f_207'
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 100
  }
  features: {
    input_names: 'f_210'
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 10000
  }
}

model_config: {
  model_class: 'DeepFM'
  feature_groups: {
    group_name: 'wide'
    feature_names: 'user_id'
    feature_names: 'item_id'
    feature_names: 'f_301'
    feature_names: 'f_205'
    feature_names: 'f_206'
    feature_names: 'f_207'
    feature_names: 'f_210'
    wide_deep: WIDE
  }
  feature_groups: {
    group_name: 'deep'
    feature_names: 'user_id'
    feature_names: 'item_id'
    feature_names: 'f_301'
    feature_names: 'f_205'
    feature_names: 'f_206'
    feature_names: 'f_207'
    feature_names: 'f_210'
    wide_deep: DEEP
  }
  deepfm {
    dnn {
      hidden_units: [256, 128, 64]
      activation: 'relu'
      use_bn: true
    }
    final_dnn {
      hidden_units: [64, 32]
      activation: 'relu'
    }
  }
}
CONFIGEND
EOFCONFIG

# 复制配置到所有节点
ALL_IPS=($PS_IP $CHIEF_IP "${WORKER_IPS[@]}")
for ip in "${ALL_IPS[@]}"; do
    if [ "$ip" != "$CHIEF_IP" ]; then
        echo "  分发配置到: $ip"
        scp $SSH_OPTS ubuntu@$CHIEF_IP:$HOST_CONFIG_FILE ubuntu@$ip:$HOST_CONFIG_FILE
    fi
done

echo -e "${GREEN}✓ 配置文件已生成并分发${NC}"

echo ""
echo -e "${YELLOW}[1.5/5] 创建模型目录并设置权限...${NC}"

# 在所有节点创建模型目录并设置权限
for ip in "${ALL_IPS[@]}"; do
    ssh $SSH_OPTS ubuntu@$ip "sudo mkdir -p /home/ubuntu/easyrec_data/ckpt/${EXPERIMENT_NAME} && sudo chmod 777 /home/ubuntu/easyrec_data/ckpt/${EXPERIMENT_NAME}"
done

echo -e "${GREEN}✓ 模型目录已创建${NC}"

echo ""
echo -e "${YELLOW}[2/5] 启动 PS Server...${NC}"
ssh $SSH_OPTS ubuntu@$PS_IP << EOFPS
cd /home/ubuntu/easyrec_data
docker run -d --name easyrec_ps_0 \
    --network host \
    -v \$(pwd):/workspace \
    -w /workspace \
    -e TF_CONFIG='$PS_TF_CONFIG' \
    -e OMP_NUM_THREADS=16 \
    -e TF_NUM_INTRAOP_THREADS=16 \
    -e TF_NUM_INTEROP_THREADS=16 \
    $DOCKER_IMAGE \
    python -m easy_rec.python.train_eval \
    --pipeline_config_path $CONFIG_FILE
EOFPS
echo -e "${GREEN}✓ PS Server 已启动: $PS_IP${NC}"

echo ""
echo -e "${YELLOW}[3/5] 启动 Chief Worker...${NC}"
ssh $SSH_OPTS ubuntu@$CHIEF_IP << EOFCHIEF
cd /home/ubuntu/easyrec_data
docker run -d --name easyrec_chief \
    --network host \
    -v \$(pwd):/workspace \
    -w /workspace \
    -e TF_CONFIG='$CHIEF_TF_CONFIG' \
    -e OMP_NUM_THREADS=8 \
    -e TF_NUM_INTRAOP_THREADS=8 \
    -e TF_NUM_INTEROP_THREADS=8 \
    $DOCKER_IMAGE \
    python -m easy_rec.python.train_eval \
    --pipeline_config_path $CONFIG_FILE
EOFCHIEF
echo -e "${GREEN}✓ Chief Worker 已启动: $CHIEF_IP${NC}"

echo ""
echo -e "${YELLOW}[4/5] 启动 Workers...${NC}"
for i in "${!WORKER_IPS[@]}"; do
    ip="${WORKER_IPS[$i]}"
    tf_config="${WORKER_TF_CONFIGS[$i]}"

    echo "  启动 Worker-$i: $ip"
    ssh $SSH_OPTS ubuntu@$ip << EOFW
cd /home/ubuntu/easyrec_data
docker run -d --name easyrec_worker_$i \
    --network host \
    -v \$(pwd):/workspace \
    -w /workspace \
    -e TF_CONFIG='$tf_config' \
    -e OMP_NUM_THREADS=8 \
    -e TF_NUM_INTRAOP_THREADS=8 \
    -e TF_NUM_INTEROP_THREADS=8 \
    $DOCKER_IMAGE \
    python -m easy_rec.python.train_eval \
    --pipeline_config_path $CONFIG_FILE
EOFW
    echo -e "    ${GREEN}✓ Worker-$i 已启动${NC}"
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}分布式训练已启动!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}实验信息:${NC}"
echo "  实验名称: $EXPERIMENT_NAME"
echo "  模型目录: /workspace/ckpt/$EXPERIMENT_NAME"
echo "  配置文件: $CONFIG_FILE"
echo ""
echo -e "${BLUE}集群信息:${NC}"
echo "  PS Server:    $PS_IP"
echo "  Chief Worker: $CHIEF_IP"
for i in "${!WORKER_IPS[@]}"; do
    echo "  Worker-$i:     ${WORKER_IPS[$i]}"
done
echo ""
echo -e "${BLUE}监控命令:${NC}"
echo "  查看 Chief 日志:"
echo "    ${GREEN}ssh -i $SSH_KEY ubuntu@$CHIEF_IP 'docker logs -f easyrec_chief'${NC}"
echo ""
echo "  查看所有节点状态:"
echo "    ${GREEN}bash 04_monitor_training.sh${NC}"
echo ""
echo -e "${YELLOW}提示: 每次运行都会创建新的实验版本${NC}"
echo -e "${YELLOW}下一步: bash 05_setup_local_tensorboard.sh (查看所有实验的TensorBoard)${NC}"

# 保存当前实验信息
echo "export CURRENT_EXPERIMENT='$EXPERIMENT_NAME'" > "$SCRIPT_DIR/current_experiment.sh"
echo "export CURRENT_MODEL_DIR='/workspace/ckpt/$EXPERIMENT_NAME'" >> "$SCRIPT_DIR/current_experiment.sh"
