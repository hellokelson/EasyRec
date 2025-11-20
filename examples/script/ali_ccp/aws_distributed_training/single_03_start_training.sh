#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/single_instance_info.sh"
source "$SCRIPT_DIR/single_tf_configs.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

EXPERIMENT_NAME="deepfm_ali_ccp_${DATASET_SIZE}_single_$(date +%Y%m%d_%H%M%S)"
CONFIG_FILE="${EXPERIMENT_NAME}.config"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}启动单节点训练${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}实验名称: $EXPERIMENT_NAME${NC}"
echo ""

cat > /tmp/$CONFIG_FILE <<EOF
train_input_path: "/workspace/data/ali_ccp_train_${DATASET_SIZE}.csv"
eval_input_path: "/workspace/data/ali_ccp_test_${DATASET_SIZE}.csv"
model_dir: "/workspace/ckpt/${EXPERIMENT_NAME}"
train_config { num_steps: $NUM_STEPS save_checkpoints_steps: 1000 log_step_count_steps: 100 }
data_config {
  batch_size: $BATCH_SIZE dataset_type: CsvDataset separator: ","
  input_fields { input_name: "label" input_type: FLOAT default_val: "0" }
  input_fields { input_name: "user_id" input_type: STRING default_val: "" }
  input_fields { input_name: "item_id" input_type: STRING default_val: "" }
  input_fields { input_name: "f_301" input_type: STRING default_val: "" }
  input_fields { input_name: "f_205" input_type: STRING default_val: "" }
  input_fields { input_name: "f_206" input_type: STRING default_val: "" }
  input_fields { input_name: "f_207" input_type: STRING default_val: "" }
  input_fields { input_name: "f_210" input_type: STRING default_val: "" }
  label_fields: "label"
}
feature_configs { input_names: "user_id" feature_type: IdFeature embedding_dim: 16 hash_bucket_size: 100000 }
feature_configs { input_names: "item_id" feature_type: IdFeature embedding_dim: 16 hash_bucket_size: 100000 }
feature_configs { input_names: "f_301" feature_type: IdFeature embedding_dim: 16 hash_bucket_size: 10000 }
feature_configs { input_names: "f_205" feature_type: IdFeature embedding_dim: 16 hash_bucket_size: 100 }
feature_configs { input_names: "f_206" feature_type: IdFeature embedding_dim: 16 hash_bucket_size: 100 }
feature_configs { input_names: "f_207" feature_type: IdFeature embedding_dim: 16 hash_bucket_size: 100 }
feature_configs { input_names: "f_210" feature_type: IdFeature embedding_dim: 16 hash_bucket_size: 100 }
model_config {
  model_class: "DeepFM"
  feature_groups { group_name: "deep" feature_names: "user_id" feature_names: "item_id" feature_names: "f_301" feature_names: "f_205" feature_names: "f_206" feature_names: "f_207" feature_names: "f_210" wide_deep: DEEP }
  deepfm { dnn { hidden_units: [256, 128, 64] } }
  losses { loss_type: CLASSIFICATION weight: 1.0 }
  metrics { auc {} }
}
EOF

scp $SSH_OPTS /tmp/$CONFIG_FILE ubuntu@$SINGLE_INSTANCE_IP:/home/ubuntu/easyrec_data/configs/

echo -e "${YELLOW}启动 PS Server...${NC}"
ssh $SSH_OPTS ubuntu@$SINGLE_INSTANCE_IP "docker run -d --name easyrec_ps --network host -v /home/ubuntu/easyrec_data:/workspace -e TF_CONFIG='$PS_TF_CONFIG' $DOCKER_IMAGE python -m easy_rec.python.train_eval --pipeline_config_path /workspace/configs/$CONFIG_FILE"

echo -e "${YELLOW}启动 Chief Worker...${NC}"
ssh $SSH_OPTS ubuntu@$SINGLE_INSTANCE_IP "docker run -d --name easyrec_chief --network host -v /home/ubuntu/easyrec_data:/workspace -e TF_CONFIG='$CHIEF_TF_CONFIG' $DOCKER_IMAGE python -m easy_rec.python.train_eval --pipeline_config_path /workspace/configs/$CONFIG_FILE"

echo -e "${YELLOW}启动 Worker 0...${NC}"
ssh $SSH_OPTS ubuntu@$SINGLE_INSTANCE_IP "docker run -d --name easyrec_worker_0 --network host -v /home/ubuntu/easyrec_data:/workspace -e TF_CONFIG='$WORKER_0_TF_CONFIG' $DOCKER_IMAGE python -m easy_rec.python.train_eval --pipeline_config_path /workspace/configs/$CONFIG_FILE"

echo -e "${YELLOW}启动 Worker 1...${NC}"
ssh $SSH_OPTS ubuntu@$SINGLE_INSTANCE_IP "docker run -d --name easyrec_worker_1 --network host -v /home/ubuntu/easyrec_data:/workspace -e TF_CONFIG='$WORKER_1_TF_CONFIG' $DOCKER_IMAGE python -m easy_rec.python.train_eval --pipeline_config_path /workspace/configs/$CONFIG_FILE"

echo -e "${YELLOW}启动 Worker 2...${NC}"
ssh $SSH_OPTS ubuntu@$SINGLE_INSTANCE_IP "docker run -d --name easyrec_worker_2 --network host -v /home/ubuntu/easyrec_data:/workspace -e TF_CONFIG='$WORKER_2_TF_CONFIG' $DOCKER_IMAGE python -m easy_rec.python.train_eval --pipeline_config_path /workspace/configs/$CONFIG_FILE"

echo -e "${YELLOW}启动 Worker 3...${NC}"
ssh $SSH_OPTS ubuntu@$SINGLE_INSTANCE_IP "docker run -d --name easyrec_worker_3 --network host -v /home/ubuntu/easyrec_data:/workspace -e TF_CONFIG='$WORKER_3_TF_CONFIG' $DOCKER_IMAGE python -m easy_rec.python.train_eval --pipeline_config_path /workspace/configs/$CONFIG_FILE"

echo "export SINGLE_EXPERIMENT_NAME='$EXPERIMENT_NAME'" > single_current_experiment.sh

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}训练已启动!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}监控命令:${NC}"
echo "  bash single_04_monitor.sh"
echo ""
echo -e "${BLUE}查看日志:${NC}"
echo "  ssh -i $SSH_KEY ubuntu@$SINGLE_INSTANCE_IP 'docker logs -f easyrec_chief'"
