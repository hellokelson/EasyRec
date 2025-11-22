#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config_single.sh"
source "$SCRIPT_DIR/cluster_info_single.sh"
source "$SCRIPT_DIR/tf_configs_single.sh"

SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

RESULTS_FILE="$SCRIPT_DIR/eval_results/single_node_results.txt"
TEMPLATE_CONFIG="${HOME}/tensorboard_logs_single/deepfm_ali_ccp_full_single_20251120_105747/pipeline.config"

echo "========================================" | tee -a $RESULTS_FILE
echo "Starting 4 training cycles" | tee -a $RESULTS_FILE
echo "Started: $(date)" | tee -a $RESULTS_FILE
echo "========================================" | tee -a $RESULTS_FILE

for i in {1..4}; do
    echo "" | tee -a $RESULTS_FILE
    echo "========================================" | tee -a $RESULTS_FILE
    echo "Cycle $i/4 - $(date)" | tee -a $RESULTS_FILE
    echo "========================================" | tee -a $RESULTS_FILE
    
    # Stop training containers
    ssh $SSH_OPTS ubuntu@$INSTANCE_IP "docker stop \$(docker ps -q --filter 'name=easyrec') 2>/dev/null && docker rm \$(docker ps -aq --filter 'name=easyrec') 2>/dev/null" || true
    sleep 5
    
    # Create experiment name
    EXPERIMENT_NAME="deepfm_ali_ccp_${DATASET_SIZE}_single_$(date +%Y%m%d_%H%M%S)"
    
    # Create config from working template
    cp "$TEMPLATE_CONFIG" /tmp/temp_config.config
    sed -i "s|/workspace/ckpt/deepfm_ali_ccp_full_single_20251120_105747|/workspace/ckpt/${EXPERIMENT_NAME}|" /tmp/temp_config.config
    
    scp $SSH_OPTS /tmp/temp_config.config ubuntu@$INSTANCE_IP:/home/ubuntu/easyrec_data/configs/${EXPERIMENT_NAME}.config > /dev/null 2>&1
    
    # Start training
    ssh $SSH_OPTS ubuntu@$INSTANCE_IP "docker run -d --name easyrec_ps --network host -v /home/ubuntu/easyrec_data:/workspace -e TF_CONFIG='$PS_TF_CONFIG' $DOCKER_IMAGE python -m easy_rec.python.train_eval --pipeline_config_path /workspace/configs/${EXPERIMENT_NAME}.config" > /dev/null 2>&1
    ssh $SSH_OPTS ubuntu@$INSTANCE_IP "docker run -d --name easyrec_chief --network host -v /home/ubuntu/easyrec_data:/workspace -e TF_CONFIG='$CHIEF_TF_CONFIG' $DOCKER_IMAGE python -m easy_rec.python.train_eval --pipeline_config_path /workspace/configs/${EXPERIMENT_NAME}.config" > /dev/null 2>&1
    
    for j in {0..3}; do
        WORKER_VAR="WORKER_${j}_TF_CONFIG"
        ssh $SSH_OPTS ubuntu@$INSTANCE_IP "docker run -d --name easyrec_worker_$j --network host -v /home/ubuntu/easyrec_data:/workspace -e TF_CONFIG='${!WORKER_VAR}' $DOCKER_IMAGE python -m easy_rec.python.train_eval --pipeline_config_path /workspace/configs/${EXPERIMENT_NAME}.config" > /dev/null 2>&1
    done
    
    echo "Training started: $EXPERIMENT_NAME" | tee -a $RESULTS_FILE
    
    # Wait for training to complete
    while true; do
        FINISHED=$(ssh $SSH_OPTS ubuntu@$INSTANCE_IP "docker logs easyrec_chief 2>&1 | grep -c 'Train and evaluate finish' || echo 0" 2>/dev/null)
        FINISHED=${FINISHED:-0}
        
        if [ "$FINISHED" -gt 0 ] 2>/dev/null; then
            echo "Training completed" | tee -a $RESULTS_FILE
            break
        fi
        sleep 30
    done
    
    # Evaluate
    EVAL_OUTPUT=$(ssh $SSH_OPTS ubuntu@$INSTANCE_IP "docker run --rm --network host -v /home/ubuntu/easyrec_data:/workspace $DOCKER_IMAGE python -m easy_rec.python.eval --pipeline_config_path /workspace/configs/${EXPERIMENT_NAME}.config" 2>&1)
    
    AUC=$(echo "$EVAL_OUTPUT" | grep "auc = " | grep -oP 'auc = \K[0-9.]+' | head -1)
    
    echo "Experiment: $EXPERIMENT_NAME" | tee -a $RESULTS_FILE
    echo "AUC: $AUC" | tee -a $RESULTS_FILE
done

echo "" | tee -a $RESULTS_FILE
echo "========================================" | tee -a $RESULTS_FILE
echo "All cycles completed: $(date)" | tee -a $RESULTS_FILE
echo "========================================" | tee -a $RESULTS_FILE
