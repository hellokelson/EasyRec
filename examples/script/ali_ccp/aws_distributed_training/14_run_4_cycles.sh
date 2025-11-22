#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

RESULTS_FILE="eval_results/4_cycles_distributed_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p eval_results

echo "========================================" | tee $RESULTS_FILE
echo "4 Training Cycles (Distributed) - Started: $(date)" | tee -a $RESULTS_FILE
echo "========================================" | tee -a $RESULTS_FILE

for i in {1..4}; do
    echo "" | tee -a $RESULTS_FILE
    echo "========================================" | tee -a $RESULTS_FILE
    echo "Cycle $i/4 - $(date)" | tee -a $RESULTS_FILE
    echo "========================================" | tee -a $RESULTS_FILE
    
    # Stop training
    bash 06_stop_training.sh > /dev/null 2>&1
    sleep 5
    
    # Start training
    echo "Starting training..." | tee -a $RESULTS_FILE
    bash 03_start_training.sh > /dev/null 2>&1
    
    # Wait for completion
    source config.sh
    source cluster_info.sh
    SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
    SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"
    
    while true; do
        FINISHED=$(ssh $SSH_OPTS ubuntu@$CHIEF_IP "docker logs easyrec_chief 2>&1 | grep -c 'Train and evaluate finish' || echo 0" 2>/dev/null)
        FINISHED=${FINISHED:-0}
        
        if [ "$FINISHED" -gt 0 ] 2>/dev/null; then
            echo "Training completed" | tee -a $RESULTS_FILE
            break
        fi
        sleep 60
    done
    
    # Sync checkpoints from PS to Chief
    echo "Syncing checkpoints..." | tee -a $RESULTS_FILE
    bash 10_sync_checkpoints.sh > /dev/null 2>&1
    
    # Evaluate
    echo "Evaluating..." | tee -a $RESULTS_FILE
    EVAL_OUTPUT=$(bash 08_evaluate_model.sh 2>&1)
    AUC=$(echo "$EVAL_OUTPUT" | grep "auc = " | grep -oP 'auc = \K[0-9.]+' | head -1)
    
    source current_experiment.sh
    echo "Experiment: $CURRENT_EXPERIMENT" | tee -a $RESULTS_FILE
    echo "AUC: $AUC" | tee -a $RESULTS_FILE
done

echo "" | tee -a $RESULTS_FILE
echo "========================================" | tee -a $RESULTS_FILE
echo "All 4 cycles completed: $(date)" | tee -a $RESULTS_FILE
echo "Results: $RESULTS_FILE" | tee -a $RESULTS_FILE
