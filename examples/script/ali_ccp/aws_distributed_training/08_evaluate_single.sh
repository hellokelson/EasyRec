#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config_single.sh"
source "$SCRIPT_DIR/cluster_info_single.sh"
source "$SCRIPT_DIR/current_experiment_single.sh"

SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

echo "Evaluating: $CURRENT_EXPERIMENT"

ssh $SSH_OPTS ubuntu@$INSTANCE_IP "docker run --rm --network host -v /home/ubuntu/easyrec_data:/workspace $DOCKER_IMAGE python -m easy_rec.python.eval --pipeline_config_path /workspace/configs/${CURRENT_EXPERIMENT}.config" 2>&1 | tee /tmp/eval_output.txt

AUC=$(grep -oP '"auc":\s*\K[0-9.]+' /tmp/eval_output.txt | head -1)
echo ""
echo "========================================="
echo "Evaluation Result"
echo "========================================="
echo "Experiment: $CURRENT_EXPERIMENT"
echo "AUC: $AUC"
echo "========================================="

mkdir -p eval_results
echo "$CURRENT_EXPERIMENT: AUC=$AUC" >> eval_results/single_node_results.txt
