#!/bin/bash

source cluster_info.sh
SSH_KEY="${HOME}/.ssh/zk-global-admin-tokyo.pem"

echo "════════════════════════════════════════"
echo "Checking Training Status..."
echo "════════════════════════════════════════"
echo ""

# Check if training finished
FINISHED=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY ubuntu@$CHIEF_IP \
  "docker logs easyrec_chief 2>&1 | grep -c 'Train and evaluate finish'" 2>/dev/null)

if [ "$FINISHED" -gt 0 ]; then
    echo "✅ Training Status: FINISHED"
    echo ""
    
    # Get final step
    FINAL_STEP=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY ubuntu@$CHIEF_IP \
      "docker logs easyrec_chief 2>&1 | grep 'global step' | tail -1" 2>/dev/null)
    echo "Final Step Info:"
    echo "$FINAL_STEP" | grep -oP 'global step \K[0-9]+'
    
    # Check for DONE marker
    DONE_FILE=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY ubuntu@$CHIEF_IP \
      "ls /home/ubuntu/easyrec_data/ckpt/deepfm_ali_ccp_large_ps/ESTIMATOR_TRAIN_DONE 2>/dev/null")
    
    if [ -n "$DONE_FILE" ]; then
        echo "✅ DONE marker exists"
    fi
else
    echo "⏳ Training Status: RUNNING"
    echo ""
    
    # Get current step
    CURRENT=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY ubuntu@$CHIEF_IP \
      "docker logs easyrec_chief 2>&1 | grep 'global step' | tail -1" 2>/dev/null)
    echo "Current Progress:"
    echo "$CURRENT"
fi

echo ""
echo "════════════════════════════════════════"
