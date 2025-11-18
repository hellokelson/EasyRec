#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/cluster_info.sh"

SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

# Optimized: Use standard EasyRec examples/ckpt directory
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../../../" && pwd)"
LOCAL_MODEL_DIR="${EXAMPLES_DIR}/ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps"

# Create target directory if not exists
mkdir -p "$LOCAL_MODEL_DIR"

echo "════════════════════════════════════════"
echo "Syncing Remote Model to Local"
echo "════════════════════════════════════════"
echo "Remote: ubuntu@$CHIEF_IP:/home/ubuntu/easyrec_data/ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps/"
echo "Local:  $LOCAL_MODEL_DIR"
echo ""

# Optimized rsync with progress and compression
rsync -avz --progress --delete \
    -e "ssh $SSH_OPTS" \
    ubuntu@$CHIEF_IP:/home/ubuntu/easyrec_data/ckpt/deepfm_ali_ccp_${DATASET_SIZE}_ps/ \
    $LOCAL_MODEL_DIR/

SYNC_STATUS=$?

echo ""
if [ $SYNC_STATUS -eq 0 ]; then
    echo "✅ 同步完成: $(date)"
    echo ""
    echo "Model Location: $LOCAL_MODEL_DIR"
    echo "Files synced:"
    ls -lh "$LOCAL_MODEL_DIR" | tail -10
else
    echo "❌ 同步失败，错误码: $SYNC_STATUS"
    exit 1
fi
