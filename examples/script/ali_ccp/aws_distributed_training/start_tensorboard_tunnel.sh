#!/bin/bash

# SSH Tunnel to Remote TensorBoard
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/cluster_info.sh"

SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"

echo "============================================"
echo "Starting SSH Tunnel to TensorBoard"
echo "============================================"
echo ""
echo "Remote TensorBoard: $CHIEF_IP:6006"
echo "Local Access: http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop the tunnel"
echo ""

# Create SSH tunnel
ssh -i "$SSH_KEY" \
    -L 6006:localhost:6006 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=60 \
    -N ubuntu@$CHIEF_IP
