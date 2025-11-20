#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/single_instance_info.sh"

SSH_KEY="${HOME}/.ssh/${KEY_NAME}.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"

echo "检查训练状态..."
FINISHED=$(ssh $SSH_OPTS ubuntu@$SINGLE_INSTANCE_IP "docker logs easyrec_chief 2>&1 | grep -c 'Train and evaluate finish' || echo 0")

if [ "$FINISHED" -gt 0 ]; then
  echo "✅ 训练已完成"
else
  echo "⏳ 训练进行中"
  echo ""
  echo "最近日志:"
  ssh $SSH_OPTS ubuntu@$SINGLE_INSTANCE_IP "docker logs easyrec_chief 2>&1 | tail -30"
fi
