#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/single_instance_info.sh"

echo "终止实例: $SINGLE_INSTANCE_ID"
aws ec2 terminate-instances --region ap-northeast-1 --instance-ids $SINGLE_INSTANCE_ID --output text

echo "✓ 实例已终止"
rm -f single_instance_info.sh single_tf_configs.sh single_current_experiment.sh
