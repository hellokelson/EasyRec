#!/bin/bash
echo "创建 SSH 隧道到远程 TensorBoard..."
echo "本地访问: http://localhost:6006"
echo "按 Ctrl+C 停止隧道"
echo ""
ssh -i /home/zhangkap/.ssh/zk-global-admin-tokyo.pem -L 6006:localhost:6006 ubuntu@172.16.112.45 -N
