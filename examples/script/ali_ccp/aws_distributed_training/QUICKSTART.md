# 快速开始指南

## 最快速的方式启动训练

### 第 1 步: 修改配置 (2分钟)

编辑 `config.sh`，修改以下必要配置：

```bash
export AWS_REGION="ap-northeast-1"        # 改为你的区域
export KEY_NAME="zk-global-admin-tokyo"    # 改为你的 SSH key 名称
```

### 第 2 步: 环境检查 (1分钟)

```bash
bash 00_pre_check.sh
```

确保所有检查通过后继续。

### 第 3 步: 启动集群 (5分钟)

```bash
# 启动 6 个 EC2 实例
bash 01_launch_ec2_instances.sh

# ⚠️ 等待 5 分钟让实例初始化
```

### 第 4 步: 配置集群 (10分钟)

```bash
# 同步代码和数据
bash 02_setup_cluster.sh
```

### 第 5 步: 启动训练 (1分钟)

```bash
# 启动分布式训练
bash 03_start_training.sh
```

### 第 6 步: 监控训练 (可选)

```bash
# 方式 1: 使用监控脚本
bash 04_monitor_training.sh

# 方式 2: 查看 Chief 日志
ssh -i ~/.ssh/zk-global-admin-tokyo.pem ubuntu@<CHIEF_IP> 'docker logs -f easyrec_chief'
```

### 第 7 步: 本地 TensorBoard (可选)

```bash
# 配置本地 TensorBoard
bash 05_setup_local_tensorboard.sh

# 访问 http://localhost:6006

# 同步远程模型
bash sync_models.sh  # 手动同步
watch -n 300 bash sync_models.sh  # 每5分钟自动同步
```

### 第 8 步: 完成后清理

```bash
# 停止训练（保留实例）
bash 06_stop_training.sh

# 终止所有实例
bash 07_terminate_instances.sh
```

## 脚本说明

| 脚本 | 功能 | 时间 |
|------|------|------|
| `00_pre_check.sh` | 环境检查 | 1分钟 |
| `01_launch_ec2_instances.sh` | 启动 EC2 实例 | 5分钟 |
| `02_setup_cluster.sh` | 配置集群 | 10分钟 |
| `03_start_training.sh` | 启动训练 | 1分钟 |
| `04_monitor_training.sh` | 监控训练 | - |
| `05_setup_local_tensorboard.sh` | 本地 TensorBoard | 2分钟 |
| `06_stop_training.sh` | 停止训练 | 1分钟 |
| `07_terminate_instances.sh` | 终止实例 | 2分钟 |

## 估算成本

**实例配置**:
- 1x r6i.2xlarge (PS): $0.50/小时
- 5x m6i.xlarge (Chief + Workers): $0.95/小时
- **总计**: ~$1.45/小时

**训练时间**: 2-4 小时

**总成本**: ~$3-6

## 常见命令

```bash
# 查看集群信息
source cluster_info.sh
echo "PS: $PS_IP"
echo "Chief: $CHIEF_IP"
echo "Workers: ${WORKER_IPS[@]}"

# 查看训练进度
ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$CHIEF_IP \
    'docker logs easyrec_chief 2>&1 | grep -E "(global step|loss|auc)" | tail -20'

# 检查所有容器状态
bash 04_monitor_training.sh

# 进入 Chief 容器
ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$CHIEF_IP \
    'docker exec -it easyrec_chief bash'

# 下载模型到本地
rsync -avz -e "ssh -i ~/.ssh/${KEY_NAME}.pem" \
    ubuntu@$CHIEF_IP:/home/ubuntu/EasyRec/examples/ckpt/ \
    ./local_backup/
```

## 故障排查

### 实例启动失败
- 检查 AWS 配额
- 验证 VPC/Subnet/SG 配置
- 查看 AWS Console 错误信息

### SSH 连接失败
- 检查 SSH key 权限: `chmod 400 ~/.ssh/${KEY_NAME}.pem`
- 检查 Security Group 是否开放 22 端口

### 训练无法启动
- 检查 Docker 容器状态: `docker ps -a`
- 查看容器日志: `docker logs <container_name>`
- 验证 TF_CONFIG 配置

### 数据同步慢
- 使用更大的实例类型
- 确保在同一可用区
- 检查网络带宽

## 获取帮助

详细文档: [README.md](README.md)
