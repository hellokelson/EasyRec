# AWS EC2 分布式训练 - 使用指南

本目录包含在 AWS EC2 集群上运行 EasyRec 分布式训练的完整脚本。

## 架构说明

### 集群配置
- **1个 PS Server**: r6i.2xlarge (64GB RAM, 8 vCPUs)
- **1个 Chief Worker**: m6i.xlarge (16GB RAM, 4 vCPUs)
- **4个 Workers**: m6i.xlarge (16GB RAM, 4 vCPUs)
- **总计**: 6个实例

### 训练配置
- **数据集**: large (300M 训练数据)
- **训练步数**: 10000 steps
- **Batch Size**: 1024/worker
- **全局 Batch Size**: 5120 (1024 × 5 workers)
- **模型**: DeepFM

## 前置准备

### 1. AWS 配置

确保你已配置 AWS CLI：
```bash
aws configure
```

### 2. SSH Key

确保你有 SSH key 用于连接 EC2 实例：
```bash
# 如果没有，创建新的 key pair
aws ec2 create-key-pair \
    --region ap-northeast-1 \
    --key-name zk-global-admin-tokyo \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/zk-global-admin-tokyo.pem

chmod 400 ~/.ssh/zk-global-admin-tokyo.pem
```

### 3. 修改配置

编辑 `config.sh` 文件，更新以下配置：
- `AWS_REGION`: 你的 VPC 所在区域
- `KEY_NAME`: 你的 SSH key 名称
- 其他参数根据需要调整

## 使用步骤

### 步骤 1: 启动 EC2 实例

```bash
bash 01_launch_ec2_instances.sh
```

这个脚本会：
- 启动 6 个 EC2 实例（1 PS + 1 Chief + 4 Workers）
- 自动安装 Docker
- 拉取 EasyRec Docker 镜像
- 保存集群信息到 `cluster_info.sh`

**预计时间**: 3-5 分钟

**输出示例**:
```
PS Server:    i-0xxx -> 172.31.1.10
Chief Worker: i-0yyy -> 172.31.1.11
Worker-0:     i-0zzz -> 172.31.1.12
...
```

⚠️ **等待 5 分钟让实例完成初始化（Docker 安装）**

### 步骤 2: 配置集群

```bash
bash 02_setup_cluster.sh
```

这个脚本会：
- 同步 EasyRec 代码到所有节点
- 分发 large 数据集到所有节点
- 生成 TF_CONFIG 配置
- 创建模型输出目录

**预计时间**: 5-10 分钟（取决于数据大小和网络速度）

### 步骤 3: 启动训练

```bash
bash 03_start_training.sh
```

这个脚本会：
- 生成训练配置文件
- 按顺序启动 PS、Chief、Workers
- 配置分布式 TensorFlow 环境

**预计训练时间**: 2-4 小时（取决于数据集大小）

### 步骤 4: 监控训练

```bash
# 查看实时状态
bash 04_monitor_training.sh

# 查看 Chief 实时日志
ssh -i ~/.ssh/zk-global-admin-tokyo.pem ubuntu@<CHIEF_IP> 'docker logs -f easyrec_chief'
```

### 步骤 5: 配置本地 TensorBoard

```bash
bash 05_setup_local_tensorboard.sh
```

这个脚本会：
- 创建本地模型目录
- 配置 SSH 连接
- 创建模型同步脚本
- 启动本地 TensorBoard Docker 容器

**访问**: http://localhost:6006

**同步远程模型**:
```bash
# 手动同步
bash sync_models.sh

# 自动同步（每5分钟）
watch -n 300 bash sync_models.sh
```

### 步骤 6: 停止训练

```bash
# 停止训练容器（保留实例）
bash 06_stop_training.sh

# 终止所有 EC2 实例
bash 07_terminate_instances.sh
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.sh` | 配置文件（AWS、实例类型、训练参数） |
| `01_launch_ec2_instances.sh` | 启动 EC2 实例 |
| `02_setup_cluster.sh` | 配置集群（同步代码和数据） |
| `03_start_training.sh` | 启动分布式训练 |
| `04_monitor_training.sh` | 监控训练状态 |
| `05_setup_local_tensorboard.sh` | 配置本地 TensorBoard |
| `06_stop_training.sh` | 停止训练 |
| `07_terminate_instances.sh` | 终止 EC2 实例 |
| `cluster_info.sh` | 集群信息（自动生成） |
| `tf_configs.sh` | TF_CONFIG 配置（自动生成） |

## 监控和调试

### 查看训练进度

```bash
# 查看 Chief 日志中的训练指标
ssh -i ~/.ssh/zk-global-admin-tokyo.pem ubuntu@<CHIEF_IP> \
    'docker logs easyrec_chief 2>&1 | grep -E "(global step|loss|auc)"'
```

### 检查容器状态

```bash
# 所有节点
bash 04_monitor_training.sh

# 单个节点
ssh -i ~/.ssh/zk-global-admin-tokyo.pem ubuntu@<NODE_IP> 'docker ps'
```

### 查看资源使用

```bash
ssh -i ~/.ssh/zk-global-admin-tokyo.pem ubuntu@<NODE_IP> 'docker stats'
```

### 进入容器调试

```bash
ssh -i ~/.ssh/zk-global-admin-tokyo.pem ubuntu@<CHIEF_IP> \
    'docker exec -it easyrec_chief bash'
```

## 常见问题

### Q1: SSH 连接失败

**原因**:
- Security Group 没有开放 SSH (22) 端口
- SSH key 权限不正确

**解决**:
```bash
# 检查 Security Group
aws ec2 describe-security-groups --group-ids sg-009992f079f745216

# 修复 SSH key 权限
chmod 400 ~/.ssh/zk-global-admin-tokyo.pem
```

### Q2: 实例之间无法通信

**原因**: Security Group 没有开放训练端口 (2222-2227)

**解决**:
在 Security Group 中添加入站规则：
- Type: Custom TCP
- Port Range: 2222-2227
- Source: 同一个 Security Group

### Q3: Docker 容器启动失败

**检查方法**:
```bash
ssh -i ~/.ssh/zk-global-admin-tokyo.pem ubuntu@<NODE_IP> 'docker logs <container_name>'
```

### Q4: 训练数据找不到

**原因**: 数据同步失败或路径错误

**解决**:
```bash
# 检查数据文件
ssh -i ~/.ssh/zk-global-admin-tokyo.pem ubuntu@<NODE_IP> \
    'ls -lh /home/ubuntu/EasyRec/examples/data/ali_ccp/'

# 重新同步
bash 02_setup_cluster.sh
```

### Q5: 模型保存失败

**原因**: 模型目录权限或磁盘空间不足

**解决**:
```bash
# 检查磁盘空间
ssh -i ~/.ssh/zk-global-admin-tokyo.pem ubuntu@<CHIEF_IP> 'df -h'

# 检查模型目录
ssh -i ~/.ssh/zk-global-admin-tokyo.pem ubuntu@<CHIEF_IP> \
    'ls -la /home/ubuntu/EasyRec/examples/ckpt/'
```

## 成本估算

基于按需实例价格（以 ap-northeast-1 为例）：

| 实例类型 | 数量 | 价格/小时 | 总计/小时 |
|---------|------|----------|----------|
| r6i.2xlarge | 1 | ~$0.50 | $0.50 |
| m6i.xlarge | 5 | ~$0.19 | $0.95 |
| **总计** | 6 | - | **$1.45/小时** |

**训练 4 小时的估算成本**: ~$6

⚠️ **建议**: 训练完成后立即终止实例以避免不必要的费用

## 性能优化

### 增加 Worker 数量

修改 `config.sh`:
```bash
export NUM_WORKERS=8  # 从 4 增加到 8
```

### 调整 Batch Size

修改 `config.sh`:
```bash
export BATCH_SIZE=2048  # 从 1024 增加到 2048
```

### 使用 Spot 实例

在 `01_launch_ec2_instances.sh` 中添加 `--instance-market-options`:
```bash
--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,MaxPrice=0.30}'
```

## 备份和恢复

### 备份模型

```bash
# 从 Chief 下载模型
rsync -avz -e "ssh -i ~/.ssh/zk-global-admin-tokyo.pem" \
    ubuntu@<CHIEF_IP>:/home/ubuntu/EasyRec/examples/ckpt/ \
    ./local_backup/
```

### 恢复训练

如果训练中断，只需重新运行 `03_start_training.sh`，训练会从最新的 checkpoint 继续。

## 重要提示和最佳实践

### PS 模式评估机制

**关键发现**：EasyRec 的 Parameter Server (PS) 模式**不支持训练过程中的自动评估**。

- ❌ **TensorBoard 不会显示 AUC/GAUC**: 只显示训练 loss
- ❌ **throttle_secs 字段不存在**: 添加会导致配置解析错误
- ✅ **使用post-training评估**: 训练完成后运行 `08_evaluate_model.sh`

详细说明见：[TENSORBOARD_GUIDE.md](./TENSORBOARD_GUIDE.md)

### Checkpoint 文件同步

在 PS 模式下，模型权重保存在 **PS Server** 上，而评估运行在 **Chief 节点**。

**必须操作**：
```bash
# 评估前同步 checkpoint 文件
bash 10_sync_checkpoints.sh
```

如果跳过此步骤，评估会失败或使用未初始化的模型（AUC ~ 0.5）。

### 配置文件分发

配置文件必须分发到**所有节点**（PS + Chief + Workers），否则训练会卡在初始化阶段。

`03_start_training.sh` 已正确实现：
```bash
# 在 Chief 生成配置
ssh ubuntu@$CHIEF_IP "cat > config_file ..."

# 复制到所有节点
for ip in "${ALL_IPS[@]}"; do
    scp ubuntu@$CHIEF_IP:config_file ubuntu@$ip:config_file
done
```

### 完整工作流

```bash
# 1. 启动训练
bash 03_start_training.sh

# 2. 监控进度（等待训练完成）
bash 04_monitor_training.sh

# 3. 同步 checkpoint 文件（重要！）
bash 10_sync_checkpoints.sh

# 4. 评估模型
bash 08_evaluate_model.sh

# 5. 查看结果
cat eval_result_full.txt
```

或使用自动化脚本：
```bash
# 启动训练
bash 03_start_training.sh

# 后台自动等待并评估
nohup bash 09_wait_and_evaluate.sh > eval.log 2>&1 &
```

### 常见问题快速参考

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| TensorBoard 没有 AUC | PS 模式限制 | 使用 `08_evaluate_model.sh` |
| 评估找不到 checkpoint | 文件在 PS Server | 运行 `10_sync_checkpoints.sh` |
| 训练卡在 CreateSession | 配置文件未分发 | 确认所有节点有配置文件 |
| 评估 AUC 为 0.5 | 模型未加载 | 检查 checkpoint 同步 |

## 技术支持

遇到问题请：
1. 查看本文档的常见问题章节
2. 检查容器日志
3. 查阅 [TENSORBOARD_GUIDE.md](./TENSORBOARD_GUIDE.md) 和 [EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md)
4. 查阅 EasyRec 官方文档: https://easyrec.readthedocs.io/
