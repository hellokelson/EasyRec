# 文件和脚本总结

## 📁 文件清单

### 核心脚本（按执行顺序）

| 脚本 | 描述 | 用途 |
|------|------|------|
| **00_pre_check.sh** | 环境检查 | 验证 AWS CLI、SSH key 等前置条件 |
| **01_launch_ec2_instances.sh** | 启动 EC2 实例 | 创建分布式训练集群（1 PS + 1 Chief + 4 Workers） |
| **02_setup_cluster.sh** | 集群配置 | 安装 Docker、下载数据、同步代码 |
| **03_start_training.sh** | 启动训练 | 生成配置并启动分布式训练 |
| **04_monitor_training.sh** | 监控训练 | 实时查看所有节点的训练状态和日志 |
| **05_setup_local_tensorboard.sh** | TensorBoard 配置 | 在 Chief 节点启动 TensorBoard 并配置本地访问 |
| **06_stop_training.sh** | 停止训练 | 停止所有训练容器 |
| **07_terminate_instances.sh** | 终止实例 | 删除所有 EC2 实例 |
| **08_evaluate_model.sh** | 模型评估 | 训练完成后评估模型获取 AUC/GAUC |
| **09_wait_and_evaluate.sh** | 自动评估 | 自动等待训练完成并运行评估 |
| **10_sync_checkpoints.sh** | 同步 Checkpoint | 从 PS Server 复制模型文件到 Chief 节点 |

### 辅助脚本

| 脚本 | 描述 |
|------|------|
| **check_training_status.sh** | 检查训练状态 |
| **start_tensorboard_tunnel.sh** | 启动 SSH 隧道访问 TensorBoard |
| **sync_models.sh** | 下载训练好的模型到本地 |

### 配置文件

| 文件 | 描述 |
|------|------|
| **config.sh** | 主配置文件（AWS 区域、实例类型、数据集大小等） |
| **cluster_info.sh** | 集群信息（自动生成，包含所有节点 IP） |
| **tf_configs.sh** | TensorFlow 分布式配置（PS/Chief/Worker TF_CONFIG） |

### 文档

| 文档 | 描述 |
|------|------|
| **README.md** | 完整使用指南和架构说明 |
| **QUICKSTART.md** | 快速开始指南 |
| **TENSORBOARD_GUIDE.md** | TensorBoard 和评估机制详解 |
| **EVALUATION_GUIDE.md** | 模型评估详细指南 |
| **SUMMARY.md** | 本文档（文件清单和总结） |

## 🔍 重要发现和修复

### 1. PS 模式评估限制

**问题**：TensorBoard 不显示 AUC 等评估指标

**原因**：
- EasyRec PS 模式不支持训练过程中的自动评估
- `throttle_secs` 字段在 `TrainConfig` 中不存在
- 添加该字段会导致配置解析错误

**解决方案**：
- 训练后使用 `08_evaluate_model.sh` 手动评估
- 或使用 `09_wait_and_evaluate.sh` 自动等待并评估

**文档**：`TENSORBOARD_GUIDE.md`

### 2. Checkpoint 文件同步问题

**问题**：评估时找不到模型文件或 AUC 为 0.5（随机）

**原因**：
- PS 模式下，模型权重保存在 PS Server 上
- 评估运行在 Chief 节点，没有访问权限
- 缺少 `.data-*` 和 `.index` 文件

**解决方案**：
- 创建 `10_sync_checkpoints.sh` 脚本
- 从 PS Server 复制到 Chief 节点

**修改文件**：
- 新增：`10_sync_checkpoints.sh`
- 更新：`EVALUATION_GUIDE.md`、`README.md`

### 3. 配置文件分发问题

**问题**：训练卡在 "CreateSession still waiting for response"

**原因**：
- 配置文件只生成在 Chief 节点
- 其他节点（PS/Workers）找不到配置文件

**解决方案**：
- `03_start_training.sh` 已修复
- 配置生成后立即分发到所有节点

**代码**：
```bash
# 在 Chief 生成配置
ssh ubuntu@$CHIEF_IP "cat > config ..."

# 分发到所有节点
for ip in "${ALL_IPS[@]}"; do
    scp ubuntu@$CHIEF_IP:config ubuntu@$ip:config
done
```

### 4. 配置文件错误

**问题**：多次尝试添加自动评估配置失败

**错误配置尝试**：
- ❌ `throttle_secs: 300` → 字段不存在
- ❌ `save_checkpoints_secs: null` → protobuf 不接受 null
- ❌ `accuracy { num_class: 2 }` → num_class 字段无效

**最终配置**：
```protobuf
train_config {
  log_step_count_steps: 100
  save_checkpoints_steps: 1000
  keep_checkpoint_max: 5
  num_steps: 10000
  sync_replicas: true
}

eval_config {
  metrics_set: { auc {} }
  metrics_set: { gauc { uid_field: 'user_id' } }
  # 注意: 仅用于训练后评估，不会在训练中触发
}
```

## 📊 训练结果

最后一次训练完成后的评估结果：

```json
{
  "auc": 0.5279,
  "gauc": 0.5392,
  "global_step": 10000,
  "loss": 0.1683
}
```

**注意**：AUC 较低（~0.53），接近随机（0.5）。可能原因：
1. 训练步数不够（10000步可能太少）
2. 数据集质量或特征工程需要优化
3. 超参数需要调整

## 🚀 推荐工作流

### 标准流程

```bash
# 1. 启动训练
bash 03_start_training.sh

# 2. 监控进度
bash 04_monitor_training.sh

# 3. 查看 TensorBoard（训练 loss）
bash 05_setup_local_tensorboard.sh

# 4. 等待训练完成
bash check_training_status.sh

# 5. 同步 checkpoint（重要！）
bash 10_sync_checkpoints.sh

# 6. 评估模型
bash 08_evaluate_model.sh

# 7. 查看结果
cat eval_result_full.txt

# 8. 清理资源
bash 07_terminate_instances.sh
```

### 自动化流程

```bash
# 1. 启动训练
bash 03_start_training.sh

# 2. 后台自动等待并评估
nohup bash 09_wait_and_evaluate.sh > eval.log 2>&1 &

# 3. 监控进度
tail -f eval.log

# 4. 训练和评估完成后清理
bash 07_terminate_instances.sh
```

## 📝 关键经验教训

1. **PS 模式特性**
   - 不支持训练中评估
   - Checkpoint 文件在 PS Server 上
   - 配置文件必须分发到所有节点

2. **配置文件验证**
   - 不是所有 TensorFlow 配置字段在 EasyRec PS 模式中都可用
   - 使用 protobuf，不接受 null 值
   - 评估配置只在训练后生效

3. **调试技巧**
   - 检查所有节点日志，不只是 Chief
   - 验证配置文件在所有节点存在
   - 确认 checkpoint 文件完整性（`.meta`、`.data-*`、`.index`）

4. **成本控制**
   - 训练完成后立即终止实例
   - 使用 Spot 实例可节省 70% 成本
   - 估算成本：6 实例 × $1.45/小时 × 训练时长

## 🔗 相关链接

- [EasyRec 官方文档](https://easyrec.readthedocs.io/)
- [TensorFlow Parameter Server](https://www.tensorflow.org/tutorials/distribute/parameter_server_training)
- [AWS EC2 实例类型](https://aws.amazon.com/ec2/instance-types/)
