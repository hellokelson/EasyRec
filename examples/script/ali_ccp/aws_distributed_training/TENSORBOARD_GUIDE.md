# TensorBoard 使用指南

## 问题：为什么TensorBoard没有显示AUC指标？

**简短回答**：EasyRec 的 **Parameter Server (PS) 模式不支持训练过程中的自动评估**。TensorBoard 只会显示训练 loss，不会显示 AUC/GAUC 等评估指标。

## 为什么 PS 模式不支持自动评估？

1. **架构限制**：PS 模式是分布式训练架构，Chief 负责协调训练，PS Server 存储模型参数。自动评估需要额外的资源和同步机制，EasyRec 的 PS 模式没有实现这个功能。

2. **配置字段不存在**：我们尝试过添加 `throttle_secs` 等配置字段来触发自动评估，但这些字段在 PS 模式的 `TrainConfig` 中**不存在**，会导致配置解析错误。

3. **设计理念**：PS 模式专注于大规模分布式训练的效率，评估通常在训练完成后进行。

## 解决方案：训练后评估

### 方法1：自动等待并评估（推荐）

启动训练后，使用自动等待脚本：

```bash
# 启动训练
bash 03_start_training.sh

# 后台自动等待训练完成并评估
nohup bash 09_wait_and_evaluate.sh > eval.log 2>&1 &

# 查看日志
tail -f eval.log
```

### 方法2：手动评估

训练完成后手动运行评估：

```bash
# 1. 检查训练是否完成
bash check_training_status.sh

# 2. 同步 checkpoint 文件（重要！）
bash 10_sync_checkpoints.sh

# 3. 运行评估
bash 08_evaluate_model.sh
```

## 重要：Checkpoint 文件同步

在 PS 模式下，模型权重存储在 **PS Server** 上，而评估需要在 **Chief 节点**运行。因此需要先同步 checkpoint 文件：

```bash
# 同步checkpoint文件
bash 10_sync_checkpoints.sh
```

这个脚本会：
1. 从 PS Server 下载 checkpoint 文件（`.data-*`, `.index`）
2. 上传到 Chief 节点
3. 然后才能运行评估

## TensorBoard 显示的内容

### PS 模式下 TensorBoard 显示：

```
Scalars/
├── loss                      # 总损失
├── cross_entropy_loss        # 交叉熵损失
├── regularization_loss       # 正则化损失
└── learning_rate             # 学习率变化
```

**不显示**：AUC, GAUC, Accuracy 等评估指标

### 评估脚本输出：

```json
{
  "auc": 0.7234,
  "gauc": 0.7456,
  "loss": 0.1702,
  "global_step": 10000
}
```

## 配置文件说明

### 当前配置（03_start_training.sh 生成）

```protobuf
train_config {
  log_step_count_steps: 100
  save_checkpoints_steps: 1000
  keep_checkpoint_max: 5
  num_steps: 10000
  sync_replicas: true
  # PS 模式：没有 throttle_secs 或 save_checkpoints_secs 等触发评估的配置
}

eval_config {
  metrics_set: { auc {} }
  metrics_set: { gauc { uid_field: 'user_id' } }
  # 注意: 这个配置用于训练后评估，不会在训练过程中触发
}
```

**为什么不添加 throttle_secs？**

我们在调试过程中发现：
1. `throttle_secs` 字段在 PS 模式的 `TrainConfig` 中**不存在**
2. 添加会导致错误：`Message type "protos.TrainConfig" has no field named "throttle_secs"`
3. `save_checkpoints_secs: null` 也会失败：`Couldn't parse integer: null`

## 完整训练和评估流程

```bash
# 1. 启动训练
bash 03_start_training.sh

# 2. 监控训练进度（可选）
bash 04_monitor_training.sh

# 3. 查看 TensorBoard（只有训练 loss）
bash 05_setup_local_tensorboard.sh
# 访问 http://localhost:6006

# 4. 等待训练完成
bash check_training_status.sh

# 5. 同步 checkpoint 文件
bash 10_sync_checkpoints.sh

# 6. 评估模型获取 AUC
bash 08_evaluate_model.sh

# 7. 查看评估结果
cat eval_result_${DATASET_SIZE}.txt
```

## 评估指标预期

对于 Ali CCP 数据集 + DeepFM 模型：

- **AUC**: 0.70 - 0.78 （良好的 CTR 预测性能）
- **GAUC**: 0.70 - 0.78（分组 AUC）
- **训练 Loss**: 0.15 - 0.20（训练完成时）

## 故障排查

### 问题1：评估时找不到 checkpoint

**错误**：`Could not find trained model in model_dir`

**原因**：Checkpoint 文件在 PS Server 上，Chief 节点没有

**解决**：
```bash
bash 10_sync_checkpoints.sh
```

### 问题2：TensorBoard 显示空白

**原因**：训练还未生成事件文件

**解决**：等待训练开始（通常几分钟后）

### 问题3：评估 AUC 为 0.5 左右

**原因**：模型未正确加载（使用了随机初始化）

**解决**：
1. 确认 checkpoint 文件已同步
2. 检查 `global_step` 是否为 0（应该是 10000）
3. 重新运行 `10_sync_checkpoints.sh`

## 总结

- ✅ **PS 模式只显示训练 loss**，这是正常的
- ✅ **评估必须在训练后进行**，使用 `08_evaluate_model.sh`
- ✅ **Checkpoint 文件需要同步**，使用 `10_sync_checkpoints.sh`
- ❌ **不要尝试添加 throttle_secs**，PS 模式不支持
- ❌ **不要期望 TensorBoard 显示 AUC**，使用评估脚本获取

## 参考

- [09_wait_and_evaluate.sh](./09_wait_and_evaluate.sh) - 自动等待并评估
- [08_evaluate_model.sh](./08_evaluate_model.sh) - 手动评估
- [10_sync_checkpoints.sh](./10_sync_checkpoints.sh) - 同步 checkpoint
- [EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md) - 详细评估指南
