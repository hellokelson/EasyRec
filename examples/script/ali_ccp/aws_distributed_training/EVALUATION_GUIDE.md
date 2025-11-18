# 模型评估指南

## 概述

训练完成后，需要使用测试集评估模型性能，获取AUC、准确率等关键指标。

## 评估方法

### 方法1：自动等待并评估（推荐）

启动后会自动等待训练完成，然后自动评估：

```bash
bash 09_wait_and_evaluate.sh
```

**特点：**
- 自动检测训练完成
- 每60秒检查一次训练状态
- 训练完成后自动运行评估
- 最多等待4小时

### 方法2：手动评估

在训练完成后手动运行评估：

```bash
# 1. 检查训练是否完成
bash check_training_status.sh

# 2. 同步 checkpoint 文件（重要！）
bash 10_sync_checkpoints.sh

# 3. 如果显示 "Training Status: FINISHED"，运行评估
bash 08_evaluate_model.sh
```

**重要提示**：在 PS 模式下，模型权重保存在 PS Server 上，必须先同步到 Chief 节点才能评估！

## 评估流程

评估脚本会执行以下步骤：

1. **检查训练状态** - 确认训练已完成
2. **查找模型文件** - 定位最新的checkpoint
3. **运行评估** - 使用测试集评估模型
4. **输出结果** - 显示AUC、准确率等指标

## 评估结果

评估完成后会生成以下内容：

- **控制台输出**: 实时显示评估指标
- **本地文件**: `eval_result_${DATASET_SIZE}.txt` - 保存所有评估指标
- **远程文件**: 在Chief节点的 `/home/ubuntu/easyrec_data/eval_result_${DATASET_SIZE}.txt`

## 关键评估指标

DeepFM模型通常会输出以下指标：

- **AUC** (Area Under Curve): ROC曲线下面积，衡量模型区分能力
- **GAUC** (Group AUC): 分组AUC，更精确的评估指标
- **Loss**: 测试集上的损失值
- **Accuracy**: 分类准确率
- **Precision/Recall**: 精确率和召回率

## 完整训练+评估流程

```bash
# 1. 启动训练
bash 03_start_training.sh

# 2. 监控训练（可选）
bash 04_monitor_training.sh

# 3. 方式A：后台自动等待并评估
nohup bash 09_wait_and_evaluate.sh > eval.log 2>&1 &

# 或方式B：训练完成后手动评估
bash check_training_status.sh
bash 08_evaluate_model.sh
```

## 评估示例输出

```
========================================
评估结果
========================================

auc: 0.7856
gauc: 0.7923
loss: 0.1567
accuracy: 0.8234

关键指标:
  AUC: 0.7856 - 模型区分能力良好
  GAUC: 0.7923 - 分组评估表现优秀
  准确率: 82.34%
```

## 故障排查

### 问题1: "训练尚未完成，无法评估"

**原因**: 训练还在进行中

**解决**:
```bash
# 检查训练进度
bash check_training_status.sh

# 等待训练完成或使用自动等待脚本
bash 09_wait_and_evaluate.sh
```

### 问题2: "未找到checkpoint文件"

**原因**: 在 PS 模式下，checkpoint 文件保存在 PS Server 上，Chief 节点没有

**解决**:
```bash
# 同步 checkpoint 文件从 PS 到 Chief
bash 10_sync_checkpoints.sh

# 验证文件已复制
ssh -i ~/.ssh/zk-global-admin-tokyo.pem ubuntu@$CHIEF_IP \
  "ls -la /home/ubuntu/easyrec_data/ckpt/deepfm_ali_ccp_*/model.ckpt-*.data-*"
```

### 问题3: 评估结果不理想

**可能原因和改进方向**:

1. **训练步数不够**: 增加 NUM_STEPS
2. **学习率不当**: 调整 train_config 中的学习率
3. **特征工程**: 优化特征组合和embedding维度
4. **数据质量**: 检查数据预处理和特征分布

## 重新评估

如果需要用不同配置重新评估：

```bash
# 编辑配置文件
vim config.sh

# 重新评估
bash 08_evaluate_model.sh
```

## 评估不同checkpoint

默认评估最新checkpoint，如需评估特定checkpoint：

```bash
# 在Chief节点手动运行
ssh -i ~/.ssh/zk-global-admin-tokyo.pem ubuntu@$CHIEF_IP

docker run --rm \
  --network host \
  -v /home/ubuntu/easyrec_data:/workspace \
  ${DOCKER_IMAGE} \
  python3 -m easy_rec.python.eval \
  --pipeline_config_path=/workspace/configs/deepfm_on_ali_ccp_full_ps.config \
  --checkpoint_path=/workspace/ckpt/deepfm_ali_ccp_full_ps/model.ckpt-5000 \
  --eval_result_path=/workspace/eval_result_step5000.txt
```

## 下一步

评估完成后，可以：

1. **导出模型**: 用于生产部署
2. **优化超参数**: 基于评估结果调整
3. **关闭资源**: 终止EC2实例节省成本

```bash
# 终止EC2实例
bash 07_terminate_instances.sh
```
