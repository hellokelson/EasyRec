# 大规模数据训练策略

## 当前场景分析

### 我们的数据规模
```
训练数据: 2.5 GB (full dataset)
测试数据: 2.6 GB
总计: 5.1 GB

Worker 内存: 16 GB
页面缓存: 12 GB
```

**结论**: 数据完全可以放入内存 ✅

---

## 真实电商场景的数据规模

### 典型规模对比

| 场景 | 数据量 | 样本数 | 特点 |
|------|--------|--------|------|
| **小型电商** | 10-100 GB | 千万级 | 可以放入内存 |
| **中型电商** | 100 GB - 1 TB | 亿级 | 部分放入内存 |
| **大型电商** | 1-10 TB | 十亿级 | 无法全部放入内存 |
| **超大型电商** | 10+ TB | 百亿级 | 必须流式处理 |

### 阿里巴巴/淘宝级别
```
用户数: 10亿+
商品数: 数十亿
日活跃: 数亿
训练样本: 数百亿到千亿
数据量: 10-100+ TB
```

**显然无法全部放入内存！**

---

## 数据无法放入内存时的解决方案

### 方案1: 流式读取 + TFRecord 格式 ⭐⭐⭐⭐⭐

**最常用的方案**

#### TFRecord 优势
```python
# CSV 格式（当前）
- 文本格式，需要解析
- 每次读取都要 parse
- 占用空间大
- 随机访问慢

# TFRecord 格式（推荐）
- 二进制格式，无需解析
- 预处理好的数据
- 压缩后占用空间小 (30-50% 压缩率)
- 顺序读取快
```

#### 实现方式
```python
# 1. 转换数据为 TFRecord
import tensorflow as tf

def create_tfrecord():
    writer = tf.io.TFRecordWriter('train.tfrecord')
    for row in csv_data:
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'user_id': tf.train.Feature(int64_list=...),
                'item_id': tf.train.Feature(int64_list=...),
                'label': tf.train.Feature(int64_list=...),
            }
        ))
        writer.write(example.SerializeToString())

# 2. 流式读取
dataset = tf.data.TFRecordDataset(['train.tfrecord'])
dataset = dataset.map(parse_function)
dataset = dataset.shuffle(buffer_size=10000)  # 只缓存 10000 条
dataset = dataset.batch(1024)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

**关键配置**:
```python
# 不要一次性加载所有数据
dataset = dataset.cache()  # ❌ 会缓存到内存

# 使用流式处理
dataset = dataset.prefetch(buffer_size=AUTOTUNE)  # ✅ 只预取少量
```

---

### 方案2: 数据分片 (Sharding) ⭐⭐⭐⭐

**将大数据集分成多个小文件**

#### 实现方式
```bash
# 分片数据
train_data/
├── shard_0000.tfrecord  (1 GB)
├── shard_0001.tfrecord  (1 GB)
├── shard_0002.tfrecord  (1 GB)
├── ...
└── shard_0999.tfrecord  (1 GB)

总计: 1000 个分片 = 1 TB
```

```python
# TensorFlow 自动处理分片
file_pattern = 'train_data/shard_*.tfrecord'
dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
dataset = dataset.interleave(
    lambda x: tf.data.TFRecordDataset(x),
    cycle_length=10,  # 同时读取 10 个文件
    num_parallel_calls=tf.data.AUTOTUNE
)
```

**优势**:
- ✅ 并行读取多个文件
- ✅ 每次只加载部分数据
- ✅ 支持分布式训练
- ✅ 易于增量更新

---

### 方案3: 对象存储 + 流式读取 ⭐⭐⭐⭐⭐

**大型电商的标准做法**

#### 架构
```
S3/OSS (对象存储)
    ↓ 网络流式读取
训练节点 (只缓存当前 batch)
    ↓
训练
```

#### 实现方式
```python
# 直接从 S3 读取
import tensorflow as tf

# 方式1: TensorFlow 原生支持
dataset = tf.data.TFRecordDataset('s3://bucket/train/*.tfrecord')

# 方式2: 使用 S3 文件系统
import s3fs
fs = s3fs.S3FileSystem()
files = fs.glob('s3://bucket/train/*.tfrecord')
dataset = tf.data.TFRecordDataset(files)

# 方式3: 使用 Petastorm (Uber 开源)
from petastorm.tf_utils import make_petastorm_dataset
dataset = make_petastorm_dataset('s3://bucket/train/')
```

**优势**:
- ✅ 无限存储容量
- ✅ 无需本地磁盘
- ✅ 多个训练任务共享数据
- ✅ 易于管理和备份

**成本**:
- S3 存储: $0.023/GB/月
- 数据传输: 免费（同区域）
- 比 EBS 便宜 4-5 倍

---

### 方案4: 数据采样 ⭐⭐⭐

**不是所有数据都需要每次训练**

#### 策略
```python
# 1. 随机采样
dataset = full_dataset.take(int(total_size * 0.1))  # 使用 10%

# 2. 重要性采样
# 优先选择困难样本、新样本
dataset = dataset.filter(lambda x: importance_score(x) > threshold)

# 3. 时间窗口
# 只使用最近 N 天的数据
dataset = dataset.filter(lambda x: x['timestamp'] > cutoff_date)
```

**适用场景**:
- 数据量极大（TB 级）
- 数据有时效性
- 需要快速迭代

---

### 方案5: 在线学习 (Online Learning) ⭐⭐⭐⭐

**流式处理，不需要全量数据**

#### 架构
```
实时数据流 (Kafka/Kinesis)
    ↓
Mini-batch (1000 条)
    ↓
增量更新模型
    ↓
部署
```

#### 实现
```python
# 使用 TensorFlow Streaming
import tensorflow as tf

# 从 Kafka 读取
dataset = tf.data.Dataset.from_generator(
    kafka_consumer,
    output_types=(tf.float32, tf.int32)
)

# 小批量训练
for batch in dataset.batch(1000):
    model.train_on_batch(batch)
    if step % 1000 == 0:
        model.save()  # 定期保存
```

**优势**:
- ✅ 模型实时更新
- ✅ 无需存储全量数据
- ✅ 适应数据分布变化

**阿里/淘宝使用场景**:
- 实时推荐
- 广告点击率预测
- 欺诈检测

---

### 方案6: 特征工程优化 ⭐⭐⭐⭐

**减少数据量，而不是处理更多数据**

#### 策略
```python
# 1. 特征选择
# 只保留重要特征，去除冗余特征
selected_features = ['user_id', 'item_id', 'category', 'price']

# 2. 特征降维
# PCA, Embedding 等
user_embedding = reduce_dimension(user_features, dim=128)

# 3. 特征预计算
# 提前计算统计特征，存储为小文件
user_stats = precompute_user_statistics()  # 100 GB → 1 GB
```

---

## 大型电商的实际做法

### 阿里巴巴 / 淘宝

#### 数据处理流程
```
1. 数据收集
   用户行为日志 → Kafka → HDFS/MaxCompute
   
2. 数据预处理
   Spark/Flink → 特征工程 → TFRecord
   
3. 数据存储
   OSS (对象存储) → 分片存储 (10000+ 文件)
   
4. 训练
   PAI-DLC → 流式读取 → 分布式训练
   
5. 增量更新
   每天增量训练 → 合并模型
```

#### 关键技术
- **XDL (X-DeepLearning)**: 阿里自研分布式训练框架
- **MaxCompute**: 大数据处理平台
- **PAI**: 机器学习平台
- **OSS**: 对象存储

### 京东

#### 数据规模
```
用户: 5亿+
商品: 数亿
训练样本: 百亿级
数据量: 10+ TB
```

#### 技术栈
- **TFRecord + Parquet**: 数据格式
- **S3 + HDFS**: 存储
- **Horovod**: 分布式训练
- **特征平台**: 统一特征管理

---

## EasyRec 的大规模数据支持

### 已支持的功能

#### 1. TFRecord 格式
```python
# 配置文件
data_config {
  input_type: TFRecordInput
  input_path: "oss://bucket/train/*.tfrecord"
  batch_size: 1024
  num_epochs: 1
  prefetch_size: 1000
}
```

#### 2. OSS/HDFS 支持
```python
# 直接从 OSS 读取
train_input_path: "oss://bucket/train/*.tfrecord"

# 从 HDFS 读取
train_input_path: "hdfs://namenode:9000/train/*.tfrecord"
```

#### 3. 数据分片
```python
# 自动处理多个文件
input_path: "train_data/part-*.tfrecord"
```

#### 4. 流式处理
```python
data_config {
  prefetch_size: 1000  # 只预取 1000 个 batch
  num_parallel_calls: 16  # 并行读取
  shuffle: true
  shuffle_buffer_size: 10000  # 只缓存 10000 条
}
```

---

## 推荐方案对比

| 数据规模 | 推荐方案 | 理由 |
|---------|---------|------|
| < 10 GB | 当前方案（CSV + 内存） | 简单高效 |
| 10-100 GB | TFRecord + 流式读取 | 平衡性能和复杂度 |
| 100 GB - 1 TB | TFRecord + OSS + 分片 | 标准大规模方案 |
| 1-10 TB | OSS + 数据采样 + 增量训练 | 企业级方案 |
| 10+ TB | 在线学习 + 特征平台 | 超大规模方案 |

---

## 迁移建议

### 从当前方案迁移到 TFRecord

#### 步骤1: 转换数据
```bash
# 使用 EasyRec 工具转换
python -m easy_rec.python.tools.csv_to_tfrecord \
  --input_path ali_ccp_train_full.csv \
  --output_path train.tfrecord \
  --config pipeline.config
```

#### 步骤2: 修改配置
```python
# 原配置
data_config {
  input_type: CSVInput
  input_path: "/workspace/data/ali_ccp_train_full.csv"
  separator: ','
}

# 新配置
data_config {
  input_type: TFRecordInput
  input_path: "/workspace/data/train.tfrecord"
  # 无需 separator
}
```

#### 步骤3: 测试性能
```bash
# 对比训练时间
# CSV: 16 分钟
# TFRecord: 预计 12-14 分钟 (快 15-25%)
```

---

## 总结

### 当前方案（CSV + 内存缓存）

**适用场景**: ✅
- 数据量 < 10 GB
- 单机或小规模分布式
- 快速原型开发

**优势**:
- ✅ 简单易用
- ✅ 无需数据转换
- ✅ 性能已经很好（16分钟）

**局限**:
- ❌ 无法扩展到 TB 级数据
- ❌ 依赖页面缓存（重启后需重新加载）

### 大规模方案（TFRecord + OSS + 流式）

**适用场景**: ✅
- 数据量 > 100 GB
- 大规模分布式训练
- 生产环境

**优势**:
- ✅ 可扩展到 PB 级
- ✅ 无内存限制
- ✅ 支持增量更新
- ✅ 成本更低

**成本**:
- 需要数据转换
- 配置稍复杂
- 需要对象存储

### 建议

**当前阶段**: 保持 CSV 方案 ✅
- 数据量小，性能已优
- 无需过度优化

**未来扩展**: 准备 TFRecord 方案
- 数据量增长到 50+ GB 时
- 需要更快的训练速度时
- 迁移到生产环境时

**参考文档**:
- [EasyRec TFRecord 支持](https://easyrec.readthedocs.io/en/latest/feature/data.html)
- [TensorFlow Data Pipeline](https://www.tensorflow.org/guide/data)
