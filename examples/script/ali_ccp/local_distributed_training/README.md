# EasyRec 分布式训练 - Docker PS-Worker模式

基于EasyRec的Docker分布式训练解决方案，使用Parameter Server (PS) 策略，针对Ali CCP数据集上的DeepFM模型训练。

## 目录

- [架构概览](#架构概览)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [详细说明](#详细说明)
- [监控与调试](#监控与调试)
- [性能优化](#性能优化)
- [常见问题](#常见问题)
- [参考资料](#参考资料)

---

## 架构概览

### PS-Worker 架构

本方案使用TensorFlow的Parameter Server策略实现分布式训练：

```
┌─────────────────────────────────────────────────────────┐
│                   Docker Network                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │                                                   │   │
│  │  ┌─────────────┐        ┌──────────────┐        │   │
│  │  │     PS-0    │◄──────►│    Chief     │        │   │
│  │  │ (2 CPUs/4GB)│        │ (4 CPUs/8GB) │        │   │
│  │  └─────────────┘        └──────────────┘        │   │
│  │         ▲                       ▲                │   │
│  │         │                       │                │   │
│  │         │   ┌───────────────────┘                │   │
│  │         │   │                                    │   │
│  │  ┌──────┴───┴─────┐      ┌─────────────┐        │   │
│  │  │   Worker-0      │      │  Worker-1   │        │   │
│  │  │ (4 CPUs/8GB)    │      │(4 CPUs/8GB) │        │   │
│  │  └─────────────────┘      └─────────────┘        │   │
│  │                                                   │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 组件说明

| 组件 | 角色 | 资源配置 | 端口 | 职责 |
|------|------|----------|------|------|
| **ps-0** | Parameter Server | 2 CPUs, 4GB RAM | 2222 | 存储和同步模型参数 |
| **chief** | Chief Worker | 4 CPUs, 8GB RAM | 2223, 6006 | 训练 + 保存checkpoint + 评估 |
| **worker-0** | Worker | 4 CPUs, 8GB RAM | 2224 | 训练 |
| **worker-1** | Worker | 4 CPUs, 8GB RAM | 2225 | 训练 |

**总资源需求**: 14 CPUs, 28GB RAM

---

## 环境要求

### 必需软件

- **Docker**: >= 20.10
- **Docker Compose**: >= 1.29 (或 Docker Compose V2)
- **操作系统**: Linux / macOS / Windows with WSL2

### 系统资源

- **CPU**: 建议 ≥ 16 核心
- **内存**: 建议 ≥ 32GB
- **磁盘**: ≥ 20GB 可用空间
- **网络**: 内网通信，端口 2222-2225, 6006

### Docker镜像

```bash
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py38-tf2.12-0.8.5
```

---

## 快速开始

### 1. 验证环境

首次运行前，验证所有环境配置：

```bash
cd /home/zhangkap/sourcecode/EasyRec/examples/script/ali_ccp
bash validate.sh
```

验证脚本会检查：
- ✓ Docker环境
- ✓ 数据文件
- ✓ 配置文件
- ✓ 系统资源
- ✓ 网络端口

### 2. 准备数据

如果数据文件不存在，运行预处理：

```bash
cd ../../data/ali_ccp
python preprocess.py small    # 100K样本，快速测试
# 或
python preprocess.py medium   # 1M样本，平衡性能
# 或
python preprocess.py full     # 42M样本，完整数据
```

### 3. 启动训练

```bash
cd /home/zhangkap/sourcecode/EasyRec/examples/script/ali_ccp

# 使用small数据集 (默认)
bash start_training.sh small

# 或使用medium数据集
bash start_training.sh medium

# 或使用full数据集
bash start_training.sh full
```

启动脚本会：
1. 检查环境和数据文件
2. 更新配置文件路径
3. 清理旧容器（可选清理checkpoint）
4. 启动4个Docker容器（1 PS + 1 Chief + 2 Workers）
5. 显示监控命令

### 4. 监控训练

**使用监控脚本（推荐）**:
```bash
bash monitor.sh
```

**手动查看日志**:
```bash
# 查看所有容器日志
docker-compose logs -f

# 查看特定容器
docker-compose logs -f chief
docker-compose logs -f worker-0
```

**查看资源使用**:
```bash
docker stats easyrec_ps_0 easyrec_chief easyrec_worker_0 easyrec_worker_1
```

### 5. 停止训练

```bash
docker-compose down
```

---

## 详细说明

### 文件结构

```
examples/script/ali_ccp/
├── README.md                          # 本文档
├── docker-compose.yml                 # Docker Compose配置
├── deepfm_on_ali_ccp_ps.config       # 训练配置（PS模式）
├── start_training.sh                  # 启动训练脚本
├── monitor.sh                         # 监控脚本
├── validate.sh                        # 环境验证脚本
├── model_dir/                         # 模型检查点目录（训练时创建）
└── logs/                              # 日志目录（训练时创建）
```

### 配置文件说明

#### docker-compose.yml

定义4个服务的Docker容器配置：

**关键配置项**：
- `TF_CONFIG`: TensorFlow分布式配置，定义集群拓扑和任务角色
- `OMP_NUM_THREADS`: OpenMP线程数，CPU训练优化
- `TF_NUM_INTRAOP_THREADS`: TensorFlow算子内并行
- `TF_NUM_INTEROP_THREADS`: TensorFlow算子间并行
- `cpus`: CPU核心数限制
- `mem_limit`: 内存限制
- `networks`: Docker网络配置

**TF_CONFIG 示例**:
```json
{
  "cluster": {
    "ps": ["ps-0:2222"],
    "chief": ["chief:2223"],
    "worker": ["worker-0:2224", "worker-1:2225"]
  },
  "task": {
    "type": "chief",
    "index": 0
  }
}
```

#### deepfm_on_ali_ccp_ps.config

EasyRec训练配置文件：

**关键配置项**：
```protobuf
train_config {
  # 分布式策略
  train_distribute: PSStrategy

  # 同步训练（true）vs 异步训练（false）
  sync_replicas: true

  # 训练步数
  num_steps: 5000

  # 保存checkpoint频率
  save_checkpoints_steps: 500
}

data_config {
  # 每个worker的batch size
  # 全局batch size = 512 * 2 workers = 1024
  batch_size: 512

  # CPU优化：增加并行度
  num_parallel_calls: 8
  prefetch_size: 64
}
```

### 脚本详解

#### start_training.sh

**功能**：
1. 检查Docker环境
2. 验证数据文件
3. 更新配置文件路径
4. 清理旧容器和日志
5. 启动Docker Compose
6. 显示监控命令

**使用示例**：
```bash
bash start_training.sh small    # 100K样本
bash start_training.sh medium   # 1M样本
bash start_training.sh full     # 42M样本
```

#### monitor.sh

**功能**：
- 实时显示容器状态
- 显示资源使用情况（CPU、内存）
- 显示训练进度（Step、Loss、AUC）
- 显示最新日志
- 显示checkpoint信息
- 提供交互式菜单

**交互选项**：
1. 实时监控（自动刷新）
2. 查看Chief完整日志
3-5. 查看Worker/PS日志
6. 查看所有容器日志
7. 显示详细资源使用
8. 导出日志到文件
9. 停止训练
0. 退出

#### validate.sh

**功能**：
- 检查Docker环境
- 检查数据文件
- 检查配置文件
- 检查脚本文件
- 检查网络端口
- 检查系统资源
- 检查Docker Compose配置

**输出**：通过/失败统计，并给出修复建议

---

## 监控与调试

### 查看训练进度

**方法1: 使用监控脚本**
```bash
bash monitor.sh
# 选择 1) 实时监控
```

**方法2: 查看Chief日志**
```bash
docker-compose logs -f chief | grep -E "(global_step|auc|loss)"
```

**方法3: TensorBoard**
```bash
# TensorBoard已在chief容器中自动启动
# 访问: http://localhost:6006
```

### 检查容器状态

```bash
# 查看所有容器
docker-compose ps

# 查看容器资源使用
docker stats easyrec_ps_0 easyrec_chief easyrec_worker_0 easyrec_worker_1

# 进入容器调试
docker exec -it easyrec_chief bash
```

### 日志位置

- **容器日志**: `docker-compose logs [service]`
- **导出日志**: 运行 `bash monitor.sh` → 选择 `8) 导出日志`
- **训练日志**: `model_dir/` 中的event文件

### 常见训练指标

```
[INFO] global_step = 1000
[INFO] loss = 0.45
[INFO] auc = 0.68
[INFO] learning_rate = 0.001
```

---

## 性能优化

### CPU优化

**1. 调整线程数**

编辑 `docker-compose.yml`:
```yaml
environment:
  - OMP_NUM_THREADS=8          # 建议设为物理核心数
  - TF_NUM_INTRAOP_THREADS=8   # 算子内并行
  - TF_NUM_INTEROP_THREADS=8   # 算子间并行
```

**2. 增加Batch Size**

编辑 `deepfm_on_ali_ccp_ps.config`:
```protobuf
data_config {
  batch_size: 1024  # 从512增加到1024
}
```

**3. 增加数据读取并行度**

```protobuf
data_config {
  num_parallel_calls: 16      # 增加并行读取
  prefetch_size: 128          # 增加预取buffer
}
```

### 内存优化

**1. 减少Batch Size**
```protobuf
data_config {
  batch_size: 256  # 如果OOM，减小batch size
}
```

**2. 减少Hash Bucket**
```protobuf
feature_config {
  features {
    hash_bucket_size: 10000  # 从50000减少
  }
}
```

**3. 限制容器内存**
```yaml
services:
  worker-0:
    mem_limit: 4g  # 限制为4GB
```

### 扩展性优化

**1. 增加Worker数量**

编辑 `docker-compose.yml`:
```yaml
# 添加worker-2
worker-2:
  image: mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py38-tf2.12-0.8.5
  container_name: easyrec_worker_2
  hostname: worker-2
  environment:
    - TF_CONFIG={"cluster":{"ps":["ps-0:2222"],"chief":["chief:2223"],"worker":["worker-0:2224","worker-1:2225","worker-2:2226"]},"task":{"type":"worker","index":2}}
  ports:
    - "2226:2226"
```

同时更新其他服务的TF_CONFIG以包含新worker。

**2. 增加PS数量**

对于大模型，可增加PS数量以分担参数存储压力。

### 同步 vs 异步训练

**同步训练** (sync_replicas: true):
- 优点：收敛更稳定，梯度质量高
- 缺点：受最慢worker限制，通信开销大
- 适用：Workers性能均衡，网络良好

**异步训练** (sync_replicas: false):
- 优点：不等待慢worker，吞吐量高
- 缺点：梯度可能陈旧，收敛不稳定
- 适用：Workers性能不均，追求速度

修改配置：
```protobuf
train_config {
  sync_replicas: false  # 改为异步
}
```

---

## 常见问题

### Q1: 容器启动失败

**现象**：
```
ERROR: for easyrec_chief  Cannot start service chief: ...
```

**可能原因**：
1. 端口被占用
2. Docker资源不足
3. 镜像不存在

**解决方法**：
```bash
# 检查端口
netstat -tuln | grep -E ':(2222|2223|2224|2225|6006)'

# 检查Docker资源
docker system df
docker system prune  # 清理

# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py38-tf2.12-0.8.5
```

### Q2: 训练不开始或卡住

**现象**：容器运行但无训练日志

**可能原因**：
1. TF_CONFIG配置错误
2. 网络连接失败
3. 数据文件路径错误

**解决方法**：
```bash
# 检查容器日志
docker-compose logs chief

# 检查网络
docker network inspect ali_ccp_easyrec-network

# 验证数据路径
docker exec easyrec_chief ls -la /workspace/EasyRec/examples/data/ali_ccp/
```

### Q3: Out of Memory (OOM)

**现象**：
```
ResourceExhaustedError: OOM when allocating tensor
```

**解决方法**：
```bash
# 方法1: 减小batch size
# 编辑 deepfm_on_ali_ccp_ps.config
batch_size: 256  # 从512减少

# 方法2: 增加容器内存
# 编辑 docker-compose.yml
mem_limit: 12g  # 从8g增加

# 方法3: 减小模型大小
embedding_dim: 8  # 从16减少
hash_bucket_size: 10000  # 从50000减少
```

### Q4: 训练速度慢

**可能原因**：
1. CPU资源不足
2. 数据读取瓶颈
3. 同步等待时间长

**解决方法**：
```bash
# 1. 增加CPU核心
# 编辑 docker-compose.yml
cpus: 8  # 从4增加

# 2. 增加数据并行度
# 编辑 deepfm_on_ali_ccp_ps.config
num_parallel_calls: 16
prefetch_size: 128

# 3. 改为异步训练
sync_replicas: false
```

### Q5: Checkpoint不保存

**现象**：model_dir目录为空

**可能原因**：
1. 目录权限问题
2. Chief未正常运行
3. 训练步数不足

**解决方法**：
```bash
# 检查目录权限
ls -la model_dir/

# 检查Chief日志
docker-compose logs chief | grep -i "saving checkpoint"

# 确认训练步数
# global_step需要达到save_checkpoints_steps (默认500)
```

### Q6: Worker之间不通信

**现象**：每个Worker独立训练

**可能原因**：
1. TF_CONFIG配置不一致
2. Docker网络问题
3. 端口无法访问

**解决方法**：
```bash
# 验证TF_CONFIG
docker exec easyrec_chief env | grep TF_CONFIG
docker exec easyrec_worker_0 env | grep TF_CONFIG

# 测试网络连通性
docker exec easyrec_worker_0 ping -c 3 ps-0
docker exec easyrec_worker_0 telnet ps-0 2222

# 重建网络
docker-compose down
docker network prune
docker-compose up -d
```

### Q7: 数据预处理失败

**现象**：
```
FileNotFoundError: sample_skeleton_train.csv
```

**解决方法**：
```bash
# 1. 下载原始数据
# 访问: https://tianchi.aliyun.com/dataset/408

# 2. 将文件放到正确位置
cp sample_skeleton_*.csv examples/data/ali_ccp/

# 3. 运行预处理
cd examples/data/ali_ccp
python preprocess.py small
```

---

## 性能基准

### Small数据集 (100K samples)

| 配置 | 训练时间 | 峰值内存 | 最终AUC |
|------|---------|---------|---------|
| 1 PS + 1 Chief + 2 Workers | ~15分钟 | ~6GB | 0.65-0.70 |

### Medium数据集 (1M samples)

| 配置 | 训练时间 | 峰值内存 | 最终AUC |
|------|---------|---------|---------|
| 1 PS + 1 Chief + 2 Workers | ~1小时 | ~10GB | 0.70-0.75 |

### Full数据集 (42M samples)

| 配置 | 训练时间 | 峰值内存 | 最终AUC |
|------|---------|---------|---------|
| 1 PS + 1 Chief + 2 Workers | ~6小时 | ~20GB | 0.75-0.80 |

*注：实际性能取决于硬件配置*

---

## 参考资料

### EasyRec 文档
- [EasyRec官方文档](https://easyrec.readthedocs.io/)
- [分布式训练指南](https://easyrec.readthedocs.io/en/latest/train.html)
- [DeepFM模型](https://easyrec.readthedocs.io/en/latest/models/deepfm.html)

### TensorFlow 文档
- [TensorFlow分布式训练](https://www.tensorflow.org/guide/distributed_training)
- [Parameter Server策略](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/ParameterServerStrategy)
- [TF_CONFIG说明](https://www.tensorflow.org/guide/distributed_training#setting_up_tf_config)

### Docker 文档
- [Docker Compose文档](https://docs.docker.com/compose/)
- [Docker网络](https://docs.docker.com/network/)

### 数据集
- [Tianchi Ali CCP数据集](https://tianchi.aliyun.com/dataset/408)

---

## 联系与反馈

如有问题或建议，请：
1. 查阅本文档的[常见问题](#常见问题)章节
2. 查看EasyRec [GitHub Issues](https://github.com/alibaba/EasyRec/issues)
3. 阅读EasyRec [官方文档](https://easyrec.readthedocs.io/)

---

## 许可证

本项目基于Apache License 2.0开源许可证。

## 更新日志

- **2024-11-17**: 初始版本
  - 实现PS-Worker分布式训练
  - 支持Docker Compose部署
  - 支持CPU训练
  - 添加监控和验证脚本
