# 快速启动指南

## 5分钟上手EasyRec分布式训练

### 第一步：验证环境（1分钟）

```bash
cd /home/zhangkap/sourcecode/EasyRec/examples/script/ali_ccp
bash validate.sh
```

如果看到 `✓ 所有检查通过！环境配置正确。`，继续下一步。

### 第二步：准备数据（2分钟）

```bash
# 检查数据是否存在
ls ../../data/ali_ccp/ali_ccp_train_small.csv

# 如果不存在，运行预处理
cd ../../data/ali_ccp
python preprocess.py small
cd -
```

### 第三步：启动训练（1分钟）

```bash
bash start_training.sh small
```

等待容器启动完成（约10秒）。

### 第四步：监控训练（1分钟）

```bash
bash monitor.sh
```

选择 `1) 实时监控` 查看训练进度。

---

## 常用命令

### 查看日志
```bash
# 所有容器
docker-compose logs -f

# Chief（主要训练日志）
docker-compose logs -f chief

# 特定Worker
docker-compose logs -f worker-0
```

### 检查状态
```bash
# 容器状态
docker-compose ps

# 资源使用
docker stats easyrec_ps_0 easyrec_chief easyrec_worker_0 easyrec_worker_1
```

### 停止训练
```bash
docker-compose down
```

### 清理重启
```bash
# 停止容器
docker-compose down

# 清理模型目录
rm -rf model_dir/

# 重新启动
bash start_training.sh small
```

---

## 预期输出

### 启动成功后看到：
```
========================================
分布式训练启动成功!
========================================
```

### 训练日志示例：
```
[INFO] global_step = 100, loss = 0.52, auc = 0.62
[INFO] global_step = 200, loss = 0.48, auc = 0.65
[INFO] global_step = 300, loss = 0.45, auc = 0.68
```

### 容器状态：
```
NAME                 STATUS          PORTS
easyrec_ps_0        Up 2 minutes    0.0.0.0:2222->2222/tcp
easyrec_chief       Up 2 minutes    0.0.0.0:2223->2223/tcp, 0.0.0.0:6006->6006/tcp
easyrec_worker_0    Up 2 minutes    0.0.0.0:2224->2224/tcp
easyrec_worker_1    Up 2 minutes    0.0.0.0:2225->2225/tcp
```

---

## 遇到问题？

### 1. 容器启动失败
```bash
# 检查端口占用
netstat -tuln | grep -E ':(2222|2223|2224|2225)'

# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py38-tf2.12-0.8.5
```

### 2. 数据文件不存在
```bash
cd ../../data/ali_ccp
python preprocess.py small
```

### 3. 内存不足
编辑 `deepfm_on_ali_ccp_ps.config`，减小batch_size：
```protobuf
data_config {
  batch_size: 256  # 从512减小
}
```

### 4. 查看详细错误
```bash
docker-compose logs chief | tail -50
```

---

## 下一步

- **扩展到更大数据集**: `bash start_training.sh medium` 或 `bash start_training.sh full`
- **查看详细文档**: `cat README.md` 或访问 [EasyRec文档](https://easyrec.readthedocs.io/)
- **调优性能**: 参考 README.md 的"性能优化"章节
- **使用TensorBoard**: 访问 http://localhost:6006

---

## 完整训练流程（Small数据集）

```bash
# 1. 进入目录
cd /home/zhangkap/sourcecode/EasyRec/examples/script/ali_ccp

# 2. 验证环境
bash validate.sh

# 3. 启动训练
bash start_training.sh small

# 4. 在新终端监控
bash monitor.sh

# 5. 等待训练完成（约15分钟）

# 6. 查看结果
ls -lh model_dir/

# 7. 停止并清理
docker-compose down
```

完成！🎉
