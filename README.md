# APS RPC Demo - Advanced Production Scheduling with RPC

## 项目简介

这是一个基于RPC（Remote Procedure Call）的高级生产调度系统演示项目。项目实现了多种调度算法，包括遗传算法（GA）、启发式规则等，用于解决柔性作业车间调度问题（FJSP）。

## 项目结构

```
APS_RPC_demo/
├── Algorithms/           # 调度算法实现
│   ├── DispatchingRules/ # 调度规则算法
│   └── GA/              # 遗传算法
├── Data/                # 数据目录
│   ├── InputData/       # 输入数据
│   └── OutputData/      # 输出数据
├── Environment/         # 环境配置
├── RPC/                 # RPC服务实现
└── requirements.txt     # 依赖包列表
```

## 主要功能

### 1. 调度算法
- **遗传算法（GA）**: 实现基于遗传算法的生产调度优化
- **EST-EET规则**: 最早开始时间-最早结束时间调度规则
- **EST-SPT规则**: 最早开始时间-最短处理时间调度规则
- **加权调度**: 支持优先级加权的调度算法

### 2. RPC服务
- **客户端-服务器架构**: 基于Socket的RPC通信
- **多算法支持**: 支持多种调度算法的远程调用
- **实时状态监控**: 服务器状态和算法执行状态监控
- **交互式界面**: 支持命令行交互和演示模式

### 3. 数据处理
- **JSON格式**: 支持JSON格式的输入输出数据
- **甘特图生成**: 自动生成调度结果的甘特图
- **性能分析**: 调度结果统计和性能分析

## 安装和运行

### 环境要求
- Python 3.7+
- 依赖包：见 `requirements.txt`

### 安装步骤

1. 克隆项目
```bash
git clone <your-repository-url>
cd APS_RPC_demo
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

### 运行方式

#### 1. 启动RPC服务器
```bash
cd RPC
python start_scheduling_server.py
```

#### 2. 运行客户端
```bash
cd RPC
python start_scheduling_client.py
```

#### 3. 运行演示
```bash
cd RPC
python demo_scheduling_rpc.py
```

#### 4. 运行测试
```bash
cd RPC
python test_scheduling_rpc.py
```

#### 5. 直接运行GA算法
```bash
cd Algorithms/GA
python ga_scheduler.py
```

## 使用示例

### 1. RPC服务演示
```python
from RPC.scheduling_rpc_client import SchedulingRPCClient

# 创建客户端
with SchedulingRPCClient() as client:
    # 获取服务器状态
    status = client.get_server_status()
    print(f"Server Status: {status}")
    
    # 运行调度算法
    result = client.est_eet_weighted_scheduling(input_data, alpha=0.7)
    print(f"Result: {result}")
```

### 2. GA算法使用
```python
from Algorithms.GA.ga_scheduler import GAScheduler, GAConfig
from Environment.dfjspt_env import FjspMaEnv

# 创建环境和配置
env = FjspMaEnv({'train_or_eval_or_test': 'test', 'inputdata_json': 'input.json'})
config = GAConfig(population_size=100, generations=50)
ga = GAScheduler(env, config)

# 运行遗传算法
best_solution = ga.evolve()
print(f"Best fitness: {best_solution.fitness}")
```

## 配置说明

### GA算法参数
- `population_size`: 种群大小
- `generations`: 迭代代数
- `crossover_rate`: 交叉概率
- `mutation_rate`: 变异概率
- `elitism`: 精英保留数量
- `alpha`: 优先级权重

### RPC服务配置
- 默认端口: 8080
- 缓冲区大小: 8192
- 超时时间: 60秒

## 输出格式

### 调度结果JSON格式
```json
{
    "status": "success",
    "makespan": 150.5,
    "execution_time": 2.34,
    "output_file": "output.json",
    "schedule": {
        "job_assignments": [...],
        "machine_assignments": [...],
        "start_times": [...],
        "finish_times": [...]
    }
}
```

## 开发说明

### 添加新算法
1. 在 `Algorithms/` 目录下创建新的算法模块
2. 实现算法接口
3. 在RPC服务中注册新算法
4. 更新客户端支持

### 扩展RPC功能
1. 在 `RPC/rpc_server.py` 中添加新的RPC方法
2. 在 `RPC/rpc_client.py` 中添加对应的客户端方法
3. 更新测试用例

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 创建 Pull Request

## 更新日志

### v1.0.0
- 初始版本发布
- 实现基础RPC服务
- 支持GA算法和启发式规则
- 添加演示和测试功能 