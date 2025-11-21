# Fixed Schedule 功能详解

## 功能概述

**Industry Standard Formats: Supports Mooncake trace format and custom JSONL datasets when using the --fixed-schedule option**

这个功能允许 AIPerf 使用**固定调度（Fixed Schedule）**模式来执行基准测试，支持两种数据格式：
1. **Mooncake Trace 格式**：行业标准的跟踪格式
2. **自定义 JSONL 数据集**：包含时间戳的自定义格式

## 核心概念

### 1. Fixed Schedule（固定调度）

Fixed Schedule 是一种精确的时间控制模式，它根据数据文件中指定的时间戳（timestamp）来精确控制每个请求的发送时间。这与传统的并发模式或请求速率模式不同：

- **并发模式**：同时发送 N 个请求
- **请求速率模式**：按固定速率（如每秒 10 个）发送请求
- **固定调度模式**：在**精确的时间点**发送请求（如第 0ms、500ms、1000ms）

### 2. Mooncake Trace 格式

Mooncake Trace 是一个行业标准的跟踪格式，用于记录 AI 推理请求的元数据。它参考了 [Mooncake 项目](https://github.com/kvcache-ai/Mooncake) 的格式规范。

**格式定义**：

```python
class MooncakeTrace:
    type: Literal["mooncake_trace"] = "mooncake_trace"
    
    # 输入字段（二选一）
    input_length: int | None = None      # 输入序列长度（token数）
    text_input: str | None = None        # 实际文本输入
    
    # 可选字段
    output_length: int | None = None     # 输出序列长度
    hash_ids: list[int] | None = None    # 哈希ID（用于模拟文本重用）
    timestamp: int | None = None         # 时间戳（毫秒）- 用于固定调度
    delay: int | None = None             # 延迟（毫秒）
    session_id: str | None = None        # 会话ID
```

**示例 JSONL 文件**：

```jsonl
{"timestamp": 0, "input_length": 100, "output_length": 200, "hash_ids": [1001]}
{"timestamp": 500, "input_length": 200, "output_length": 400, "hash_ids": [1002]}
{"timestamp": 1000, "input_length": 550, "output_length": 500, "hash_ids": [1003, 1005]}
{"timestamp": 2000, "text_input": "What is deep learning?", "output_length": 300}
```

### 3. 自定义 JSONL 数据集

除了 Mooncake Trace 格式，AIPerf 还支持其他自定义 JSONL 格式，只要它们包含 `timestamp` 字段：

**Single Turn 格式示例**：
```jsonl
{"timestamp": 0, "text": "What is deep learning?"}
{"timestamp": 1000, "text": "Who are you?"}
{"timestamp": 2000, "text": "What is AI?"}
```

**Multi Turn 格式示例**：
```jsonl
{"timestamp": 0, "session_id": "conv-1", "turns": [{"role": "user", "content": "Hello"}]}
{"timestamp": 2000, "session_id": "conv-1", "turns": [{"role": "user", "content": "How are you?"}]}
```

## 实现机制

### 1. 自动检测机制

当使用 `mooncake_trace` 数据集类型时，系统会自动检测文件中是否包含 `timestamp` 字段：

```python
def _should_use_fixed_schedule_for_mooncake_trace(self) -> bool:
    """检查 mooncake_trace 数据集是否有时间戳，应该使用固定调度"""
    if self.input.custom_dataset_type != CustomDatasetType.MOONCAKE_TRACE:
        return False
    
    # 读取文件，检查是否包含 timestamp 字段
    with open(self.input.file) as f:
        for line in f:
            data = load_json_str(line)
            if "timestamp" in data and data["timestamp"] is not None:
                return True
    return False
```

如果检测到时间戳，系统会自动启用 `fixed_schedule` 模式。

### 2. Fixed Schedule Strategy

Fixed Schedule Strategy 负责根据时间戳精确调度请求：

```python
@CreditIssuingStrategyFactory.register(TimingMode.FIXED_SCHEDULE)
class FixedScheduleStrategy(CreditIssuingStrategy):
    def __init__(self, config, credit_manager, schedule: list[tuple[int, str]]):
        # schedule: [(timestamp_ms, conversation_id), ...]
        self._schedule = schedule
        self._num_requests = len(self._schedule)
        
    async def _execute_single_phase(self, phase_stats):
        start_time_ms = self._perf_counter_ms()
        
        for timestamp in self._sorted_timestamp_keys:
            # 计算等待时间
            wait_duration_ms = (timestamp - self._schedule_zero_ms) - (
                self._perf_counter_ms() - start_time_ms
            )
            
            if wait_duration_ms > 0:
                await asyncio.sleep(wait_duration_ms / 1000)
            
            # 在精确的时间点发送请求
            for conversation_id in self._timestamp_groups[timestamp]:
                await self.credit_manager.drop_credit(
                    credit_phase=CreditPhase.PROFILING,
                    conversation_id=conversation_id,
                    credit_drop_ns=None,  # 立即发送
                )
```

### 3. 时间偏移处理

支持三种时间偏移模式：

1. **自动偏移** (`--fixed-schedule-auto-offset`)：
   - 自动将第一个时间戳设为 0
   - 所有后续时间戳相对于第一个时间戳

2. **手动起始偏移** (`--fixed-schedule-start-offset`)：
   - 指定起始时间戳（毫秒）
   - 只执行该时间戳之后的请求

3. **时间窗口** (`--fixed-schedule-start-offset` + `--fixed-schedule-end-offset`)：
   - 只执行指定时间窗口内的请求
   - 用于测试特定时间段

## 使用场景

### 1. 流量回放（Traffic Replay）

使用生产环境的真实请求时间戳来重现流量模式：

```bash
# 从生产日志提取的时间戳数据
cat > production_trace.jsonl << 'EOF'
{"timestamp": 0, "input_length": 100}
{"timestamp": 234, "input_length": 150}
{"timestamp": 567, "input_length": 200}
{"timestamp": 1234, "input_length": 300}
EOF

aiperf profile \
    --input-file production_trace.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule-auto-offset
```

### 2. 峰值负载测试

测试系统在已知高流量时段的行为：

```bash
# 只测试 2-6 秒时间窗口的请求
aiperf profile \
    --input-file peak_traffic.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule-start-offset 2000 \
    --fixed-schedule-end-offset 6000
```

### 3. 时间性能分析

研究性能如何随请求时间变化：

```bash
# 精确控制每个请求的发送时间
aiperf profile \
    --input-file precise_schedule.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule
```

### 4. SLA 验证

验证系统在特定时间约束下的性能：

```bash
# 模拟突发流量模式
cat > burst_pattern.jsonl << 'EOF'
{"timestamp": 0, "input_length": 100}
{"timestamp": 10, "input_length": 100}
{"timestamp": 20, "input_length": 100}
{"timestamp": 1000, "input_length": 200}
EOF

aiperf profile \
    --input-file burst_pattern.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule-auto-offset
```

## 代码实现位置

### 关键文件

1. **数据集加载器**：
   - `src/aiperf/dataset/loader/mooncake_trace.py` - Mooncake Trace 格式加载器
   - `src/aiperf/dataset/loader/models.py` - MooncakeTrace 模型定义

2. **调度策略**：
   - `src/aiperf/timing/fixed_schedule_strategy.py` - 固定调度策略实现
   - `src/aiperf/timing/timing_manager.py` - 时序管理器

3. **配置检测**：
   - `src/aiperf/common/config/user_config.py` - 自动检测时间戳逻辑

4. **文档**：
   - `docs/tutorials/fixed-schedule.md` - 使用教程

## 与其他模式的区别

| 特性 | 并发模式 | 请求速率模式 | 固定调度模式 |
|------|---------|------------|------------|
| 时间控制 | 无 | 相对时间（速率） | 绝对时间（时间戳） |
| 适用场景 | 压力测试 | 持续负载 | 流量回放、精确测试 |
| 数据格式 | 任意 | 任意 | 需要时间戳 |
| 可重现性 | 低 | 中 | 高 |

## 总结

Fixed Schedule 功能通过支持 Mooncake Trace 格式和自定义 JSONL 数据集，使得 AIPerf 能够：

1. **精确控制**：在毫秒级精度下控制请求发送时间
2. **流量回放**：重现生产环境的真实流量模式
3. **时间分析**：研究时间相关的性能特征
4. **标准化**：支持行业标准的 Mooncake Trace 格式
5. **灵活性**：支持自定义 JSONL 格式，只要包含时间戳

这使得 AIPerf 不仅适用于传统的性能测试，还能进行更精细的时间敏感型测试和流量回放分析。

