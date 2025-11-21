# AIPerf 技术问题解答

## 问题 1：数据采集方式

### 答案：基于 Client 端性能打点

AIPerf 的性能数据采集**完全在 Client 端进行**，不依赖推理框架返回的内容。具体实现如下：

### 1.1 时间戳采集机制

**核心代码位置**：`src/aiperf/workers/worker.py` 和 `src/aiperf/transports/aiohttp_client.py`

#### 关键时间点记录

```python
# Worker 端打点
drop_perf_ns = time.perf_counter_ns()  # 接收 credit 的时间
pre_send_perf_ns = time.perf_counter_ns()  # 发送请求前
timestamp_ns = time.time_ns()  # 系统时间戳

# Transport 端打点（HTTP 请求）
record.start_perf_ns = time.perf_counter_ns()  # 请求开始
record.recv_start_perf_ns = time.perf_counter_ns()  # 开始接收响应
record.end_perf_ns = time.perf_counter_ns()  # 请求完成
```

#### 使用的计时器

1. **`time.perf_counter_ns()`**：
   - 高精度性能计数器（纳秒级）
   - 单调递增，不受系统时间调整影响
   - 用于计算延迟、吞吐量等性能指标

2. **`time.time_ns()`**：
   - 系统时钟时间戳（纳秒级）
   - 用于记录请求的绝对时间
   - 用于时间对齐和排序

### 1.2 数据采集流程

```
1. TimingManager 发送 CreditDropMessage
   ↓
2. Worker 接收 Credit（记录 drop_perf_ns）
   ↓
3. Worker 从 DatasetManager 获取对话数据
   ↓
4. Worker 准备发送请求（记录 pre_send_perf_ns）
   ↓
5. Transport 发送 HTTP 请求（记录 start_perf_ns）
   ↓
6. Transport 开始接收响应（记录 recv_start_perf_ns）
   ↓
7. Transport 完成接收（记录 end_perf_ns）
   ↓
8. Worker 解析响应，创建 RequestRecord
   ↓
9. Worker 发送 InferenceResultsMessage 到 RecordsManager
```

### 1.3 RequestRecord 结构

**位置**：`src/aiperf/common/models/record_models.py`

```python
class RequestRecord:
    # 时间戳（纳秒）
    timestamp_ns: int  # 系统时间戳
    start_perf_ns: int  # 请求开始（perf_counter）
    end_perf_ns: int  # 请求结束（perf_counter）
    recv_start_perf_ns: int | None  # 开始接收响应
    
    # 延迟计算
    credit_drop_latency: int | None  # Credit 接收延迟
    
    # 响应数据
    responses: list[TextResponse]  # 从服务器接收的响应
    status: int | None  # HTTP 状态码
    
    # 错误信息
    error: ErrorDetails | None
```

### 1.4 关键指标计算

所有性能指标都是基于 Client 端的时间戳计算：

- **请求延迟**：`end_perf_ns - start_perf_ns`
- **首 Token 延迟**：`first_token_perf_ns - start_perf_ns`
- **Token 间延迟**：基于 SSE 流中每个 chunk 的时间戳
- **吞吐量**：基于完成时间和 Token 数量计算

### 1.5 与推理框架的关系

- **不依赖框架返回的时间戳**：即使推理框架返回了时间信息，AIPerf 也不使用
- **只使用响应内容**：从响应中提取文本、Token 数量等数据
- **完全 Client 端测量**：所有时间测量都在 AIPerf 的 Client 端完成

---

## 问题 2：硬件和服务化平台耦合

### 答案：核心功能不耦合，GPU 遥测可选

AIPerf 的**核心性能测试功能与硬件平台解耦**，但 GPU 遥测功能需要 NVIDIA 硬件支持。

### 2.1 核心功能（无硬件依赖）

#### 支持的推理服务

AIPerf 通过标准的 HTTP/HTTPS 协议与推理服务通信，支持：

- **任何提供 HTTP API 的推理服务**：
  - vLLM
  - TensorRT-LLM
  - Triton Inference Server
  - OpenAI API
  - HuggingFace TEI
  - 自定义推理服务

- **任何硬件平台**：
  - NVIDIA GPU
  - AMD GPU
  - CPU
  - 云端服务（AWS、Azure、GCP 等）

#### 代码证据

**Transport 层**：`src/aiperf/transports/aiohttp_transport.py`
- 使用标准的 aiohttp 库
- 支持 HTTP/HTTPS 协议
- 无硬件特定代码

**Endpoint 层**：`src/aiperf/endpoints/`
- 支持多种 API 格式（OpenAI、HuggingFace、NIM 等）
- 通过工厂模式可扩展
- 无硬件依赖

### 2.2 GPU 遥测功能（可选，NVIDIA 特定）

#### 实现方式

**位置**：`src/aiperf/gpu_telemetry/`

```python
# 使用 DCGM (Data Center GPU Manager) 收集 GPU 指标
class TelemetryDataCollector:
    def __init__(self, dcgm_url: str):
        # 从 DCGM exporter 的 /metrics 端点获取数据
        # DCGM 是 NVIDIA 的 GPU 监控工具
        self._dcgm_url = dcgm_url
```

#### 支持的指标

- GPU 功耗（Power Usage）
- GPU 利用率（Utilization）
- GPU 内存使用（Memory Used）
- GPU 温度（Temperature）
- 能耗（Energy Consumption）

#### 可选性

- **默认禁用**：如果不配置 `--gpu-telemetry-urls`，GPU 遥测不会启用
- **不影响核心功能**：即使没有 GPU 遥测，性能测试功能完全正常
- **支持多节点**：可以配置多个 DCGM 端点，支持多节点 GPU 监控

### 2.3 服务化平台支持

#### 本地部署

- **多进程模式**：`controller/multiprocess_service_manager.py`
- 支持单机多进程部署
- 无特殊依赖

#### Kubernetes 部署

- **Kubernetes 模式**：`controller/kubernetes_service_manager.py`
- 支持在 K8s 集群中部署
- 支持服务发现和自动扩缩容

#### 云平台支持

- **任何云平台**：只要推理服务提供 HTTP API
- **无平台锁定**：不依赖特定的云服务
- **灵活配置**：通过 URL 和认证配置连接

### 2.4 扩展性

#### 添加新的硬件监控

可以通过扩展 `TelemetryDataCollector` 来支持其他硬件：

```python
# 示例：添加 AMD GPU 监控
class AMDGPUTelemetryCollector(TelemetryDataCollector):
    def __init__(self, rocm_url: str):
        # 实现 AMD ROCm 监控逻辑
        pass
```

#### 添加新的服务化平台

可以通过实现新的 `ServiceManager` 来支持其他平台：

```python
# 示例：添加 Docker Compose 支持
@ServiceManagerFactory.register(ServiceRunType.DOCKER_COMPOSE)
class DockerComposeServiceManager(ServiceManagerProtocol):
    # 实现 Docker Compose 服务管理
    pass
```

---

## 问题 3：实时展示性能结果的技术方案

### 答案：基于消息总线和 UI 框架的实时更新机制

### 3.1 整体架构

```
RecordsManager (数据源)
    ↓ (ZMQ PUB/SUB)
SystemController (消息分发)
    ↓ (Hook 机制)
UI Dashboard (Textual 框架)
    ↓ (实时渲染)
用户界面
```

### 3.2 核心组件

#### 3.2.1 数据生成层

**位置**：`src/aiperf/records/records_manager.py`

```python
@background_task(interval=Environment.UI.REALTIME_METRICS_INTERVAL)
async def _report_realtime_inference_metrics_task(self):
    """定期生成并发送实时指标"""
    while not self.stop_requested:
        await asyncio.sleep(Environment.UI.REALTIME_METRICS_INTERVAL)
        
        # 检查是否有新数据
        if self.processing_stats.processed == self._previous_realtime_records:
            continue  # 无新数据，跳过
        
        self._previous_realtime_records = self.processing_stats.processed
        
        # 生成实时指标
        await self._report_realtime_metrics()

async def _report_realtime_metrics(self):
    """生成并发布实时指标"""
    # 1. 从所有结果处理器汇总指标
    metrics = await self._generate_realtime_metrics()
    
    # 2. 通过消息总线发布
    await self.publish(
        RealtimeMetricsMessage(
            service_id=self.service_id,
            metrics=metrics,
        )
    )
    
    # 3. 如果启用 GPU 遥测，也发布遥测指标
    if self.user_config.gpu_telemetry_mode == GPUTelemetryMode.REALTIME_DASHBOARD:
        telemetry_metrics = await self._generate_realtime_telemetry_metrics()
        await self.publish(
            RealtimeTelemetryMetricsMessage(
                service_id=self.service_id,
                metrics=telemetry_metrics,
            )
        )
```

#### 3.2.2 消息总线层

**通信模式**：ZMQ PUB/SUB

```python
# RecordsManager 发布消息
await self.publish(RealtimeMetricsMessage(...))

# SystemController 订阅消息
@on_message(MessageType.REALTIME_METRICS)
async def _on_realtime_metrics(self, message: RealtimeMetricsMessage):
    # 触发 Hook
    await self.trigger_hook(
        AIPerfHook.ON_REALTIME_METRICS,
        metrics=message.metrics
    )
```

#### 3.2.3 UI 层

**位置**：`src/aiperf/ui/dashboard/`

**技术栈**：
- **Textual**：Python 终端 UI 框架
- **Rich**：富文本渲染
- **异步更新**：基于 asyncio 的实时更新

```python
class RealtimeMetricsDashboard(Container):
    def on_realtime_metrics(self, metrics: list[MetricResult]):
        """处理实时指标更新"""
        if not self.metrics:
            # 首次更新，显示表格
            self.query_one("#metrics-table").remove_class("hidden")
            self.query_one("#realtime-metrics-status").add_class("hidden")
        
        # 更新表格数据
        self.metrics = metrics
        if self.metrics_table:
            self.metrics_table.update(metrics)

class RealtimeMetricsTable(Widget):
    def update(self, metrics: list[MetricResult]):
        """更新表格内容"""
        for metric in metrics:
            if metric.tag in self._metric_row_keys:
                # 更新现有行
                self._update_single_row(row_cells, row_key)
            else:
                # 添加新行
                row_key = self.data_table.add_row(*row_cells)
                self._metric_row_keys[metric.tag] = row_key
```

### 3.3 数据流

```
1. RecordsManager 收集 RequestRecord
   ↓
2. RecordProcessor 处理记录，计算指标
   ↓
3. ResultsProcessor 汇总指标（每 N 秒）
   ↓
4. RecordsManager 生成 RealtimeMetricsMessage
   ↓
5. 通过 ZMQ PUB 发布消息
   ↓
6. SystemController 接收消息，触发 Hook
   ↓
7. UI Dashboard 的 Hook 处理器被调用
   ↓
8. Textual Widget 更新表格内容
   ↓
9. 终端界面实时刷新显示
```

### 3.4 更新频率

**配置位置**：`src/aiperf/common/environment.py`

```python
class UI:
    REALTIME_METRICS_INTERVAL = 1.0  # 每秒更新一次
```

### 3.5 性能优化

#### 3.5.1 增量更新

- 只更新变化的指标
- 使用行键（RowKey）快速定位需要更新的行
- 避免全表重绘

#### 3.5.2 异步处理

- 指标计算在后台任务中进行
- UI 更新不阻塞主事件循环
- 使用 asyncio 实现非阻塞更新

#### 3.5.3 数据过滤

- 跳过不需要显示的指标（ERROR_ONLY、INTERNAL 等）
- 只处理有效的指标数据
- 避免处理异常数据

### 3.6 支持的实时指标

#### 推理性能指标

- 请求延迟（Request Latency）
- 首 Token 延迟（Time to First Token）
- Token 间延迟（Inter Token Latency）
- 吞吐量（Throughput）
- Token 速率（Token Rate）

#### GPU 遥测指标（可选）

- GPU 功耗
- GPU 利用率
- GPU 内存使用
- GPU 温度

### 3.7 UI 模式

#### Dashboard 模式

- 使用 Textual 框架
- 实时表格更新
- 支持多面板显示
- 支持最大化/最小化面板

#### Console 模式

- 使用 Rich 库
- 定期打印指标表格
- 适合脚本和自动化场景

---

## 总结

### 数据采集
- ✅ **完全 Client 端打点**：使用 `time.perf_counter_ns()` 和 `time.time_ns()`
- ✅ **不依赖推理框架**：只使用响应内容，不依赖框架返回的时间戳
- ✅ **高精度测量**：纳秒级精度，单调递增计数器

### 硬件耦合
- ✅ **核心功能解耦**：支持任何提供 HTTP API 的推理服务
- ✅ **GPU 遥测可选**：需要 NVIDIA DCGM，但不影响核心功能
- ✅ **平台无关**：支持本地、K8s、云平台等多种部署方式

### 实时展示
- ✅ **消息驱动**：基于 ZMQ PUB/SUB 实现解耦
- ✅ **异步更新**：使用 asyncio 和 Textual 框架
- ✅ **高性能**：增量更新，避免全表重绘
- ✅ **可配置**：支持 Dashboard 和 Console 两种模式

