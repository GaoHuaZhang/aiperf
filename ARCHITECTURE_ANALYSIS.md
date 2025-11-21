# AIPerf 架构分析

## 一、整体架构概述

AIPerf 是一个基于微服务架构的 AI 模型性能基准测试工具，采用异步编程模型和消息驱动架构。

### 1.1 核心设计理念

- **微服务架构**：系统由多个独立服务组成，通过消息总线（ZMQ）进行通信
- **工厂模式**：使用工厂模式实现组件的注册和创建，支持插件化扩展
- **异步编程**：基于 asyncio 和 uvloop 实现高并发处理
- **多进程支持**：支持多进程部署，可水平扩展
- **模块化设计**：各模块职责清晰，易于扩展和维护

### 1.2 系统层次结构

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Layer (cli.py)                    │
│              Cyclopts-based command interface            │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              System Controller                          │
│  - 服务生命周期管理                                       │
│  - 服务注册与发现                                         │
│  - 配置分发                                              │
│  - 结果聚合与导出                                         │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
│   Workers    │ │  Timing    │ │  Records   │
│              │ │  Manager   │ │  Manager   │
│ - 发送请求   │ │ - 调度控制 │ │ - 结果收集 │
│ - 处理响应   │ │ - 信用管理 │ │ - 指标计算 │
└──────────────┘ └────────────┘ └────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
│   Dataset    │ │ Telemetry  │ │  Endpoints │
│   Manager    │ │  Manager   │ │            │
│ - 数据加载   │ │ - GPU监控  │ │ - API适配  │
│ - 对话管理   │ │ - 指标收集 │ │ - 格式转换 │
└──────────────┘ └────────────┘ └────────────┘
```

## 二、核心组件详解

### 2.1 SystemController（系统控制器）

**位置**：`controller/system_controller.py`

**职责**：
- 管理所有服务的生命周期（启动、停止、配置）
- 协调服务间的通信
- 处理服务注册和心跳
- 聚合和导出基准测试结果
- 处理信号和优雅关闭

**关键实现**：

```63:135:src/aiperf/controller/system_controller.py
@ServiceFactory.register(ServiceType.SYSTEM_CONTROLLER)
class SystemController(SignalHandlerMixin, BaseService):
    """System Controller service.

    This service is responsible for managing the lifecycle of all other services.
    It will start, stop, and configure all other services.
    """

    def __init__(
        self,
        user_config: UserConfig,
        service_config: ServiceConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
        )
        self.debug("Creating System Controller")
        if Environment.DEV.MODE:
            # Print a warning message to the console if developer mode is enabled, once at load time
            print_developer_mode_warning()

        self._was_cancelled = False
        # List of required service types, in no particular order
        # These are services that must be running before the system controller can start profiling
        self.required_services: dict[ServiceTypeT, int] = {
            ServiceType.DATASET_MANAGER: 1,
            ServiceType.TIMING_MANAGER: 1,
            ServiceType.WORKER_MANAGER: 1,
            ServiceType.RECORDS_MANAGER: 1,
        }
        if self.service_config.record_processor_service_count is not None:
            self.required_services[ServiceType.RECORD_PROCESSOR] = (
                self.service_config.record_processor_service_count
            )
            self.scale_record_processors_with_workers = False
        else:
            self.scale_record_processors_with_workers = True

        self.proxy_manager: ProxyManager = ProxyManager(
            service_config=self.service_config
        )
        self.service_manager: ServiceManagerProtocol = (
            ServiceManagerFactory.create_instance(
                self.service_config.service_run_type.value,
                required_services=self.required_services,
                user_config=self.user_config,
                service_config=self.service_config,
                log_queue=get_global_log_queue(),
            )
        )
        self.ui: AIPerfUIProtocol = AIPerfUIFactory.create_instance(
            self.service_config.ui_type,
            service_config=self.service_config,
            user_config=self.user_config,
            log_queue=get_global_log_queue(),
            controller=self,
        )
        self.attach_child_lifecycle(self.ui)
        self._stop_tasks: set[asyncio.Task] = set()
        self._profile_results: ProcessRecordsResult | None = None
        self._exit_errors: list[ExitErrorInfo] = []
        self._telemetry_results: TelemetryResults | None = None
        self._profile_results_received = False
        self._should_wait_for_telemetry = False

        self._shutdown_triggered = False
        self._shutdown_lock = asyncio.Lock()
        self._endpoints_configured: list[str] = []
        self._endpoints_reachable: list[str] = []
        self.debug("System Controller created")
```

**关键特性**：
- 使用 `ServiceManager` 管理服务的启动和停止
- 通过 `ProxyManager` 管理 ZMQ 代理
- 支持多种 UI 类型（Dashboard、Console）
- 协调配置和启动流程

### 2.2 Worker（工作进程）

**位置**：`workers/worker.py`

**职责**：
- 从 TimingManager 接收信用（Credit）
- 从 DatasetManager 获取对话数据
- 调用推理 API（通过 InferenceClient）
- 处理多轮对话和延迟模拟
- 返回结果到 RecordsManager

**关键实现**：

```50:106:src/aiperf/workers/worker.py
@ServiceFactory.register(ServiceType.WORKER)
class Worker(PullClientMixin, BaseComponentService, ProcessHealthMixin):
    """Worker is primarily responsible for making API calls to the inference server.
    It also manages the conversation between turns and returns the results to the Inference Results Parsers.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            pull_client_address=CommAddress.CREDIT_DROP,
            pull_client_bind=False,
            # NOTE: We set the max concurrency to the same as the HTTP connection limit to ensure
            # that the worker will not receive any more credits while the connection limit is reached.
            pull_client_max_concurrency=Environment.HTTP.CONNECTION_LIMIT,
            **kwargs,
        )

        self.debug(lambda: f"Worker process __init__ (pid: {self._process.pid})")

        self.health_check_interval = Environment.WORKER.HEALTH_CHECK_INTERVAL

        self.task_stats: WorkerTaskStats = WorkerTaskStats()

        self.credit_return_push_client: PushClientProtocol = (
            self.comms.create_push_client(
                CommAddress.CREDIT_RETURN,
            )
        )
        self.inference_results_push_client: PushClientProtocol = (
            self.comms.create_push_client(
                CommAddress.RAW_INFERENCE_PROXY_FRONTEND,
            )
        )
        self.conversation_request_client: RequestClientProtocol = (
            self.comms.create_request_client(
                CommAddress.DATASET_MANAGER_PROXY_FRONTEND,
            )
        )

        self.model_endpoint = ModelEndpointInfo.from_user_config(self.user_config)

        self.inference_client: InferenceClient = InferenceClient(
            model_endpoint=self.model_endpoint
        )
        self.debug(
            lambda: f"Creating inference client for {self.model_endpoint.endpoint.type}, "
            f"class: {self.inference_client.__class__.__name__}",
        )
        self.attach_child_lifecycle(self.inference_client)
```

**关键特性**：
- 使用 Pull 客户端接收信用，控制并发
- 支持请求取消（timeout）
- 支持多轮对话的延迟模拟
- 通过 InferenceClient 适配不同的 API 格式

### 2.3 TimingManager（时序管理器）

**位置**：`timing/timing_manager.py`

**职责**：
- 根据配置的基准测试模式生成请求调度
- 管理信用（Credit）的发放和回收
- 支持多种调度策略：
  - 并发模式（Concurrency）
  - 请求速率模式（Request Rate）
  - 固定调度模式（Fixed Schedule）
  - 跟踪回放模式（Trace Replay）

**关键实现**：

```50:149:src/aiperf/timing/timing_manager.py
@implements_protocol(ServiceProtocol)
@ServiceFactory.register(ServiceType.TIMING_MANAGER)
class TimingManager(PullClientMixin, BaseComponentService, CreditPhaseMessagesMixin):
    """
    The TimingManager service is responsible to generate the schedule and issuing
    timing credits for requests.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            pull_client_address=CommAddress.CREDIT_RETURN,
            pull_client_bind=True,
        )
        self.debug("Timing manager __init__")
        self.config = TimingManagerConfig.from_user_config(self.user_config)

        self.dataset_request_client: RequestClientProtocol = (
            self.comms.create_request_client(
                CommAddress.DATASET_MANAGER_PROXY_FRONTEND,
            )
        )
        self.credit_drop_push_client: PushClientProtocol = (
            self.comms.create_push_client(
                CommAddress.CREDIT_DROP,
                bind=True,
            )
        )

        self._credit_issuing_strategy: CreditIssuingStrategy | None = None

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the timing manager."""
        self.debug(f"Configuring credit issuing strategy for {self.service_id}")

        if self.config.timing_mode == TimingMode.FIXED_SCHEDULE:
            # This will block until the dataset is ready and the timing response is received
            dataset_timing_response: DatasetTimingResponse = await self.dataset_request_client.request(
                message=DatasetTimingRequest(
                    service_id=self.service_id,
                ),
                # NOTE: We use the dataset configuration timeout here because the dataset manager
                # may take longer than a normal request timeout to respond to this request. This is
                # because it blocks until the dataset is configured.
                timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
            )
            self.debug(
                lambda: f"Received dataset timing response: {dataset_timing_response}"
            )
            self.info("Using fixed schedule strategy")
            self._credit_issuing_strategy = (
                CreditIssuingStrategyFactory.create_instance(
                    TimingMode.FIXED_SCHEDULE,
                    config=self.config,
                    credit_manager=self,
                    schedule=dataset_timing_response.timing_data,
                )
            )
        else:
            self.info(f"Using {self.config.timing_mode.title()} strategy")
            self._credit_issuing_strategy = (
                CreditIssuingStrategyFactory.create_instance(
                    self.config.timing_mode,
                    config=self.config,
                    credit_manager=self,
                )
            )

        if not self._credit_issuing_strategy:
            raise InvalidStateError("No credit issuing strategy configured")
        self.debug(
            lambda: f"Timing manager configured with credit issuing strategy: {self._credit_issuing_strategy}"
        )

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message: CommandMessage) -> None:
        """Start the timing manager and issue credit drops according to the configured strategy."""
        self.debug("Starting profiling")

        self.debug("Waiting for timing manager to be initialized")
        await self.initialized_event.wait()
        self.debug("Timing manager initialized, starting profiling")

        if not self._credit_issuing_strategy:
            raise InvalidStateError("No credit issuing strategy configured")

        self.execute_async(self._credit_issuing_strategy.start())
        self.info(
            f"Credit issuing strategy for {self.config.timing_mode.title()} started"
        )
```

**关键特性**：
- 使用策略模式实现不同的调度算法
- 通过信用机制控制请求的发送时机
- 支持精确的时间调度（固定调度模式）

### 2.4 RecordsManager（记录管理器）

**位置**：`records/records_manager.py`

**职责**：
- 收集所有 Worker 返回的推理结果
- 计算性能指标（延迟、吞吐量等）
- 处理阶段完成检测
- 聚合 GPU 遥测数据
- 生成最终报告

**关键实现**：

```90:150:src/aiperf/records/records_manager.py
@implements_protocol(ServiceProtocol)
@ServiceFactory.register(ServiceType.RECORDS_MANAGER)
class RecordsManager(PullClientMixin, BaseComponentService):
    """
    The RecordsManager service is primarily responsible for holding the
    results returned from the workers.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            pull_client_address=CommAddress.RECORDS,
            pull_client_bind=True,
            pull_client_max_concurrency=Environment.ZMQ.PULL_MAX_CONCURRENCY,
        )

        #########################################################
        # Protected by processing_status_lock
        self.processing_status_lock: asyncio.Lock = asyncio.Lock()
        self.start_time_ns: int | None = None
        self.processing_stats: ProcessingStats = ProcessingStats()
        self.final_request_count: int | None = None
        self.end_time_ns: int | None = None
        self.sent_all_records_received: bool = False
        self.profile_cancelled: bool = False
        self.timeout_triggered: bool = False
        self.expected_duration_sec: float | None = None
        #########################################################

        self._completion_checker = PhaseCompletionChecker()

        self.error_summary: dict[ErrorDetails, int] = {}
        self.error_summary_lock: asyncio.Lock = asyncio.Lock()
        # Track per-worker statistics
        self.worker_stats: dict[str, ProcessingStats] = {}
        self.worker_stats_lock: asyncio.Lock = asyncio.Lock()

        self._previous_realtime_records: int | None = None

        self._telemetry_state = TelemetryTrackingState()
        self._telemetry_enable_event = asyncio.Event()

        self._metric_results_processors: list[ResultsProcessorProtocol] = []
        self._telemetry_results_processors: list[TelemetryResultsProcessorProtocol] = []
        self._telemetry_accumulator: TelemetryResultsProcessorProtocol | None = None

        for results_processor_type in ResultsProcessorFactory.get_all_class_types():
            try:
                results_processor = ResultsProcessorFactory.create_instance(
                    class_type=results_processor_type,
                    service_id=self.service_id,
                    service_config=self.service_config,
                    user_config=self.user_config,
                )
```

**关键特性**：
- 使用锁保护共享状态
- 支持实时指标更新
- 集成多种结果处理器（Metrics、Telemetry）
- 支持阶段完成检测

### 2.5 Endpoint 系统（端点适配层）

**位置**：`endpoints/`

**职责**：
- 适配不同的 API 格式（OpenAI、HuggingFace、NIM 等）
- 格式化请求负载
- 解析响应数据
- 提取文本、嵌入、排名等数据

**关键实现**：

```30:65:src/aiperf/endpoints/base_endpoint.py
@implements_protocol(EndpointProtocol)
class BaseEndpoint(AIPerfLoggerMixin, ABC):
    """Base for all endpoints.

    Endpoints handle API-specific formatting and parsing.
    """

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs):
        super().__init__(**kwargs)
        self.model_endpoint = model_endpoint

    @classmethod
    @abstractmethod
    def metadata(cls) -> EndpointMetadata:
        """Return endpoint metadata."""

    def get_endpoint_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Get endpoint headers (auth + user custom). Override to customize."""
        cfg = self.model_endpoint.endpoint
        headers = dict(cfg.headers) if cfg.headers else {}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        return headers

    def get_endpoint_params(self, request_info: RequestInfo) -> dict[str, str]:
        """Get endpoint URL query params (e.g., api-version). Override to customize."""
        cfg = self.model_endpoint.endpoint
        return dict(cfg.url_params) if cfg.url_params else {}

    @abstractmethod
    def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format request payload from RequestInfo.

        Uses request_info.turns[0] as the turn data (currently hardcoded to first turn).
        """

    @abstractmethod
    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse response. Return None to skip."""
```

**支持的端点类型**：
- OpenAI Chat Completions
- OpenAI Completions
- OpenAI Embeddings
- OpenAI Image Generation
- HuggingFace TEI Rankings
- NIM Rankings
- Cohere Rankings
- Template Endpoint（自定义模板）

## 三、关键特性实现方案

### 3.1 工厂模式与组件注册

**位置**：`common/factories.py`

**实现方案**：
- 使用装饰器模式注册组件
- 支持优先级覆盖
- 支持单例工厂和普通工厂

**关键代码**：

```74:183:src/aiperf/common/factories.py
class AIPerfFactory(Generic[ClassEnumT, ClassProtocolT]):
    """Defines a custom factory for AIPerf components.

    This class is used to create a factory for a given class type and protocol.

    Example:
    ```python
        # Define a new enum for the expected implementation types
        # This is optional, but recommended for type safety.
        class DatasetLoaderType(CaseInsensitiveStrEnum):
            FILE = "file"
            S3 = "s3"

        # Define a new class protocol.
        class DatasetLoaderProtocol(Protocol):
            def load(self) -> Dataset:
                pass

        # Create a new factory for a given class type and protocol.
        class DatasetFactory(FactoryMixin[DatasetLoaderType, DatasetLoaderProtocol]):
            pass

        # Register a new class type mapping to its corresponding class. It should implement the class protocol.
        @DatasetFactory.register(DatasetLoaderType.FILE)
        class FileDatasetLoader:
            def __init__(self, filename: str):
                self.filename = filename

            def load(self) -> Dataset:
                return Dataset.from_file(filename)

        DatasetConfig = {
            "type": DatasetLoaderType.FILE,
            "filename": "data.csv"
        }

        # Create a new instance of the class.
        if DatasetConfig["type"] == DatasetLoaderType.FILE:
            dataset_instance = DatasetFactory.create_instance(DatasetLoaderType.FILE, filename=DatasetConfig["filename"])
        else:
            raise ValueError(f"Unsupported dataset loader type: {DatasetConfig['type']}")

        dataset_instance.load()
    ```
    """

    _logger: AIPerfLogger
    _registry: dict[ClassEnumT | str, type[ClassProtocolT]]
    _override_priorities: dict[ClassEnumT | str, int]

    def __init_subclass__(cls) -> None:
        cls._registry = {}
        cls._override_priorities = {}
        cls._logger = AIPerfLogger(cls.__name__)
        super().__init_subclass__()

    @classmethod
    def register_all(
        cls, *class_types: ClassEnumT | str, override_priority: int = 0
    ) -> Callable:
        """Register multiple class types mapping to a single corresponding class.
        This is useful if a single class implements multiple types. Currently only supports
        registering as a single override priority for all types."""

        def decorator(class_cls: type[ClassProtocolT]) -> type[ClassProtocolT]:
            for class_type in class_types:
                cls.register(class_type, override_priority)(class_cls)
            return class_cls

        return decorator

    @classmethod
    def register(
        cls, class_type: ClassEnumT | str, override_priority: int = 0
    ) -> Callable:
        """Register a new class type mapping to its corresponding class.

        Args:
            class_type: The type of class to register
            override_priority: The priority of the override. The higher the priority,
                the more precedence the override has when multiple classes are registered
                for the same class type. Built-in classes have a priority of 0.

        Returns:
            Decorator for the class that implements the class protocol
        """

        def decorator(class_cls: type[ClassProtocolT]) -> type[ClassProtocolT]:
            existing_priority = cls._override_priorities.get(class_type, -1)
            if class_type in cls._registry and existing_priority >= override_priority:
                cls._logger.warning(
                    f"{class_type!r} class {cls._registry[class_type].__name__} already registered with same or higher priority "
                    f"({existing_priority}). The new registration of class {class_cls.__name__} with priority "
                    f"{override_priority} will be ignored.",
                )
                return class_cls

            if class_type not in cls._registry:
                cls._logger.debug(
                    lambda: f"{class_type!r} class {class_cls.__name__} registered with priority {override_priority}.",
                )
            else:
                cls._logger.warning(
                    f"{class_type!r} class {class_cls.__name__} with priority {override_priority} overrides "
                    f"already registered class {cls._registry[class_type].__name__} with lower priority ({existing_priority}).",
                )
            cls._registry[class_type] = class_cls
            cls._override_priorities[class_type] = override_priority
            return class_cls

        return decorator
```

### 3.2 消息总线通信

**位置**：`zmq/`, `common/base_comms.py`

**实现方案**：
- 基于 ZMQ 实现消息总线
- 支持多种通信模式（PUB/SUB、PUSH/PULL、REQ/REP）
- 使用代理模式实现服务发现

**通信模式**：
- **PUB/SUB**：广播消息（命令、状态更新）
- **PUSH/PULL**：工作队列（信用分发、结果收集）
- **REQ/REP**：请求-响应（数据集请求、配置查询）

### 3.3 服务生命周期管理

**位置**：`common/base_service.py`, `common/bootstrap.py`

**实现方案**：
- 基于状态机管理服务生命周期
- 使用钩子（Hooks）机制实现事件处理
- 支持优雅关闭和错误恢复

**生命周期状态**：
- INITIALIZING → RUNNING → STOPPING → STOPPED
- 支持 FAILED 状态处理

### 3.4 模块延迟加载

**位置**：`module_loader.py`

**实现方案**：
- 延迟加载所有模块，避免 CLI 启动时的性能开销
- 使用线程锁确保只加载一次
- 触发装饰器注册

**关键代码**：

```47:59:src/aiperf/module_loader.py
def ensure_modules_loaded() -> None:
    """Ensure all modules are loaded exactly once."""
    global _modules_loaded
    with _modules_loaded_lock:
        if not _modules_loaded:
            start_time = time.perf_counter()
            _logger.debug("Loading all modules")
            _load_all_modules()
            _logger.debug(
                f"Modules loaded in {time.perf_counter() - start_time:.2f} seconds"
            )
            _modules_loaded = True
```

### 3.5 GPU 遥测集成

**位置**：`gpu_telemetry/`

**实现方案**：
- 使用 DCGM（Data Center GPU Manager）收集 GPU 指标
- 支持自定义指标配置
- 实时收集和聚合 GPU 数据
- 与推理结果时间对齐

### 3.6 多进程服务管理

**位置**：`controller/multiprocess_service_manager.py`, `controller/kubernetes_service_manager.py`

**实现方案**：
- 支持本地多进程部署
- 支持 Kubernetes 部署
- 使用进程池管理 Worker 进程
- 处理进程间通信和日志聚合

### 3.7 数据导出系统

**位置**：`exporters/`

**实现方案**：
- 支持多种导出格式（CSV、JSON、Console）
- 可扩展的导出器架构
- 集成 GPU 遥测数据
- 生成详细的性能报告

## 四、数据流

### 4.1 请求流程

```
1. TimingManager 生成调度 → 发送 CreditDropMessage
2. Worker 接收 Credit → 从 DatasetManager 获取对话数据
3. Worker 调用 Inference API → 通过 Transport 发送请求
4. Worker 接收响应 → 解析并创建 RequestRecord
5. Worker 发送 InferenceResultsMessage → RecordsManager
6. RecordsManager 处理记录 → 计算指标 → 更新 UI
```

### 4.2 配置流程

```
1. CLI 解析用户配置 → UserConfig
2. SystemController 初始化 → 加载 ServiceConfig
3. SystemController 启动服务 → 等待服务注册
4. SystemController 发送 ProfileConfigureCommand → 所有服务配置
5. SystemController 发送 ProfileStartCommand → 开始基准测试
```

### 4.3 结果聚合流程

```
1. RecordsManager 收集所有 RequestRecord
2. RecordsManager 检测阶段完成 → 触发处理
3. RecordProcessor 处理记录 → 计算指标
4. ResultsProcessor 生成最终结果
5. SystemController 接收 ProcessRecordsResultMessage
6. SystemController 等待 TelemetryResults（如果启用）
7. ExporterManager 导出结果（CSV、JSON、Console）
```

## 五、扩展点

### 5.1 添加新的端点类型

1. 创建新的 Endpoint 类，继承 `BaseEndpoint`
2. 实现 `format_payload` 和 `parse_response` 方法
3. 使用 `@EndpointFactory.register(EndpointType.XXX)` 注册

### 5.2 添加新的调度策略

1. 实现 `CreditIssuingStrategy` 接口
2. 在 `CreditIssuingStrategyFactory` 中注册
3. 在 `TimingManagerConfig` 中添加新的模式

### 5.3 添加新的导出格式

1. 实现 `DataExporterProtocol` 或 `ConsoleExporterProtocol`
2. 在对应的 Factory 中注册
3. 在配置中启用

### 5.4 添加新的结果处理器

1. 实现 `ResultsProcessorProtocol`
2. 在 `ResultsProcessorFactory` 中注册
3. RecordsManager 会自动加载并使用

## 六、性能优化

### 6.1 异步编程
- 全面使用 asyncio，避免阻塞操作
- 使用 uvloop 提升事件循环性能

### 6.2 并发控制
- Worker 使用 Pull 客户端的 max_concurrency 控制并发
- HTTP 连接池限制连接数

### 6.3 内存管理
- 流式处理结果，避免大量数据驻留内存
- 及时释放不需要的记录

### 6.4 进程隔离
- 多进程架构避免 GIL 限制
- 进程间通过 ZMQ 高效通信

## 七、总结

AIPerf 采用了现代化的微服务架构设计，具有以下优势：

1. **高度模块化**：各组件职责清晰，易于理解和维护
2. **易于扩展**：工厂模式和协议设计支持插件化扩展
3. **高性能**：异步编程和多进程架构支持高并发
4. **灵活配置**：支持多种基准测试模式和 API 格式
5. **完善的监控**：集成 GPU 遥测和实时指标

该架构设计使得 AIPerf 能够适应不同的使用场景，从简单的性能测试到复杂的生产环境基准测试。

