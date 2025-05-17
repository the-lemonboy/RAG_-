# Her-Server 项目

## 日志系统

本项目使用统一的日志记录系统，支持控制台输出和文件记录，具有以下特点：

### TODO

* [ ] txt、docx、ppt、excel数据向量化处理。
* [ ] pdf数据向量化处理。

### 特性

- 统一的日志格式和配置
- 支持按模块分类的日志记录
- 日志文件自动轮转（默认单文件最大 10MB，保留 5 个备份）
- 可通过环境变量自定义日志级别和格式

### 使用方法

在模块中使用日志记录器：

```python
from logger_config import get_module_logger

# 获取模块日志记录器
logger = get_module_logger("your_module_name")

# 使用日志记录器
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

### 环境变量配置

可以通过以下环境变量自定义日志行为：

- `LOG_LEVEL`: 日志级别 (debug, info, warning, error, critical)
- `LOG_FORMAT`: 日志格式
- `LOG_DIR`: 日志文件目录
- `LOG_MAX_BYTES`: 单个日志文件最大字节数
- `LOG_BACKUP_COUNT`: 保留的备份文件数量

### 日志文件位置

日志文件默认保存在项目根目录的 `logs` 文件夹中，按模块名称分别存储。
