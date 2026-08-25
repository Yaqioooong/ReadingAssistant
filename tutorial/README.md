# ReadingAssistant 开发教程

> 面向 **agent 开发新人**：会写基础 Python，但没接触过 RAG / LangGraph / FastAPI / SQLAlchemy 这类项目。
> 本教程以真实仓库 [ReadingAssistant](../README.md) 为蓝本，按 [开发计划](../DEVELOPMENT_PLAN.md) 的 P0–P7 分章节，逐阶段还原「每一步做什么、为什么、有什么坑」。

## 使用方式

- 按章节顺序阅读，每章对应一个开发阶段，与仓库实际代码一一对应。
- 每章结构统一：**本章目标 → 步骤拆解（做什么 / 为什么 / 关键代码）→ 关键语法与技巧 → 常见坑 → 测试与验证**。
- 建议边读边对照 `src/reading_assistant/` 下的真实代码；文中代码片段均取自仓库。

## 章节地图

| 章节 | 对应阶段 | 主题 |
| --- | --- | --- |
| [00 准备与项目概览](00-准备与项目概览.md) | — | 项目背景、技术栈、环境搭建、开发纪律 |
| [01 P0 工程基础](01-P0-工程基础.md) | P0 | 统一配置层、包结构修复、构建与测试基础设施 |
| [02 P1 解析器](02-P1-解析器.md) | P1 | txt / epub / pdf / docx 解析器与工厂 |
| [03 P2 存储层](03-P2-存储层.md) | P2 | SQLAlchemy 模型、分层去重、向量库适配器 |
| [04 P3 RAG 层](04-P3-RAG层.md) | P3 | 文本分块与向量检索 |
| [05 P4 LangGraph 流水线](05-P4-LangGraph流水线.md) | P4 | 入库图、问答图、HITL、检查点 |
| [06 P5 FastAPI 接口](06-P5-FastAPI接口.md) | P5 | API 路由、Schema、前端联调、静态托管 |
| [07 P6 CLI 与 P7 收尾](07-P6-CLI与P7收尾.md) | P6 / P7 | 命令行、覆盖率、文档与 git 提交 |

## 你会学到什么

- `src` 布局下 Python 包的导入与打包（`pyproject.toml` + setuptools）
- pydantic-settings 的配置优先级与 `SecretStr` 安全实践
- 可测试的解析器设计（抽象基类 + 工厂注册 + 异常体系）
- SQLAlchemy 2.0 类型化模型、事务与会话管理、唯一索引并发兜底
- 分层去重（文件哈希 → 内容哈希 → 唯一约束）的工程思路
- LangChain / LangGraph 1.x 的编排方式、状态图与检查点
- FastAPI 依赖注入、测试覆盖（`dependency_overrides`）与前后端契约对齐
- 覆盖率、lint、Conventional Commit 等工程收尾实践
