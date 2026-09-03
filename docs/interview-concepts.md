# 面试概念弹药库：ReAct / Function Calling / Memory / Eval / MCP

> 每个概念：一句话定义 → 本项目落点 → 面试怎么答（STAR 化）。

## 1. Function Calling（函数调用/工具调用）

**是什么**：模型输出不再只是文本，而是"我要调用哪个函数 + 参数 JSON"。把模型从"说话"升级为"行动"。OpenAI 2023 年引入，DeepSeek/Qwen 均已支持。

**本项目落点**：
- 双工具：`final_answer(answer)` / `request_clarification(missing_info)` —— 让 LLM 用**调用哪个工具**表态"信息是否充分"，替代关键词启发式
- 三层兜底：工具调用为主判定 → 纯文本 fallback 关键词检测 → 异常回退普通 invoke
- 验证流程：先单发真实 API 验证 tool calling 可用 → 再改代码 → 再补 mock 测试

**面试怎么答**：讲清楚"为什么不用关键词"——LLM 措辞不可控，关键词是猜；工具调用让模型在语义层自报，判定质量高一个量级。再补一句工程细节：真实环境验证过 DeepSeek function calling 才落地，带纯文本 fallback 保证稳定性。

## 2. ReAct（Reasoning + Acting）

**是什么**：思维链（Thought）+ 行动（Action）+ 观察（Observation）循环。模型边推理边调工具边看结果，迭代逼近答案。核心论文 2022《ReAct: Synergizing Reasoning and Acting in Language Models》。

**本项目落点**：尚未显式使用 ReAct 循环（当前是固定流水线 StateGraph）。可扩展方向：
- 把 `cache_check → retrieve → judge → answer` 改造成 ReAct agent：让模型自己决定"要不要检索、检索几次、是否需要澄清"
- 用 langgraph.prebuilt `create_react_agent` 或手写 `Thought/Action/Observation` 状态机

**面试怎么答**：能画出 ReAct 循环图，说出它与"固定流水线"的取舍——流水线确定性强、可观测、易评测；ReAct 灵活但不可控、token 成本高、可能死循环（要有 max_iterations 约束）。本项目选流水线是刻意的工程决策，不是不会 ReAct。

## 3. Memory（记忆）

**是什么**：让系统跨会话/跨轮保留上下文。分几层：
- 短期记忆：单会话内消息窗口
- 长期记忆：持久化事实/用户画像（向量库或键值存储）
- 程序记忆/工作记忆：Agent 循环内的中间状态

**本项目落点**：
- 会话级：`chat_sessions` / `chat_messages` 表（含 meta JSON 存引用）
- 检索级：QA 缓存表（`qa_cache`）——精确 hash + 语义相似度两级记忆，TTL + LRU
- Checkpointer：LangGraph 内存/PostgreSQL 检查点（图状态持久化，中断恢复）
- 可扩展：用户画像（偏好书籍）、跨会话事实库

**面试怎么答**：把"缓存"讲成"记忆系统的一个实现"就高级了——精确匹配（哈希）是短期记忆、语义匹配是联想记忆、TTL 是遗忘机制。再提 LangGraph checkpointer 支持任意中断点恢复，说明你懂图执行的可持久化。

## 4. Eval（评测）

**是什么**：LLM 应用必须有量化反馈，否则无法迭代。维度：回答准确率、引用正确率、幻觉率、延迟、成本、安全。方法：黄金集 + 关键词/LLM-as-judge 打分 + 人工抽检。

**本项目落点**：
- `eval/golden_set.json`：15 道黄金题（正向/陷阱/信息不足/全库四类）
- `eval/run_eval.py`：fake（链路冒烟）+ real（真实模型）双模式，走完整 API 链路含缓存
- 指标：answerable_accuracy、unanswerable_recognition、citation_coverage、avg_latency
- 实测：15/15 PASS，可答准确率 100%，不可答识别 100%，引用覆盖 100%，平均延迟 1.28s
- `tests/`：90 个单元/集成测试（回归防线）——eval 测"好不好"，tests 测"对不对"

**面试怎么答**：这是拉开差距的点。讲三层测试观：单测/集成测（对不对）→ 黄金评测集（好不好）→ 人工抽检（像不像人）。再讲评测集设计的坑：要故意放陷阱题（"王五喜欢李四吗"——防止模型被反向关系误导）和信息不足题（防幻觉，要求诚实答"不知道"）。

## 5. MCP（Model Context Protocol）

**是什么**：Anthropic 2024 年底开源的标准协议，统一"模型 ↔ 外部工具/数据源"的连接方式。类比：LLM 界的 USB-C——一次接入，到处可用。Server 暴露工具/资源/Prompt，Client（如 Claude Desktop、IDE）连接。

**本项目落点**：尚未使用。可扩展方向：把本项目封装成一个 MCP Server，暴露 `upload_book`、`ask_book`、`reindex` 等工具——任何支持 MCP 的客户端（Claude、Cursor、自建 Agent）都能直接调用你图书库的问答能力。

**面试怎么答**：能说清 MCP 解决什么（协议碎片化：每家模型各写各的工具集成）和三要素（tools/resources/prompts）。再表明理解边界：MCP 是"传输+发现"层，不做工具执行和鉴权——工具逻辑还是你自己写。

## 附：本项目技术栈 ↔ 概念映射

| 概念 | 项目里的对应物 |
|---|---|
| Function Calling | final_answer / request_clarification 双工具 |
| ReAct | 未用（可扩展为 ReAct agent） |
| Memory | qa_cache 双级缓存、chat_messages、LangGraph checkpointer |
| Eval | eval/ 黄金评测集 + run_eval.py + 90 tests |
| MCP | 未用（可封装为 MCP Server） |
