# 第 07 章 P6 CLI 与 P7 收尾

> 对应阶段：P6 CLI（ingest / ask）、P7 收尾（覆盖率、文档、git 提交）。

## 本章目标

给核心能力加一个不用开服务器的命令行入口，然后完成工程收尾：覆盖率检查、格式统一、README 接口文档、Conventional Commit。

## 1. P6：typer CLI（`cli.py`）

### 1.1 结构：工厂函数 + 依赖注入

```python
def create_cli(session_factory=None, vector_store=None, llm=None, embedding_model=None):
    app = typer.Typer(help='ReadingAssistant 命令行工具', no_args_is_help=True)

    @app.command('ingest')
    def ingest(book: Path = typer.Argument(..., help='电子书路径')):
        ...

    @app.command('ask')
    def ask(question: str = typer.Argument(...), document_id: int | None = typer.Option(None, '--document-id')):
        ...

    return app

app = create_cli()
if __name__ == '__main__':
    app()
```

**与 API 一致的模式**：`create_*` 工厂 + 可选依赖注入，测试传 mock，生产走默认。这样 CLI 和 API 共用同一套图和仓储，行为一致。

### 1.2 关键语法

- `typer.Argument(...)`：位置参数；`typer.Option(None, '--document-id')`：可选参数。
- `no_args_is_help=True`：不带参数时打印帮助而不是报错。
- 失败退出码：`raise typer.Exit(code=1)`（配合 `typer.echo(..., err=True)` 输出到 stderr）。
- `Path` 类型注解让 typer 自动做路径校验。

### 1.3 console script

```toml
[project.scripts]
readingassistant = "reading_assistant.cli:app"
```

安装后直接 `readingassistant ingest ...`，不必 `python -m`。

### 1.4 CLI 测试（CliRunner）

```python
from typer.testing import CliRunner

runner = CliRunner()
result = runner.invoke(_cli_app(env), ['ingest', str(book)])
assert result.exit_code == 0
assert '入库成功' in result.output
```

**技巧**：断言 `result.output` 的关键词而不是精确文本，减少脆断。

## 2. P7：覆盖率

加入 pytest-cov 并配置：

```toml
[tool.coverage.run]
source = ["reading_assistant"]       # 只统计项目代码，不含测试本身

[tool.coverage.report]
show_missing = true
skip_covered = true                  # 已全覆盖的文件不刷屏
```

```bash
.venv/bin/pytest -q --cov=reading_assistant --cov-report=term-missing
```

**如何看报告**：`Missing` 列是未覆盖的行号；像 `api/main.py`（uvicorn 入口）和 `logger_handler.py` 这类低覆盖属于正常（入口/工具模块），不必为凑数字硬写测试。本项目最终 74 个用例、约 94%。

## 3. P7：文档与状态同步

README 收尾三件事：

1. **移除 TODO 标注**：已实现/未实现要重新核对，别留「已实现却标 TODO」的错位。
2. **补充接口文档**：把每个端点的「方法 + 路径 + 请求/响应要点」列成表格。
3. **同步进度**：Roadmap 完成项打 ✅，开发计划每个阶段状态更新。

AGENTS.md 的目录结构、命令说明也要跟着改，否则新人会被旧文档误导。

## 4. P7：git 提交规范

### 4.1 Conventional Commits

```
feat: 完成 CLI 与项目收尾（P6/P7）

- CLI：新增 python -m reading_assistant.cli ingest/ask 命令
- 测试：74 个用例（覆盖率 94%）；加入 pytest-cov 与覆盖率配置
- 文档：README 补充 API 接口表、更新进度与 Roadmap
```

格式：`<type>: <描述>`（`feat` / `fix` / `refactor` / `docs` / `test` / `chore`），正文用列表说明「改了什么、怎么测的」。**一个提交只做一件逻辑变更**。

### 4.2 提交与推送

```bash
git add <文件列表>          # 只加本次变更的文件，别 git add -A 误收他人改动
git commit -m "feat: ..."
git push origin master
```

**注意**：工作区可能有别人（或你自己之前）未提交的改动，提交前先 `git status` 看清楚，只暂存属于本次提交的文件。

### 4.3 PR

创建 PR 需要 `gh` CLI 或 GitHub token。本仓库没有 `gh`，处理方式：推送后说明「PR 需安装 gh 或走网页版」，不强行造一个 PR。

## 5. 收尾检查清单

- [ ] `pytest -q` 全绿
- [ ] `pytest --cov=reading_assistant` 覆盖率达标并记录
- [ ] `ruff check src tests` 通过
- [ ] `ruff format --check src tests` 通过
- [ ] README / AGENTS / 开发计划状态一致，无残留 TODO
- [ ] Conventional Commit 描述变更与测试方式
- [ ] 工作区只留「有意不提交」的文件（他人改动、临时文件）

## 6. 后续可做的延伸（Roadmap）

- **Alembic**：数据库迁移工具，替代「删表重建」的开发期方案。
- **引用解析增强**：页码级来源、时间线结构化抽取。
- **本地 Embedding / LLM**：摆脱 API 依赖。
- **多租户与认证**：用户数据隔离。

到这里，你已经走完 ReadingAssistant 从零到可运行的完整开发链路。回头看这几章，真正重要的不是某个库的 API，而是贯穿始终的工程习惯：**测试先行、依赖注入、分层边界、契约对齐、诚实标注**。
