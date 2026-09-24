# 事故：`401 InvalidApiKey / API-key is blocked`（缓存节点）

> 2026-09-21 · 症状：问答时报 `ValueError: status_code: 401 / InvalidApiKey / API-key is blocked`，
> 出错节点 `cache_check`。
> 结论：**shell 里 export 了一个被轮换掉的旧 DashScope key，它盖掉了 `.env` 里的新 key。**
> 这是我 2026-09-20 那次配置修复的**回归** —— 修复本身是对的，但它暴露出一个既有的环境不一致。

## 1. 定位过程

| 步骤 | 结果 |
|---|---|
| 直接探测两个 provider | **都成功** —— 当前 shell 下 DashScope embedding 正常（1024 维）、DeepSeek 正常 |
| 日志定位 | `logs/intent_20260921.log` 里有 `InvalidApiKey / API-key is blocked`；是远端返回，不是本地异常 |
| 读**运行中进程**的环境 | 服务进程（pid 50452）持有的 `DASHSCOPE_API_KEY` = **长度 35 / 前缀 `sk-61a`** |
| 对照 `.env` | 生效值 = **长度 115 / 前缀 `sk-ws-`** |
| 对照 `.env` 的注释行 | 注释掉的旧值 = **长度 35 / 前缀 `sk-61a`** ← **指纹完全一致** |

**关键判据**：`/Users/aasing/.zshrc:7` 里 `export DASHSCOPE_API_KEY=…`，
其值的指纹与 `.env` 中**被注释掉的旧 key** 逐字节相同。

- 用户轮换 key 时把新值写进 `.env`，旧值留在注释里，但 **`.zshrc` 里的 export 没清**。
- 服务从那个终端启动 → 进程环境带着旧 key。

## 2. 为什么以前不炸，现在炸 —— 我的回归

优先级由 `model/factory.py` 的 `load_dotenv(...)` 决定：

| | 行为 | 结果 |
|---|---|---|
| 改之前 `load_dotenv(override=True)` | **`.env` 强写 `os.environ`**，.env 赢 | 旧 shell export 被静默压掉 → 一直正常 |
| 改之后 `load_dotenv()`（默认，进程环境优先） | 进程环境赢 | **shell 的旧 key 盖掉 .env 的新 key** → 401 |

2026-09-20 我改这一行的原因是：`override=True` 会让「命令行改配置」**静默失效**，
当时一次 `top_k` A/B 实验两组跑的是同一配置、指标逐字相同，差点把「无收益」当结论报出去。

**所以两个方向都会静默失败**：

- `override=True`：`TOP_K=16 uv run …` 悄悄不生效（我踩过）；
- `override=False`：shell 里的旧密钥悄悄生效（本次）。

## 3. 修复：保留标准优先级，但让不一致**变响**

没有回退到 `override=True`（那会把我修好的 bug 放回去），而是加护栏：

`config.py::warn_if_process_env_shadows_dotenv()`，在 `get_settings()` 首次加载时跑一次，
比较 `os.environ` 与 `.env`：

- 同名不同值 → 告警，并说明**当前生效的是进程环境**的值；
- 进程环境的值 **== `.env` 里被注释掉的旧值** → 额外高亮（这正是本次事故的形态）；
- 敏感键不打印内容，只报「不一致」；
- 仅告警、**不改优先级**（覆盖能力必须保留，评测与 CI 要用）。

实测告警输出：

```
配置[环境覆盖] DASHSCOPE_API_KEY 在进程环境与 .env 中不一致（<已隐藏>）——当前生效的是**进程环境**的值。
配置[环境覆盖] DASHSCOPE_API_KEY 在进程环境里的值**等于 .env 中被注释掉的旧值** ——
              这通常是 shell 里 export 了轮换前的旧值，且它会盖掉 .env 里的新值。
              请 unset DASHSCOPE_API_KEY（并检查 shell 配置）后重启。
```

**为什么不做「secret 一律以 .env 为准」**：那会破坏标准的覆盖能力（CI 注入密钥就靠它），
而且把「哪个源可信」硬编码进代码。真正的问题是**不一致没有被看见**，不是选错了源。

## 4. 验证

| 项 | 结果 |
|---|---|
| 护栏能咬住本次事故（注入旧 key 跑子进程） | ✅ 两条告警都触发，生效 key 长度为 35（复现） |
| 干净环境（`env -u DASHSCOPE_API_KEY`） | ✅ 生效 key 长度 115，embedding 成功、检索命中 7 条 |
| 回归测试 `tests/test_settings_precedence.py` | 7 条全过（新增 5 条：不一致告警 / 识别注释旧值 / 只告警一次 / 缺失 .env 静默） |
| 全量测试 | **489 passed** |
| ruff | All checks passed |

## 5. 用户侧的实际修复

根因在仓库之外，需要用户动手（我只验证了方案，没改用户的 shell 配置）：

```bash
# 删除或注释 ~/.zshrc 第 7 行的 DASHSCOPE_API_KEY export，然后新开终端重启服务
sed -i '' '/^export DASHSCOPE_API_KEY=/s/^/#/' ~/.zshrc   # 或手工编辑
unset DASHSCOPE_API_KEY                                   # 当前终端立即生效
```

**留下的守卫**：即使忘了改，启动时也会看到上面那两条告警，不必再挖到「读进程 environ」。

## 5.5 第二次排查：同一个 401，**不同的根因**

用户再次报同一症状后重新取证，发现 `.env` 里的 key 也坏了：

```
DASHSCOPE_API_KEY=西游记中师徒四人遇到的第一个妖精是谁？它有什么技能？最后是怎么解决掉它的？
```

37 字符、首字符 `U+897F`、非 ASCII —— 是**粘贴事故**（一句问题文本进了密钥字段），不是密钥。
更糟的是：那个 115 字符的有效 key（轮换后的新值）**两份都没了** ——
`.env` 的生效行被这句覆盖，而注释行留下的恰是**已 block 的旧 35 字符 key**。
全盘找过 `.env*` 备份与 shell 配置/history，**没有可恢复的副本**。

**症状相同 ≠ 根因相同** —— 第一次是「shell 旧 key 盖掉 .env 的好 key」，
第二次是「.env 本身被写坏」。这次如果套用上次结论就会误诊。

处理：把该行归置为带说明的空值（保留证据、写明还要清 `~/.zshrc`），**没有填任何假值**；
并把有效 key 的恢复交回用户。

### 新增护栏②：密钥值形状校验

`config.py::warn_if_secret_values_look_malformed()` —— 判据刻意很粗：
**非空 + 全 ASCII + 无空白**。

- 不用「`sk-` 开头」之类：密钥格式随供应商轮换（本项目就从 35 字符 `sk-` 换到 115 字符 `sk-ws-`），
  但**任何密钥都不会含中文或空格** —— 这条永不误伤，却挡得住粘贴事故；
- 同时检查 `.env` 与进程环境两个来源；非密钥键（`DATABASE_URL`、`BM25_TOKENIZER`）不检查；
- 实测：能抓住本次形态（报 `.env:DASHSCOPE_API_KEY 不像密钥`），且不误伤同文件里的正常 key。

### 两道护栏（都在 `get_settings()` 首次加载时跑一次，**只告警、不改行为**）

| 护栏 | 管什么 |
|---|---|
| `warn_if_secret_values_look_malformed()` | **值形状** —— 粘错、空值 |
| `warn_if_process_env_shadows_dotenv()` | **来源一致性** —— 进程环境 vs `.env`（含「等于注释掉的旧值」）|

## 6. 未覆盖的情形（如实记录）

护栏只检查「**两边都定义了同名键但值不同**」。
如果某个密钥**只在进程环境里存在、`.env` 里完全没有**，则不会被告警 ——
那种情况下进程环境是唯一来源，能否算「陈旧」无从比较。这次不是该情形，未做处理。

## 7. 教训

这是本项目第 N 次**静默失效**（前科：重复上传把文档打回 `indexing`、前端未 rebuild、
租期时区错位、判官词表漏词、`min_score` 误杀正确段落）。

共同形状是：**两个来源都在工作，只是选错了那一个，而且没有任何提示。**
所以修法不是「选对来源」，而是**让分歧可见** —— 与 `_warn_if_frontend_build_is_stale()`
是同一思路。
