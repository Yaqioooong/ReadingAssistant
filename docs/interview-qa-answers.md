# 面试问答弹药库（对练存档）

> 来源：Alma 对练会话（2026-09-04）。每题 = 原题 + 一句话定调 + 照念版回答 + 加分细节 + 代码锚点。
> 当前已答题号：1、4、7、8、9、10、12、13 + 追加预演（30秒版/重做设计/Chroma故障/全文检索/检索评估/提升召回率）。
> 待补题号：6+（用户清单空档，后续追加）。

---

## 题 1：项目用了 PG 和 ChromaDB，如何防双写？

**一句话定调**：没做分布式事务（PG 和 Chroma 无法同事务），方案是「PG 当唯一决策者 + 两层去重闸门 + Chroma 幂等写入」，重复上传、并发请求、失败重试都不会产生双份数据。

**照念版回答**：
- 入库前 PG 侧两层哈希短路（DocumentService.add_book）：第一层原始文件字节 `file_hash`，同文件重传直接返回已有 document_id，解析都跳过；第二层归一化 `content_hash`（折叠空白），同书不同格式也拦得住。
- 两列在 PG 上有 `unique=True` 唯一索引兜底：并发请求同时通过预检查、同时 INSERT，`insert_document` catch `IntegrityError` → rollback → 查回已有记录返回 inserted=False。唯一索引是并发下的最后防线。
- 入库是 LangGraph：`add_book` 判定 duplicate 且非 force 时条件边直接走 END，根本不进入 `chunk_and_index`——重复场景 Chroma 连写的机会都没有（上游掐死第二次触发，而不是两存储间协调）。
- 真正常写 Chroma 的场景（新书入库 / reindex force）：chunk id 确定性 `doc{document_id}-{chunk.index}` + Chroma `upsert()`——同 id 重复写是覆盖不是追加，重试/重建不会越积越多。
- 诚实交代：**不是强一致**。窗口期在 `chunk_and_index`：PG 已提交但 Chroma 写崩 → 文档停在 `indexing`/`failed`，恢复靠 `index_status` 状态机 + `reindex(force=True)` 重跑，upsert 幂等收敛。用「最终一致 + 可重入」替代 2PC，理由是单机部署 + 体量下不值得上消息中间件；业务上短暂不一致可接受（索引失败只影响检索，不影响文档元数据）。
- 自曝取舍（加分）：文档已存在但状态 `failed` 时，重传同文件会被 duplicate 短路挡住，恢复路径只有 reindex force——主动讲这个说明踩过坑。

**代码锚点**：storage/service.py、storage/repositories.py（insert_document）、storage/models.py（unique 约束）、graph/ingest.py（route_after_add）、storage/vector_store.py（upsert）、api/routes/documents.py（reindex）。

---

## 追加 1：30 秒电梯版（照着念约 30~35 秒）

> 我们双写 PG 和 ChromaDB，但没做分布式事务——我用的是「PG 决策 + 幂等写入」。入库先算两重哈希：文件字节哈希和归一化内容哈希，PG 上有唯一索引兜底，重复上传或并发同时插入，只有一个能成功，另一个拿到已存在的记录直接短路。短路之后根本不会走到向量库，所以重复场景下 Chroma 连写的机会都没有。真正要写 Chroma 的场景，chunk ID 是确定性的，写入用 upsert——重试、重建索引都是覆盖不是追加，天然不产生双份。PG 和 Chroma 之间没有强一致，我靠文档的 index 状态机 + reindex 重试收敛，接受短暂不一致。简单说：上游靠唯一键挡掉重复，下游靠幂等键承受重放。

---

## 追加 2：「如果让你重做，你会怎么设计？」

**三步答法**：
1. 先承认现状取舍：单机部署、数据量可控下方案是对的；本质是最终一致 + 可重入，代价是 PG 提交成功但 Chroma 写崩的窗口期，需 reindex 手动收敛。
2. 给升级方向（主推 Outbox）：把 Chroma 降级成「从 PG 派生、可丢弃的索引」。写文档元数据时，在**同一 PG 事务**里落一条 outbox 事件；worker 消费事件写 Chroma，写完标记 done。Chroma 写入变 at-least-once，配合确定性 chunk ID + upsert，重放一万次也不双写。PG 是唯一 truth，一致性从"两存储协调"变"一队列消费"。
3. 补一句对比收尾：另一思路是彻底不做增量一致性——chunk 当缓存，启动对账（PG 说 indexed 但 Chroma 没有就整文档重建），确定性分块保证重建后 chunk id 一致。要求秒级可用选 outbox，能容忍重建窗口选懒重建。
4. 加分反问：您更关心重试语义还是数据一致性？

---

## 追加 3：「Chroma 挂了，你的查询怎么办？」

**关键事实先行**：chunk 文本只存在 Chroma 里，PG 只存元数据和 chunk_count——"降级到 PG 全文检索"说不通（没存 chunk），别凭空说。

**照念版回答**：
- 第一层保障：QA 图第一站 `cache_check`——PG 的 qa_cache 缓存层，精确命中靠 question_hash，语义命中靠存的 question_embedding 做余弦匹配，拦截大量重复/相似问题，这些请求根本不碰 Chroma。
- 未命中缓存、走到 retrieve 的请求，Chroma 挂了目前会抛错——已知短板，不回避。
- 架构上不慌的原因：Chroma 是**可重建的派生索引**。PG 存着每本书的 file_path，分块参数固定、chunk id 确定性、写入幂等 upsert——挂了走 reindex 重解析重建，数据不丢，最坏损失是重建窗口内检索不可用。
- 若要补优雅降级：加开关——Chroma 不可用时未命中缓存的问题返回"检索暂时不可用"而非 500，后台触发重建。
- 加分细节：重建是文档级、按 index_status 状态机推进（运维路径也想好了）。

---

## 追加 4：「为什么不用 PG/MySQL 全文检索，非要多上向量库？」

**照念版回答**：
- 业务形态不匹配：读后问答，用户问的是自然语言问题，措辞与原文几乎不重叠（问"谁背叛了主角"，原文是"他在宴席上出卖了盟友"）。全文检索是字面匹配，召回不了改写式提问；向量检索做语义召回，配合相似度阈值 + document_id 过滤。
- 中文场景：PG 默认分词器对中文基本逐字切，要上 zhparser 等插件，效果不确定——与其折腾不如让 embedding 模型一并解决语义问题。
- 检索需求是"相似度排序取 top-k"，不需要布尔逻辑/词频权重这种全文索引核心能力。
- **漂亮的钩子**：如果重做可能直接上 pgvector——向量列塞进 PG，Chroma 消失，"双写一致性"问题从根上不存在，变成单库单写；十万级向量内性能够。等量级上来再拆专用向量库。取舍标准：能少一个存储就少一个，一致性永远比性能先付钱。

---

## 追加 5：「检索质量怎么评估？min_score 为什么定 0.45？」

**分层关键**：评估拆两层——检索层（召回准不准）与端到端层（回答好不好）。现有体系主要打在端到端，检索层是间接衡量，别混着说。

**照念版回答**：
- 端到端层有完整体系：eval/golden_set.json 15 道黄金题、四类（正向/陷阱/信息不足/全库）。run_eval.py 的 real 模式走生产 PG + 真实模型 + Chroma 全链路。指标四件套：answerable_accuracy、unanswerable_recognition（防幻觉）、citation_coverage、avg_latency。实测 15/15，可答 100%、不可答识别 100%、引用覆盖 100%、平均延迟 1.28s。
- 三层测试观：90 个单测/集成测管"对不对"，黄金评测集管"好不好"，人工抽检管"像不像人"。
- 检索层诚实说：目前无独立 recall@k / MRR，靠 citation_coverage 间接反映；改进项是把 chunk 命中单独做检索评测集。
- min_score 0.45：Chroma 返回距离，转余弦相似度 1/(1+d)，0.45 以下是低相关噪音。经验初值，接受它是因为有 HITL 兜底——阈值略高致检索为空时 judge 转人工澄清而非硬答；阈值过低才危险（拿无关 chunk 硬答 = 幻觉）。宁高勿低 + HITL 兜底。真要标定应对 embedding 模型画 P-R 曲线选点，不同模型余弦分布差异大。
- 加分：陷阱题专门防"模型只看关键词相关性就答"（反向关系题，如"王五喜欢李四吗"）；eval 每次清库重传文档，防缓存污染评测链路。

**代码锚点**：eval/run_eval.py、eval/golden_set.json、rag/retriever.py（min_score/top_k）、config.py。

---

## 题 4：RAG 混合检索召回如何优化？如何评测？

**一句话定调**：先评测后优化（没有检索评测集的优化是拍脑袋）；优化分三步——加 BM25 稀疏路、RRF 融合、视效果加重排。

**照念版回答**：
- 单路稠密向量的弱点写在黄金集里：`santi-01`"照片背面写了什么"（精确引语，期望"不要回答"）、`shaobai-03`"哪一年失去权力"（数字专名）——这些正是 BM25 的强势区。
- 第一步加 BM25 稀疏召回路，兜专名/引语/数字；中文处理分词（字符级或 jieba），用评测说话。
- 第二步融合用 RRF（Reciprocal Rank Fusion）不用分数加权——BM25 与 cosine 分数分布不可比，加权要调两个权重；RRF 只看排名 1/(k+rank)，k=60，零调参、对尺度不敏感。
- 第三步看评测再决定加重排：融合后 top-50 用 reranker 精排回 top-6。**重排是奢侈品，先不加**。
- 评测拆两层：检索层新增——把黄金集升级成带 gold chunk 的检索评测集；弱标注来源 = 端到端跑对的 case 里 citations 的 chunk_id（正确回答引用的 chunk 即正例）。指标 Recall@k / MRR / Hit Rate。端到端层用现有体系验收：同 golden set 跑 real，对比 answerable_accuracy + citation_coverage，**按题型分层看**（混合检索主要救引语/实体题，总量可能不动）；固定 LLM，消融"单稠密 vs 稠密+BM25 vs +rerank"，控制变量。eval 清库重传天然防缓存污染。
- 加分：BM25 值不值得加——直接点黄金集那两题"这类精确匹配题是单路向量的已知盲区"。

---

## 追加 6（提升召回率）：如何提升 RAG 召回率？

**一句话定调**：先定义 recall = 相关 chunk 进没进候选集（Recall@k），再动手。代码里真实天花板：retrieve 直接 top_k=6 + 0.45 阈值 = 候选窗口只有 6，相关 chunk 排第 7 永远没戏。

**照念版回答**（按 ROI 排序）：
- 第零步：把召回率变可测——用现有跑对 case 的 citations chunk_id 做弱标注构建检索评测集，指标 Recall@k / Hit Rate；所有优化同题消融、固定 LLM。
- 第一步改架构缺陷：候选池太小。改两段式——第一段无阈值放宽到 top 50~100 取候选，第二段再过滤精排压回 top 6。召回和精度别用一个参数同时管。
- 第二步索引侧：分块策略（800 定长切语义跨块时答案被腰斩 → parent-child 小块检索大块喂模型）；metadata 太薄（只有章节 → 把标题/上下文摘要拼进索引文本，contextual retrieval 思路）；书名/章节级单独一层召回（很多问题是"哪个文档里有"）。
- 第三步召回通道融合：BM25 稀疏路兜专名/引语/数字 + RRF。**Rerank 不算召回手段**——它提精度，候选池里没有，rerank 也变不出来。
- 第四步查询侧（成本最高放最后）：查询改写、多路 query（RAG-Fusion）、HyDE——等前三步做完、数据说不够再上。每一步拿 Recall@k 说话。

---

## 题 7：上下文管理如何做？上下文如何保证一致性？

**一句话定调**：三层上下文别混讲——会话层（chat_sessions/chat_messages）、单轮流水线层（LangGraph state + checkpointer）、跨会话记忆层（qa_cache 双级缓存）。且每轮 thread_id 都是新 uuid、LLM prompt 不塞历史——"上下文"不等于"对话历史进 prompt"。

**照念版回答**：
- 会话上下文：chat_sessions / chat_messages 落库，引用存 meta JSON，界面回放与历史都从 PG 读。刻意不做"历史全塞 prompt"的多轮：每轮问答是单轮 RAG，只带当前问题检索；读书问答每问独立，塞历史费 token 且易被上一轮带偏。跨轮信息走显式通道：HITL 澄清时补充说明作为 clarification 流入下一轮，参与检索、回答与缓存键。
- 单轮执行上下文：LangGraph 状态机，question/clarification/document_id/chunks 在 state 流转；checkpointer 支持中断恢复（生产可换 Postgres 版）；每次调用新 thread 隔离，状态不跨轮污染——防串扰设计。
- 跨会话记忆：qa_cache。精确层用归一化问题哈希；语义层用存的 question_embedding 在 PG 内余弦匹配，命中复用答案，跳过 embedding + LLM。
- 一致性四机制：①缓存与文档版本绑定（每条目存 content_hash，文档 reindex 后哈希变、旧缓存自动失效，绝不"文档改了回答还是旧版"）；②输入归一化（NFKC/小写/去标点后哈希，同问题不同写法同键）；③无效回答不进记忆（"信息不足"式回答识别删除，防坏答案污染）；④写 PG 全走 session_scope 事务原子。
- 遗忘机制：TTL + LRU（超出 cache_max_entries 按 last_hit_at 淘汰），像人的遗忘，防记忆无限膨胀。
- 加分应对：缓存跨会话共享——单用户本地产品，同书同问给同答案是特性；多租户就在键上加 tenant_id。多轮演进：增量摘要 + 最近 N 轮，检索永远基于当前问题，引用永远带 chunk_id 可溯源。checkpointer 价值：LLM 超时中断不用重跑解析检索。

**代码锚点**：api/routes/sessions.py（每问新 thread_id、clarification 传入）、graph/qa.py（cache_check/content_hash 绑定/无效缓存删除）、graph/checkpointer.py、retriever.py（L1 embedding 缓存）。

---

## 题 8：项目中有用到 hook 机制吗？

**一句话定调**：没有自研业务 hook 框架，但 hook 思想在框架层（生命周期/中间件/依赖注入）和业务层（HITL 事件流）都有体现，诚实分层讲 + 秀依赖注入设计。

**照念版回答**：
- 框架层：FastAPI `lifespan` 钩子（启动时生产模式自动 init_db 建表、退出打日志）；HTTP 中间件 access_log（每请求记 method/path/耗时）——后续加鉴权/限流/审计都挂这层，不改业务代码；`dependency_overrides` + create_app 注入参数（session_factory/vector_store/llm/embedding_model/upload_dir），测试与 eval fake 模式靠它换 mock——依赖倒置的扩展点，比 hook 更干净、组件可插拔。
- 业务层：最接近事件钩子的是 HITL——LLM 判定信息不足创建澄清任务（awaiting），用户提交后消费（approved），人在回路的异步事件流，任务表当事件队列。LangGraph 条件边是"决策钩子"（duplicate 短路、cache 命中跳检索）。
- 演进：真要加业务 hook，挂 LangGraph callback 做链路追踪，或入库完成后发 post-ingest 事件触发 QA 预缓存；当前数据量下同步链路够用，没到事件总线复杂度。
- 加分应对：中间件管请求横切面、hook 偏业务事件通知（一句分清）；不用 LangChain callback/LangSmith 是因为结构化日志够用，需要 token 级成本分析再上（知道有现成轮子、按需取舍）；收尾反问"您指的 hook 是运行时事件扩展还是外部插件接入？"

**代码锚点**：api/app.py（lifespan/middleware/dependency_overrides）、api/routes/hitl.py、graph/qa.py（条件边）。

---

## 题 9：多 agent 如何判断并行还是串行？判错了如何兜底？

**一句话定调**：依赖决定串行（产出-消费链）；并行是奢侈品（只给廉价操作）；工程原则——默认串行、证据充分才并行，让判错只往"慢"倒不往"错"倒。

**照念版回答**：
- 背景：experiments/multi_agent.py——supervisor 工具调用规划（doc_scope + search_query）→ retrieval_worker → synthesis_worker，全串行是刻意为之（plan→chunks→answer 严格数据依赖，串行是唯一正确解）。
- 判断四规则：①依赖关系——产出-消费链必须串行，无依赖的独立子任务才候选并行；②信息完整性——下游需要全部支路结果时支路可并行但合流点必须串行闸门（跨文档问题如黄金集 cross-01：每文档一路检索 worker 并行，全回再合流给 synthesis）；③共享状态——并行支路只写独立 state 字段，同 key 即竞态；④成本——检索可并行，LLM 决策尽量少并行。
- 兜底先说结论：该并行的判成串行只是延迟变高、正确性无损；该串行的判成并行才是灾难（缺依赖/竞态/上下文不一致）。所以默认串行、证据充分才并行。
- 兜底四层：①静态图兜底——LangGraph 的边即依赖声明，编译期保证拓扑，错只在动态规划里可能发生；②validator 兜底——LLM 运行时规划任务图，输出先过校验（依赖完整/无环/每路输入齐），不过回退默认串行模板；③运行时降级——并行支路超时失败重试一次，再不行标记"结果缺失"交给下游，让 synthesis 明确知道缺料并诚实回答，不拿残缺结果硬编；④合流质量闸门——复用 judge：并行合流后信息不足 → 转 HITL 或降阈值串行深检索，末端兜住前面所有判错。
- 加分：实验版 multi_agent 没接 judge 兜底、生产 QA 图有（实验暴露问题，生产解决问题）；能单 agent 流水线解决的别上多 agent，收益是职责隔离和并行、代价是状态传递/token/调试，生产问答就是单流水线，multi-agent 留在实验验证规划能力。

**代码锚点**：experiments/multi_agent.py、graph/qa.py（judge/HITL 兜底范式）。

---

## 题 10：RAG 文档切分如何处理过长/过短？图片、表格如何处理？

**一句话定调**：前一半代码有真实答案（递归切分 + 分隔符分级 + 空串兜底 + 空块过滤）；后一半是空白（图片被静默丢弃、表格被压平）——现状别装，演进分优先级，别一上来喊多模态。

**照念版回答**：
- 现状：章节感知切分，每章 RecursiveCharacterTextSplitter，chunk_size 800 / overlap 100。分隔符分级递归（段落→句号问号→字符），末级空串保证任何超长块都被兜底切完；纯空白块过滤，空章节不产生 chunk。
- 过长：超 800 切不完 = 分隔符失效（超长段落/代码/表格行）。硬切会从句子中间腰斩语义，embedding 被长文本平均化稀释。处理：overlap 100 保上下文衔接；把不可切分单元（表格/代码/公式）识别出来走结构化通道，不混进普通文本被切碎。进阶：parent-child 多粒度——小块检索、父块喂模型。
- 过短：短块 embedding 语义弱、检索噪音多。现在只做了空块过滤，没做最短长度合并——已知缺口。改进：min_chunk_size + 就近并入下一块；标题这类结构性短文本不单独成 chunk，作为 metadata/上下文前缀注入所属正文块（自解释 chunk，兼解检索上下文问题）。
- 图片：诚实——纯文本链路，epub/pdf 解析时图片被丢弃。演进按成本排序：①OCR（扫描版/有文字的图转文本）；②语义图用 VLM 生成描述，让"图"以文本参与检索，检索到绑定的块时把原图/描述交给模型，引用可溯源；③多模态 embedding 最重，前两步不够再评估。
- 表格：现在被压平成文本流、结构全丢。正确做法：解析器结构化提取表格 → 转 Markdown 表或键值行文本再入库，表格声明为不可切分单元防从单元格中间切断；检索到表格块以 Markdown 保真呈现。再进一步：解析器输出**带类型的块流**（标题/段落/表格/图），切分按块类型路由，而非一刀切纯文本。
- 加分应对：800 怎么定——对齐 embedding token 窗口（中文约 700~900 字）+ 答案定位粒度，但该拿检索评测集标定，Recall@k 说话；overlap 100——够覆盖句子跨块余量即可，overlap 越大冗余噪音越大（精度 vs 冗余权衡）。

**代码锚点**：rag/chunking.py、config.py（chunk 参数）、parsers/（epub BeautifulSoup get_text / pdf pypdf extract_text，图片表格现状）、storage/vector_store.py。

---

## 题 12：记忆机制如何做？

**一句话定调**：把"缓存"讲成"记忆系统的实现"，分四层：会话短期记忆、工作记忆（图状态）、问答长期记忆（双级缓存核心）、检索记忆（L1 embedding 缓存）。

**照念版回答**：
- 会话短期记忆：chat_sessions / chat_messages 落库，引用存 meta JSON。服务回放与追溯；不塞 prompt 做多轮，跨轮走显式通道（clarification）。短期记忆管"说过什么"。
- 工作记忆：LangGraph state + checkpointer（内存版默认 / 生产 Postgres），解决"流水线执行到一半状态不能丢"——LLM 超时中断不用重头检索。
- 问答长期记忆（核心，双级缓存）：精确级——归一化问题 + 澄清 + 文档范围拼哈希，命中复用答案跳过 embedding + LLM；语义级——条目存 question_embedding，未精确命中做余弦匹配，相似问题召回旧答案（联想记忆）。命中更新 hit_count / last_hit_at。
- 检索记忆：Retriever 内 L1 embedding 缓存（归一化问题哈希为键，LRU）——同问题反复问不重复调付费 embedding API。
- 记忆管理三配套：①遗忘机制——TTL 过期 + LRU 淘汰（超 cache_max_entries 按 last_hit_at 清最久未用）；②记忆一致性——条目绑定文档 content_hash，文档重入库内容变旧答案自动失效，记忆不落后于文档版本；③坏记忆清理——"信息不足"式回答识别删除，不让无效答案污染记忆。
- 加分应对：语义匹配快——候选先按 content_hash + document_id 缩范围（同版本内最多 200 条）再余弦，非全库扫；记忆串扰——单用户本地产品跨会话共享是特性，多租户加 tenant_id；演进——用户画像记忆（读书记录/偏好/跨会话事实，隐式抽取 + 显式标记双通道）、长会话压缩成摘要沉淀长期记忆。
- 高级收尾：记忆的本质是"第二次遇到同一问题不用重新思考"——命中缓存的那次请求延迟趋近于零（整条 LLM 链路被跳过），把记忆价值讲成产品指标。

**代码锚点**：storage/models.py（qa_cache 表结构）、graph/qa.py（cache_check/save/prune/_looks_like_no_info）、rag/retriever.py（embed L1 LRU）、api/routes/sessions.py。

---

## 题 13：多 agent 协作机制是怎样的？

**一句话定调**：experiments/multi_agent.py = supervisor-worker 流水线 + 黑板协作；五要素讲清——角色、任务分配、通信、控制、终止。注意：实验代码，生产是单流水线状态机，两层都讲姿态最稳。

**照念版回答**：
- 角色分工：supervisor 决策者（不干活，只规划：全库还是指定书、用什么 query，经 make_plan 工具调用产出结构化计划 JSON，含 doc_scope/search_query/note）；retrieval_worker 只消费计划去检索、无自主决策；synthesis_worker 拿结果总结带引用。决策与执行分离。
- 通信机制：agent 间不对话，靠**共享黑板**——中间产物写 MultiAgentState（plan/chunks/answer/citations 为 TypedDict 字段），节点读写各自字段。**结构化数据优于自然语言**：便宜、可校验、可打日志；JSON 计划天然可审计。
- 控制流：固定有向边 START → supervisor → retrieval_worker → synthesis_worker → END，一次编排、无循环、无自主分支。worker 无 ReAct 循环不会失控——**图拓扑即协作契约**。
- 任务分配：一次性分派——supervisor 入口把任务参数算清，中间不反复请示。检索计划一次定稿成本最低；探索式任务才值得 supervisor 介入循环。
- 终止条件：流水线跑完自然终止（synthesis 产出 answer + citations 即结束），无"聊到达成共识"式不确定性终止。
- 边界：实验验证 supervisor 规划能力；生产用单流水线 + HITL 人工闸门。多 agent 收益是角色隔离与未来并行扩展，代价是状态传递复杂度和调试成本——生产求稳、实验求新。
- 加分应对：为什么不直接发消息——消息传递适合去中心化拓扑，我的星型拓扑共享状态更简单；做 peer 对等协商（双 agent 辩论验答案）才引入显式消息通道。黑板写冲突——并行支路只写独立字段、合流点串行。supervisor 怎么不跑偏——工具 schema 即决策边界，只给选 doc_scope 和 query 的空间，约束即可靠。
- 最高级总结：多 agent 协作本质 = 把复杂任务拆成有依赖关系的子任务，用图拓扑固定依赖、用共享状态传递中间产物、用结构化工具约束决策——不是让一堆 agent 自由聊天。

**代码锚点**：experiments/multi_agent.py。

---

## 题 15：上下文膨胀，模型注意力和工具准确率下降，如何保障成功率？

**一句话定调**：先亮预防——单轮 RAG、prompt 不塞历史，从源头免疫大半；再讲治理（结构化排序/压缩/检索代替记忆/缓存跳过）、兜底（工具 fallback 链 + HITL）、评测（黄金集当成功率仪表盘）。四段叙事：预防 → 治理 → 兜底 → 评测。

**照念版回答**：
- 预防——上下文最小化：prompt 只含当前问题 + 检索 top-6 块 + 澄清，历史落库但不进 prompt，跨轮走显式澄清通道（架构选择不是运气）。工具双工具设计（final_answer / request_clarification），schema 即决策边界，关键信息走结构化参数不靠模型在长文本里"回忆"。
- 治理：①结构化和排序——检索块按相关度降序、最相关放最前（模型有 lost-in-the-middle：长上下文头尾注意力强、中间弱，放头部最便宜）；②压缩——多轮做滑动窗口 + 增量摘要，更进一步"检索代替记忆"（历史变可检索存储，需要时查出来，不硬塞）；③缓存跳过——命中 QA 缓存整条 LLM 链路不走，膨胀风险窗口归零。
- 兜底：bind_tools 异常回退纯文本 invoke → 纯文本命中"信息不足"措辞转 HITL；judge 对空结果兜底转 HITL——工具失败不静默给错答案，最差请用户补充信息。
- 评测：黄金集四类题型 + 四指标 = 成功率仪表盘；任何上下文策略改动（窗口/摘要/检索块数）跑同一套回归、分题型看掉没掉。改上下文 = 改模型行为，必须数据验收。
- 加分应对：top-6 太少答不出是 recall 问题不是膨胀问题（召回不够有 HITL 兜底，膨胀是精度问题，分开治理）；摘要丢细节——摘要只用于会话记忆，事实永远回原文查证（引用/数字不靠摘要）；怎么判断膨胀致败——评测按输入长度分层统计准确率，画"准确率 × 上下文长度"曲线，超拐点就压缩。

## 题 16：用户请求过来，agent 执行的全链路，从请求到回答

**一句话定调**：走查题 = 把一次提问讲成地图：入口 → LangGraph 状态机各节点（cache_check/retrieve/judge/answer/record + HITL 分支）→ 每次落库 → 返回。分支别漏：缓存命中跳过整条 LLM 链路、judge 踢去 HITL、answer 工具调用三层兜底。

**照念版走查（约 90~120 秒）**：
- 入口：POST /api/sessions/{id}/messages，body = question + 可选 document_ids + 可选 clarification。先验会话存在（404）；build_qa_graph 注入 session_factory/vector_store/llm/embedding_model；每请求新图实例 + 新 thread_id（请求间状态隔离）。
- cache_check：问题归一化（NFKC/小写/去标点）拼澄清拼文档范围 → SHA-256 缓存键；取目标文档 content_hash。两级查缓存：①精确查键命中直接复用旧答案；②miss 进语义查——完整问题 embedding（先查 L1 缓存省 API）→ 同 content_hash + 文档范围内拉 ≤200 候选 → 余弦 ≥ 阈值命中。命中更新 hit_count/last_hit_at，处理 TTL 过期与无效回答删除。**命中直接跳 record，retrieve/LLM 全不走 → 命中延迟趋近 0**。
- retrieve：问题（+澄清）embedding → Chroma 查询（document_id 过滤）→ top-6 → 距离转相似度 → ≥0.45 过滤 → chunks 带章节元数据写回 state。
- judge：无 chunks 且无澄清 → 信息不足转 create_hitl；有料 → answer。
- create_hitl（分支）：建 awaiting 澄清任务落库；写 needs_clarification 占位缓存（同问再命中 HITL 不重复烧 LLM）；prune 超限缓存。返回 hitl_task_id → UI 弹澄清 → 用户补充后带 clarification 重发（闭环第二轮：澄清进检索与 prompt，judge 有澄清不再判不足）。
- answer：prompt = 问题 + 检索块 + 澄清。LLM 绑双工具 final_answer / request_clarification：final_answer → 用 args.answer；request_clarification → 没澄清转 HITL、有澄清还不足则 missing_info 当回答；无工具调用 → 纯文本 fallback + "信息不足"措辞检测转 HITL；bind_tools 异常 → 普通 invoke fallback。引用从 chunks 生成（chunk_id/章节/excerpt 120 字）。未命中过缓存则写 qa_cache（答案+引用+question_embedding+content_hash 版本绑定）+ prune。
- record：user + assistant 消息（meta 存 citations）写 chat_messages，事务提交。返回 {answer, citations, needs_clarification, hitl_task_id}；access_log 记耗时。
- 加分应对：最慢 = answer 节点 LLM（1.28s 大头，缓存命中≈0，命中率直接决定体验）；失败 = 异常上抛 + 事务回滚 + LLM fallback 链，最差 HITL 不是错答案；状态三处 = PG（消息/缓存/HITL）+ Chroma（向量）+ LangGraph state（线程内）；两轮不串 = 每问新 thread + 无历史注入，跨轮唯一通道是显式 clarification。

**代码锚点**：api/routes/sessions.py（ask_question）、graph/qa.py（QAState 各节点/路由）、graph/checkpointer.py、storage/repositories.py（缓存与 HITL 仓库）。

## 题 17：agent 执行中出现错误，如何容错回复、如何持久化数据？

**一句话定调**：容错与持久化是一枚硬币两面——能回滚的回滚（事务）、不能回滚的落状态（状态机）、状态能恢复的重试（幂等）、恢复不了的给用户明确兜底（不伪装成回答）。

**照念版回答**：
- 错误分类：①输入错误（会话/文档不存在、格式不支持、文件损坏加密）→ 明确 4xx/422；②执行错误（LLM 超时异常、embedding 失败、向量库故障）→ 容错主战场；③一致性错误（并发冲突）→ 唯一索引兜底。
- 容错四层：
  ①事务边界——写 PG 走 session_scope（成功提交/异常回滚/finally 关闭），节点级独立事务，部分失败不污染；API 请求级会话同模式。
  ②节点级 fallback——answer 对 LLM 三层防护：bind_tools 异常回退普通 invoke；无工具调用回退纯文本；纯文本命中"信息不足"措辞转 HITL。原则：宁可请用户补充信息，不给编的答案（与评测"信息不足"题型防幻觉一脉相承）。
  ③状态机容错——入库链路 index_status：pending/indexing/indexed/failed；chunk_and_index 异常先标 failed 落库再抛错 → 失败可观测可恢复（reindex force 重跑 + Chroma upsert 幂等，重放不双写）。
  ④幂等兜底——并发撞唯一索引 catch IntegrityError → 回滚 → 返回已存在记录。
- 持久化边界：
  ①半成品不落——record 只在拿到有效 answer 才写 assistant 消息，失败轮次只留用户消息，不产生截断假回答；
  ②中间态落库——HITL 任务持久化（awaiting→approved/rejected），用户澄清不丢可接续；
  ③缓存只存"好答案"——qa_cache 只缓存成功且有效回答，无效回答删除，坏数据不污染记忆；
  ④失败留痕——文档 failed 状态 + 结构化日志（带 doc id / 问题上下文）便于定位恢复；
  ⑤图执行快照——checkpointer（内存默认 / 生产 Postgres），中断点恢复。
- 用户侧容错回复：输入错误给明确信息；信息不足走 HITL（交互式容错）；系统性故障如实告知降级——诚实是容错底线，不返回"看似正常实则没依据"的回答。
- 加分应对：全局异常处理——目前路由级 try/except + FastAPI 默认，演进为全局 exception handler（错误映射码 + 统一日志 + 用户文案）；重试预算——不无限重试，一次降级最多一次重试防雪崩，失败交给状态机与人工干预；持久化与一致性——PG 唯一真相源 + Chroma 可重建派生索引 + 状态机收敛（重做方向 outbox）。

**代码锚点**：storage/database.py（session_scope）、api/deps.py（get_db_session）、graph/qa.py（answer fallback/judge）、graph/ingest.py（index_status failed）、storage/repositories.py（IntegrityError 兜底）、graph/checkpointer.py。

## 题 18：如何评测 agent？

**一句话定调**：评测 agent ≠ 评测 LLM——多了"过程"，要看四件事：结果对不对、过程好不好（工具调用/路径效率/终止）、失败诚不诚实、花了多少钱多慢。现有 eval 是"好坏"层，缺"过程级"维度，主动讲出这层框架显得站得比代码高。

**照念版回答**：
- 结果质量（对不对）：端到端黄金集 15 题四题型——正向（答对）、陷阱（不被误导，如反向关系题"王五喜欢李四吗"）、信息不足（必须诚实识别、不能编）、全库（跨文档）。四指标：answerable_accuracy / unanswerable_recognition / citation_coverage / avg_latency。陷阱题与信息不足题是评测集灵魂——没有它们，准确率 100% 只是复读关键词。
- 过程质量（好不好）：①工具调用合理性——该调 final_answer 时调了、该 request_clarification 时识别了、参数对不对；②路径效率——同答案几步走到 vs 绕路/无效重试；③该不该终止——流水线无死循环风险，但 ReAct 式 agent 必须有 max_iterations + 终止率进评测。
- 可靠性与安全：引用真实可溯源（citation_coverage 只数"有没有引用"，不验"引用是否真支撑回答"→ 人工抽检 + 抽样溯源）；幻觉率与拒绝能力（信息不足诚实说不）；错误恢复（诱导出错后能走到兜底而非崩掉/给错答案）。
- 成本与性能：avg_latency（1.28s）、缓存命中率（命中≈0）、token 消耗与平均步数——步数是 agent 特有成本指标。
- 方法三点：①分级——90 单测/集成测管"对不对"、黄金集管"好不好"、人工抽检管"像不像人"；②双模式——fake（内存库+mock）链路冒烟、real（生产 PG+真模型+Chroma）真实指标；每次清库重传文档防缓存污染；③评测集是活的——线上人工抽检坏例定期回流成新黄金题。
- 加分应对：关键词匹配局限——主动认（换说法就误判），演进 LLM-as-judge + judge 本身人工抽检校准；分题型统计别只看总量（陷阱题掉点比正向题严重）；确定性流水线偏端到端评测，自由 agent 必须加轨迹评测（期望工具序列 vs 实际调用序列匹配度）。
- 收尾金句：评测的本质是把"感觉它变好了"变成"数据说它变好了"——没有量化反馈，agent 迭代就是掷骰子。

**代码锚点**：eval/run_eval.py、eval/golden_set.json、docs/interview-concepts.md（Eval 弹药）、tests/（90 测试）。

## 题 20：ToolFactory 如何设计？新增 tool 有哪些步骤？

**一句话定调**：先诚实定位——项目无独立 ToolFactory：final_answer / request_clarification 是图内纯函数直接 bind_tools；实验 supervisor 的 make_plan 是 LangChain @tool。量少裸写够用，工具一多必须抽象成工厂。五层设计是考点。

**照念版回答**：
- 现状：双工具纯函数 + bind_tools，函数签名即工具契约；实验版 @tool 装饰器声明。工具少裸写够用，多则抽象工厂。
- 工厂五层设计：
  ①Registry——工具元信息集中声明：name（唯一/语义化/snake_case/动词+宾语）、description（**写给模型看**：触发场景、参数含义、边界与反面例子，决定调用准确率的第一因素）、parameters JSON Schema（类型/必填/枚举/示例）、副作用声明（只读 vs 写库）、超时重试策略、可见性（暴露给哪些 agent）；注册冲突检测，重名报错。
  ②Factory——按 name 实例化并注入依赖（session_factory/vector_store/retriever），工具不可自 new 连接，保证可测纯逻辑。
  ③Schema 生成——注册元信息自动转 bind_tools / OpenAI function 等格式，上层不感知各家差异。
  ④执行拦截——鉴权/限流/超时/重试/结构化日志/结果校验裁剪挂这层，工具函数保持薄。关键设计：**工具失败返回结构化错误而非抛异常**——模型读得到才能自我修正（ReAct 场景）。
  ⑤测试——单测（参数校验/边界/失败路径）+ 真实 API 冒烟。落地铁律：**先单发真实调用验证格式 → 再改代码 → 再补 mock 测试**。
- 新增 tool 步骤：写实现 → 声明元信息 → 注册 → 加进目标 agent 工具白名单（**非全局可见**，按 agent 分域防选择准确率下降）→ 单测 → 真实冒烟 → 黄金集场景回归确认不拉低准确率。
- 加分应对：工具多——数量与选择准确率 trade-off，50 个工具模型会晕，解法分组/子代理路由（supervisor 分域，worker 只见本域工具）；description 写法——写触发场景 + 反面例子（负面约束比正面描述更能防误调）；写操作工具——声明幂等性，执行层拦截重放（继承双写幂等原则）；MCP 钩子——工具工厂注册表天然是 MCP 工具清单来源（可暴露 upload_book/ask_book/reindex）。

**代码锚点**：graph/qa.py（answer 节点 bind_tools 双工具）、experiments/multi_agent.py（@tool make_plan）、docs/interview-concepts.md（Function Calling 弹药 + 验证流程）。

## 题 21：对 LangChain、LangGraph 的理解

**一句话定调**：别背书。讲三样：用到了哪一层、为什么只用到那一层、清楚它们的坑。素材：LangChain 只用 core 接口，LangGraph 用状态机把 agent 变成可测试的确定性流水线。

**照念版回答**：
- LangChain 定位"生态与抽象层"非框架依赖：价值 = 统一接口（chat model / embedding 抽象 / splitter / 工具），langchain_core.embeddings 让换模型厂商不改业务代码、RecursiveCharacterTextSplitter 直接当切分器。刻意只用 langchain-core：prompt 自己拼、parser 自己写、向量库自己抽象 VectorStore 接口（内存/Chroma 可切）。原因 = LangChain 过度封装、抽象泄漏、升级破坏性大（0.x→1.x 碎过）。**用接口、不用全家桶**。
- LangGraph 定位"agent 编排状态机"，核心抽象四件：State（TypedDict 共享内存，跨节点传中间产物）；Node（普通函数，好测好替换）；Edge（静态边 + 条件边 = 返回路由 key 映射目标节点的函数）；Checkpointer（线程级状态快照，中断恢复/时间旅行）。
- 项目落点：入库图 add_book → chunk_and_index（条件边去重短路，duplicate → END）；问答图 cache_check → retrieve → judge → answer/HITL → record（条件边做缓存命中跳检索、信息不足转 HITL）。选 LangGraph 不选 LCEL：流程有分支有状态，LCEL 声明式线性管道拼起来拧巴；LangGraph 控制流显式、可观测、可测试、可断点恢复。对比 AutoGen/CrewAI 后选择：那些框架自治和 magic 太多，LangGraph 把控制权还给我——图拓扑即协作契约。
- 边界与取舍：checkpointer 生产默认内存版、Postgres 预留；HITL 用任务表实现人工审批而非 LangGraph 原生 interrupt——澄清流程要落库要跨请求，任务表更贴合业务。框架能力按需取舍，不照单全收。
- 加分应对：LangChain/LangGraph 关系——LangGraph 从 LangChain 生态独立出来，core 供原语、LangGraph 做有状态编排，分层不替代；State reducer——多节点写同字段的合并策略（默认覆盖，可自定义 append/去重），并行分支只写独立字段绕开其复杂性；为何不手写状态机——手写可以但 checkpointer/条件路由/图可视化是打包好的通用能力，代价是学习曲线与一层间接，值。
- 收尾金句：框架理解到什么程度，看你会不会说"我不用它的哪部分"——全盘接受叫使用者，知道边界才叫理解。

**代码锚点**：graph/ingest.py、graph/qa.py（StateGraph/条件边/路由函数）、graph/checkpointer.py（InMemory/Postgres）、rag/chunking.py（langchain splitter）、model/factory.py（embedding/chat 抽象）、storage/vector_store.py（自抽象接口）。

## 题 22：agent 有哪些范式？

**一句话定调**：用 Anthropic《Building Effective Agents》框架——先立 Workflow vs Agent 坐标，再逐个范式讲结构 + 适用 + 项目对应物；结尾点破范式是积木不是军备竞赛。

**照念版回答**：
- 坐标：Workflow（预定义代码路径编排，确定性执行）vs Agent（模型自定流程自控循环）。workflow 在可预测性/成本/可评测占优，很多"agent"其实是 workflow。
- ①Prompt chaining：固定步骤链，每步处理上步输出。项目：问答图 cache_check→retrieve→judge→answer→record 即带检索的 chaining。
- ②Routing：先分类再分发专用路径。项目：条件边全在路由（缓存命中跳检索 / 重复文档短路 / 信息不足转 HITL）；supervisor 的全库 vs 单书决策。
- ③Parallelization：sectioning（切块并行，如跨文档检索）+ voting（多路独立尝试投票/评审）。项目：仅有检索并行可能，voting 未用（贵，适合高正确率场景）。
- ④Orchestrator-Workers：中央 orchestrator 动态分解分派再综合。项目：experiments/multi_agent.py 即此范式雏形（静态版）。
- ⑤Evaluator-Optimizer：生成 + 评估循环，不合格带反馈重来。项目：HITL 算半个（judge 信息不足转澄清 = 评估门控）；完整生成-自检-修订是扩展方向。
- ⑥Autonomous agent（ReAct 循环）：Thought→Action→Observation 自循环。刻意不用——流水线确定性强/可观测/易评测；ReAct 灵活但贵、可能死循环需 max_iterations。写入架构文档的决策：不是不会，是不需要。
- ⑦Multi-agent 协作：supervisor-worker / peer 对等 / 层级嵌套（项目 = supervisor-worker 单链）。
- 选型原则：从最简范式开始，评测证明不够再升级；生产停在 chaining + routing + HITL，自主循环与多 agent 留在实验区。
- 加分应对：workflow vs agent 界定——看谁控制流程（代码定路径 = workflow，模型定 = agent），中间态 = workflow 里嵌 agent 节点（流水线内某节点内部是 ReAct 循环）；何时上自主 agent——任务空间不可穷举、路径随中间结果变化时，问答任务空间固定（问一本书）所以不需要；voting 防不了系统性偏差（同源模型错法相同）——要 diversity（不同模型/prompt/温度）；范式选型回到成本曲线——自主 agent 质量上限高但方差大，评测不达标时降级 workflow 更稳。

**代码锚点**：graph/qa.py（chaining + routing + HITL）、graph/ingest.py（duplicate 短路路由）、experiments/multi_agent.py（orchestrator-workers 雏形）、docs/interview-concepts.md（ReAct 未用 + 扩展方向）。

## 题 23：Function calling 整体流程、出错了怎么办、一直错怎么办？

**一句话定调**：先讲清架构事实——代码里的 function calling 是**单次决策式**不是完整 ReAct 循环：bind_tools 一次 invoke，取 tool_calls[0] 按 name 分发，工具不真执行、结果不回填。final_answer / request_clarification 本质是**结构化表态**（让模型用调哪个工具来表态"信息够不够"），不是行动。答案贴此设计，别答成通用 agent 循环。

**照念版回答**：
- 整体流程：answer 节点 bind_tools 双工具一次 invoke → 取 tool_calls[0] 按 name 分发：final_answer → 用 args.answer；request_clarification → 无澄清转 HITL、有澄清仍不足则 missing_info 当回答。无执行循环（表态不需要执行回填）。
- 出错四层兜底：
  ①模型不支持工具——hasattr 检查 bind_tools，无则纯文本 invoke；
  ②调用抛异常（网络/API/格式）——catch → 纯文本 fallback + 异常日志；纯文本仍过"信息不足"措辞检测转 HITL（fallback 不裸丢答案，保诚实底线）；
  ③没走工具 / 参数畸形——纯文本路径或 or '' 兜底。**已知缺口诚实说**：未知工具名（模型幻觉）目前掉进 else 被当 final_answer，无 answer 则静默空——改进 = 工具名白名单校验，不在白名单转 HITL 或重试一次，不静默空答；
  ④一直错 = 系统性信号非偶发，三条线：工程线——熔断开关：连续失败 N 次自动切纯文本模式 + 告警保基本可用；根因线——schema 太复杂 / description 有歧义 / 模型能力不够，换简单参数、加枚举、换模型；数据线——结构化日志统计 tool_call 成功率 / 幻觉工具名率，坏例回流黄金集变成红牌题。
- 加分应对：为何不做完整执行循环——工具是决策不是行动、无执行结果要回填；工具真去检索/写库才需要循环（执行→回填→再决策）+ max_iterations + 重试预算；一次多工具——取 tool_calls[0]，决策互斥（要么答要么澄清），连续多工具场景才需循环 + 每次校验；怎么知道一直错——可观测先行（成功率/unknown name 率/fallback 触发率指标），没有指标只能等用户投诉。
- 收尾金句：Function calling 容错本质 = 每条错误路径都要有出口，且出口不允许"假装成功"——空回答、编答案都不行，降级到 HITL 问用户永远比给假的强。

**代码锚点**：graph/qa.py（answer 节点 bind_tools / tool_calls 分发 / 三层 fallback / _looks_like_no_info）、docs/interview-concepts.md（Function Calling 弹药：双工具替代关键词启发式）。

## 题 24：换新模型需要更换提示词吗？

**一句话定调**：非黑即白都错。把提示词拆三层：任务规则层换模型不该动、格式协议层要验遵从度、风格示范层最可能要调。工程底气 = 换模型 = 改配置 + 跑评测回归，指标说话不靠感觉。

**照念版回答**：
- ①任务规则层（业务语义：怎么答/引用规范/诚实规则/回答结构）——换模型尽量不动，描述"业务要什么"而非"模型脾气"，最该锁死；
- ②格式协议层（输出格式/工具 schema/JSON）——大多可复用但必须验遵从度：不同模型格式遵从差异大（弱模型可能工具名幻觉、漏字段、JSON 切碎），重点回归对象；
- ③风格示范层（few-shot/措辞）——最可能调：示例为特定模型风格调，换模型风格不匹配反起反作用。
- 工程做法：模型走工厂抽象配置可切、prompt 集中 _build_answer_prompt 一处 → 换模型 = 配置级操作非代码手术。换完先不改 prompt 直接跑评测（15 题黄金集 real 模式四指标），不掉不动、掉了先定位语义 or 格式遵从再动对应层，改完同一套评测再过。bind_tools fallback：不支持工具调用的模型自动走纯文本降级不崩。
- 实战经验：先单发真实 API 验证工具调用格式在该模型可用 → 再改配置 → 再补 mock 测试；文档说支持不算数，真实调用跑通才算。
- 加分应对：换强模型指令要减（自己能推理的步骤写死反而束缚，提示词量与模型能力大致反比）；多模型共存取交集（只写业务规则不写模型怪癖适配，为 A 调的 prompt 必须跑 B 的回归防互相破坏）；格式崩除了 prompt 还有解析容错 + 校验重试（prompt 是第一道防线，工程兜底第二道）。
- 收尾金句：提示词工程终极目标是模型无关——换模型那天只改配置、跑评测，而不是连夜改 20 处 prompt，说明 prompt 写对了。

**代码锚点**：model/factory.py（模型抽象配置可切）、graph/qa.py（_build_answer_prompt 集中 + bind_tools fallback）、eval/run_eval.py（换模型验收回归）、docs/interview-concepts.md（真实 API 验证流程）。

## 题 25：RAG 原理？和模糊搜索的区别？一定比模糊搜索高效吗？怎么提召回率？

**一句话定调**：前两问送分，坑在第三问——"一定更高效吗？"答案是不一定，字面精确场景模糊搜索反而强。第四问召回率引用前面题（题 4/提召回率）一句带过不重复展开。

**照念版回答**：
- RAG 原理：动机 = LLM 参数知识静态、有截止日期、会幻觉、记不住私有文档。思路 = 把记忆从参数外置：离线（解析→分块→embedding→向量库）+ 在线（query embedding→检索 top-k→拼 prompt→看原文回答）。本质 = 检索替代记忆、外部上下文补充参数知识；知识更新不重训、引用可溯源、幻觉被"有据可依"压住。
- vs 模糊搜索（先对齐定义：LIKE/全文索引 tsvector/编辑距离 = 字面近似，与语义检索是两个物种）：
  ①匹配对象——模糊搜词面（子串/编辑距离），RAG 搜语义；
  ②鲁棒性——语义场景 RAG 完胜（"谁背叛了主角" vs 原文"宴席上出卖盟友"），精确词场景（编号/ISBN/专名/引语）模糊精确命中、向量反而漂；
  ③排序与可解释——模糊按字面重合度、可高亮直观；RAG 黑盒向量距离 → 必须用 citation 兜可解释性；
  ④成本——模糊零基建，RAG 要 embedding/向量库/算力持续成本。
- 一定更高效吗？不一定。"高效"分场景：字面精确需求（查编号/找原话）模糊/全文又快又准又便宜；语义问答才 RAG 赢；小数据量（几万条内）LIKE/全文可能比向量库更划算。RAG 是"语义召回"这一特定需求的正解，不是银弹。成熟做法 = 混合：全文/BM25 兜字面 + 向量兜语义 + RRF 融合。
- 提召回率四抓手（详见"提升 RAG 召回率"追加节）：候选池放宽 top 50~100 再精排、索引侧 parent-child + 上下文增强、BM25+向量双路 + RRF、查询侧改写兜底，每步 Recall@k 评测说话。
- 加分应对：RAG 不是消除幻觉而是换一种——压"无依据幻觉"，引入"检索错硬答"幻觉（judge + HITL 才有价值）；结构化/固定查询/小数据场景 RAG 反而更差（查库存价格用 SQL 即可）。
- 收尾金句：RAG 与模糊搜索不是新旧替代，是互补的两条召回通道——字面交给字面检索、语义交给向量，融合后才是完整搜索引擎。

**代码锚点**：rag/retriever.py、graph/qa.py（judge/HITL）、storage/vector_store.py、eval/（召回评测）、docs/interview-qa-answers.md（题 4 + 提升召回率节）。

## 题 26：怎么理解上下文？范畴、边界？工具算不算？skill 算不算？

**一句话定调**：边界判据 = 是否此刻躺在模型输入窗口里、参与这一次推理——参与即上下文，执行体永远在窗口外。工具和 skill 都"一半算一半不算"：描述/schema 进上下文（模型靠它决策），执行体不进（系统替模型做）。

**照念版回答**：
- 定义：上下文 = 模型在这一次决策时能看到的全部输入信息，决定"模型此刻知道什么"。
- 范畴四类：①任务指令（系统规则/业务约束/用户目标 = 不变骨架）；②对话与场景（会话历史/当前问题/澄清 = 流动血肉）；③外部证据（RAG 片段/记忆回唤 = 最重动态上下文，检索质量决定上下文质量）；④执行状态（agent 进行到哪步/中间产物，LangGraph state = 执行上下文）。
- 边界两判据：①是否参与本次推理——模型要"看见"才能用它决策；②最小充分原则——只放影响决策的最小信息，能外置全外置（历史落库、规则进代码），与上下文膨胀治理同一原则。
- 工具算不算：一半。schema + description 进上下文（决定该不该调/参数怎么填，写不好=往上下文扔垃圾，拉低工具准确率）；执行体不进（内部逻辑模型不推理）。工具本质 = "能力声明"是上下文的投影，"能力实现"是世界的入口——模型窗口内做决定，系统窗口外做动作。
- skill 算不算：同样一半。触发条件 + 摘要进上下文（让模型知道有此能力何时用）；内部步骤脚本不进 = 程序性记忆，平时只放索引卡片，被选中才由系统展开执行。工具 vs skill 差异 = 抽象粒度：工具是单次原子动作，skill 是打包一串动作与知识的可复用流程。
- 统一框架：上下文 = 所知，工具 = 所能，skill = 所会；记忆 = 被时间拉长的上下文（跨会话回唤），状态 = 正在发生的上下文，检索证据 = 临时注入的上下文。判据一句话：它此刻在不在模型输入里参与推理。
- 加分应对：系统 prompt 算且是唯一每轮常驻上下文 → 要克制（"地租最贵的上下文"）；记忆 = 可回唤的上下文（平时躺存储不算，被检索命中注入窗口才算，QA 缓存命中即把整轮推理替换成旧答案）；工具描述长 = 污染（50 工具 description 上千 token → 分组按 agent 分域注入）。
- 收尾金句：上下文工程的本质是决定什么该让模型看见——看见太多抓不住重点，看见太少无米之炊；工具和 skill 的价值在窗口外那一侧，是模型把手伸向世界的通道。

**代码锚点**：graph/qa.py（prompt = 问题+检索块+澄清）、graph/qa.py answer 节点（bind_tools schema 进上下文）、storage/repositories.py（qa_cache = 可回唤上下文）、graph/checkpointer.py（执行状态）。

## 题 14：如何提升 skill 选择的准确率？

**一句话定调**：技能选择本质是分类问题（输入 = 用户意图，输出 = 技能集合）。提升准确率 = 输入更清晰（描述工程）+ 类别更少更可分（技能设计与路由）+ 错误被看见（评测反馈闭环）。反直觉点：选错头号原因往往不是模型笨，是技能名片与边界画得糊。

**照念版回答**（按杠杆从大到小）：
- ①描述工程（第一杠杆，最便宜）：name 语义化动词开头望文生义；description 写给模型看——触发场景 + 参数含义 + **负面约束**（"不要在什么场景用"）；每技能 2~3 触发句 + 1~2 反例。模型对"何时不该选"的理解比"何时该选"更重要（同 tool factory 题 20 原理）。
- ②技能设计：职责互斥（描述重叠 = 必然选错，归并或明确分层，不是模型问题是设计问题）+ 粒度一致（别一个管单动作一个管全流程）。边界清晰 = 选择从语义推理降级成模式匹配。
- ③数量控制与分域路由：技能 >10 全量注入 = 候选本身成噪音（题 26）。两级：先粗路由按域筛（关键词/embedding 预筛等廉价手段，不一定用模型），域内技能个位数。**选择复杂度 = 候选数量**。
- ④两级选择机制（先召回再精排）：技能描述也 embedding，query 先召回 top-k（~5）候选，再让 LLM 精选——与 RAG 两段式检索同模式。
- ⑤评测与反馈闭环：每技能配黄金正/负例，跑技能选择准确率、按技能单独看；错选案例回流成负例。选错不可怕，可怕的是没人知道。
- ⑥兜底降级：低置信不硬选（犹豫时问用户而非赌）；确定规则走代码不走模型——确定性优先，模型只处理真需语义理解的判断。
- 加分应对：描述写触发句而非属性表（"当用户想导入一本新书时使用" >> "参数：书籍文件名"——模型对场景叙述的理解远好于字段列表）；检索式选择的召回防漏——技能描述要自包含（覆盖用户各种说法：导入/上传/添加一本书）；评测集来源——首版人工标每技能 5 正 5 负，线上用户纠偏自动沉淀负例。
- 收尾金句：技能选择准确率不是模型参数调出来的，是名片设计 + 候选管理 + 错误记账三件事做出来的——模型只是读名片的人。

**代码锚点**：docs/interview-concepts.md（MCP 扩展 upload_book/ask_book/reindex 的技能化前景）、graph/qa.py（双工具决策）、题 20/26 的互引（schema 即决策依据、技能描述进上下文）。

## 题 5：如何做 query 改写？

**一句话定调**：不是所有 query 都过 LLM——按需改写（先判断值不值得）+ 改写与原文双路召回（改写失败不能比不改更差）。项目已有一层"改写"：normalize_question 归一化 + HITL 澄清拼进问题（用户侧补全），只是未上 LLM 级。

**照念版回答**（改写策略按成本从低到高五类）：
- ①规则归一化（零成本）：去噪/去停用词/规范化。项目 normalize_question（NFKC/小写/去标点）本质是 query 改写第一层。
- ②指代消解与补全（多轮刚需）："那她后来怎么样了" → 补全历史指代成完整问题。带上历史拼接或小模型改写。
- ③LLM 改写（最常见）：口语 query → 检索友好表述，去口头语/补隐含前提/问句转陈述友好；不是关键词堆砌，是保持语义优化匹配面。成本 = 一次小模型调用 + 延迟。
- ④HyDE（假设文档法）：先生成假设答案、拿答案去检索——假设答案与文档同为"陈述体"，语义距离比问句近。效果好但更贵，只用于硬问题。
- ⑤多路改写（RAG-Fusion / multi-query，最贵）：一 query 生成 N 个角度变体各检索 + RRF 融合。适合宽泛/多义/可多角度的问题。N 倍成本，最后手段。
- 工程判断三条：①按需改写不无脑全改——先低成本判断 query 类型（事实题直接检索，口语/指代/复杂才改写），改写本身也是路由决策；②改写必须与原文双路召回——LLM 改写跑偏是常态，保留原 query 当保险，最坏退化回不改；③评测验收——改写前 vs 后跑同套 Recall@k + 端到端，提升值不值延迟与 token 成本，提不了召回的改写是自嗨。
- 项目落点：现处于第①类 + 用户侧补全（HITL 澄清拼进问题再检索 = 让用户参与改写）；LLM 级改写是"提升召回率"路线图第四步，前三步做完评测说不够才上——改写是最后一块拼图不是第一块。
- 加分应对：改写跑偏 → 双路召回 + 评测兜底（单路改写是赌博、双路是保险）；小模型够用吗 → 普通改写够，指代消解/HyDE 要求高按任务选模型；放流水线哪 → retrieve 前、cache_check 后（缓存命中的问题不值得改写，改写本身也要吃缓存防重复）。
- 收尾金句：query 改写是拿一次廉价推理换检索质量提升——值不值不看改写漂不漂亮，看召回率和答案准确率涨没涨。

**代码锚点**：rag/retriever.py + storage/service.py（normalize_question）、graph/qa.py（clarification 拼入问题 = 用户侧补全）、docs/interview-qa-answers.md（提升召回率节：改写是第四步）。

## 题 2：模型如何做技术选型？

**一句话定调**：选型 = 评测驱动的采购，不是看榜单挑最贵的。六步：硬约束 → 分角色 → 黄金集实测 → 成本建模 → 灰度验证 → 留 B 计划。factory 抽象 + eval 双模式 = 现成选型试验台。

**照念版回答**：
- ①需求画像先列硬约束：中文读书问答、需 function calling、单轮决策、延迟预算、私有数据。硬约束 = 支持工具调用 + 中文强 + 上下文够 + 延迟可接受；不满足直接出局。**工具调用能力不信文档，单发真实 API 验证**（踩过 DeepSeek 的坑，实测跑通才落地）。
- ②分角色选型：chat 模型看任务复杂度/工具调用可靠性/中文生成；embedding 模型看检索质量（中文 benchmark/维度/长文档领域适配/成本）——两套标准别混。
- ③评测驱动：eval real 模式 + 配置级切模型，同 15 题黄金集同四指标（可答/不可答识别/引用覆盖/延迟）逐个跑。跑自己任务不迷信公开榜（榜一可能在你场景里工具格式崩）。
- ④成本建模：用量模式决定成本——QA 缓存命中跳过 LLM、embedding L1 缓存，实际 token 比裸算低，必须计入（否则误杀质量好单价略高的模型）；延迟超预算线降级体验。
- ⑤灰度验证：真实流量跑一周盯三指标——工具调用成功率、fallback 触发率（高 = function calling 虚）、缓存命中后延迟。fallback 率高是选错的最早信号。
- ⑥留 B 计划：选型文档写选定 + 备选——限流/涨价/下线/版本漂移是常态；factory 抽象使切换 = 配置级操作，B 计划是随时能切的预案。
- 决策落点：chat = 支持 function calling 的中文模型（实测验证）；embedding = 中文检索达标 + 维度兼容向量库；本地 vs API 看数据隐私（私有书籍不出域则接受本地小模型折损或自建推理）。没有最好的模型，只有约束下最合适的——选型文档 = 约束 + 数据 + 成本的结算单。
- 加分应对：大/小模型混合策略——重活大模型、轻活（改写/路由/分类）小模型，不同节点不同模型；防模型漂移——锁版本 + 升版前跑黄金集回归（模型像依赖库一样管）；中文场景——中文 embedding 看中文榜、生成质量实测、分词影响 embedding，跑完自己数据才知道。
- 收尾金句：技术选型的本质是在约束下做决策并留好后路——评测数据是现在的证据，B 计划是将来的保险。

**代码锚点**：model/factory.py（模型抽象配置可切）、eval/run_eval.py（real 模式选型实测）、graph/qa.py（工具调用 + fallback 链）、docs/interview-concepts.md（DeepSeek function calling 验证流程）。

## 题 3：Embedding 模式切换，如何不影响现网、不重新全量向量化？

**一句话定调**：Embedding 切换 = 换向量空间（新旧向量不可比、维度都可能不同，混用 = 检索全错）。正解不是"不重向量化"，是"把重向量化变成无感后台任务"：版本化隔离 + 影子回填 + 原子切换 + 保留回滚。项目隐藏优势：Chroma 本就按"可从源文件重建"设计，当年为故障恢复埋的伏笔今天用在平滑迁移上。

**照念版回答**（五步）：
- 铁律：query 与文档必须在同一向量空间；切模型瞬间旧文档/新 query/缓存向量分属两空间。
- ①版本化隔离：新模型注册 v2、建独立集合（collection 名带版本号 v1/v2）；旧集合只读服务现网，一字不动（维度都可能不同，隔离必须）。
- ②影子验证：同一批代表文档用 v2 重嵌，跑黄金集 Recall@k + 端到端四指标，v2 不劣于 v1 才继续（防止换了才发现更差）。
- ③存量回填 = 后台任务不停机：分块参数固定 + chunk id 确定性 + 源文件路径存 PG + upsert 幂等 → 按文档粒度重跑入库链路到 v2，index_status 状态机追踪，断点续跑、失败重试、重跑不双写；迁移期新文档直接写 v2，存量逐个补齐。
- ④原子切换：回填 100% + v2 黄金集达标双门槛过才切；VectorStore 走工厂抽象 = 配置级切换不碰代码；切换瞬间 query 与文档同空间（v2 已完整）。
- ⑤观察期与回滚：v1 保留观察窗口不删，异常一键回切，稳定后清理。回滚能力 > 一切。
- 缓存坑（易漏）：qa_cache.question_embedding 也是旧空间——精确命中缓存（纯哈希不依赖 embedding）可保留；语义缓存（依赖向量相似度）必须作废或按 embedding 版本分区；缓存键加 embedding 版本号最干净。
- 加分应对：迁移窗口新文档——每文档记 embedding 版本号；要保回滚期 v1 完整则双写（两次 upsert 成本低）；重嵌成本——文档级并行 + 断点续跑，一次性 API 费用 vs 切错返工必须花，影子验证把关；证明切换无影响——切后盯检索指标/fallback 率 + 黄金集再跑，稳定一周才算完成。
- 收尾金句：Embedding 升级本质是换坐标系——旧地图不能用了，但不必让全城停摆重绘：先画好新图（影子回填）、当众换图（原子切换）、旧图留着以防万一（回滚窗口）。

**代码锚点**：storage/vector_store.py（VectorStore 抽象 + collection 名可配置 + upsert 幂等）、storage/service.py + graph/ingest.py（确定性 chunk id + file_path 可重建 + index_status）、storage/repositories.py（qa_cache question_embedding + 缓存键）、model/factory.py（embedding 模型配置切换）。

## 题 19：Skill 加载机制如何实现？系统 prompt 如何设计？

**一句话定调**：把题 14（选对技能）/ 26（描述进上下文、执行体在窗外）串成中间层机制。答案核心 = 两级加载：元信息随时可取、执行体按需展开；prompt 永远只放被选中的子集。

**照念版回答**（skill 加载四段）：
- ①存储层：每个 skill = 自包含包——元信息（name/description/触发场景/参数 schema）+ 实现体（脚本/提示词模板/多步流程）+ 依赖声明。文件系统目录或 DB 均可，关键是元信息与实现体分离。
- ②索引层：启动时把全部 skill 元信息加载成注册表，只留"名片"不载实现体（名片轻，全量常驻无压力）。
- ③注入层（模型感知的加载）：按请求路由/检索出 top-k 技能（题 14），只把选中技能 description 注入系统 prompt——模型永远面对子集而非全量技能表。
- ④执行层：模型决定调用后才加载实现体，此时做依赖注入（session/vector_store/retriever 按声明给）；跑完进程内 LRU 缓存实现体；技能文件变更热更新 + 版本号。
- 关键设计：加载时机分层（名片常驻 / 实现体懒加载 / prompt 按需注入，三件事别混——混了 = 50 个 description 全塞 prompt 的灾难）；安全边界（权限声明、白名单执行、外部技能走 MCP 发现，注册表 = 工具清单统一入口）。
- 系统 prompt 设计：克制 + 模块化 + 可验证。四区结构不写散文：①身份区（你是谁/服务场景，一句话）；②规则区（行为约束——诚实规则 + 引用规范，写成模型能自查的句子："每一句引用必须来自提供的检索片段，找不到就不答"可验证；"回答要准确专业"是模型要猜的，没用）；③边界区（不做什么：不编造引用/信息不足要明说，负面约束防幻觉）；④动态区（检索块/问题/澄清/选中技能描述，按请求组装）。前三区静态骨架几乎不改，动态区每次请求的血肉。
- 写作三原则：指令短句；正面指令为主 + 关键负面约束；规则区与资料区物理隔离（检索块别混进规则区，模型分不清命令与材料）。
- 版本管理：system prompt 版本化，改 prompt = 改模型行为（题 24），每次改动跑黄金集回归；_build_answer_prompt 集中管理（规则固定、检索块与澄清动态拼入），诚实规则被 eval 信息不足题型验证。
- 加分应对：技能内含提示词模板——模板是技能一部分，执行时展开拼进当次 user prompt（带进一段子上下文，用完即走不常驻）；prompt 太长——常驻上下文每轮付地租，身份一句话/规则只留高杠杆/能外置的写代码别占 prompt；动态区与规则区冲突——规则区优先（能力说明 vs 行为底线），prompt 写明优先级。
- 收尾金句：Skill 加载 = 把能力做成按需拆封的包裹（窗口里只有名片和选中说明书，仓库货架等下单才出库）；系统 prompt = 把行为底线写成模型能自查的契约——不靠模型悟，靠规则可验证。

**代码锚点**：graph/qa.py（_build_answer_prompt 集中 + 规则/资料分区）、eval/（诚实规则被信息不足题型验证）、docs/interview-concepts.md（MCP：注册表 = 工具清单来源）、题 14/20/26 互引。

## 题 11：一个工具有一二十个参数，如何提升工具调用准确率？

**一句话定调**：最值钱的认知——"一二十个参数"本身就是设计缺陷不是模型问题。三层解法：治本靠拆（胖工具拆瘦）、治标靠让模型少填系统多填（装配权收回）、兜底靠校验修复循环。

**照念版回答**（设计层 → schema 层 → 装配层 → 容错层）：
- ①设计层（最重要）——别让工具有一二十个参数：拆工具（按决策点拆，20 参数 ≈ 3~4 个决策 → 拆 4 个工具，每个 3~5 参数；准确率与参数数量强负相关）；缩必填（只留需模型语义判断的，其余默认值/可选，可选泛滥 = 乱填幻觉）；降自由度（自由文本改枚举，format 给 pdf/docx/txt）。
- ②schema 层——保留多参数时的模型友好化：嵌套分组（JSON Schema 嵌套 object 按语义分组——源定位/输出控制/行为选项，模型按层级组织比平铺 20 项稳）；每参数写"值从哪来"（可操作指引非"文件名称"式废话）；additionalProperties: false 防幻觉字段。
- ③装配层（最猛一招）——模型只填该填的、系统填剩下的：20 参数里 15 个（用户/会话/文档范围/权限/默认配置）系统从上下文状态自动装配，只暴露 3~5 个需语义理解的给模型。两段式：模型先给"意图参数" → 服务端装配完整参数再执行。大部分参数错误不是模型笨，是它压根不该填。
- ④容错层——校验修复循环：调用后 schema 校验（缺必填/类型错/枚举外）→ 返回结构化错误给模型一次修复机会；参数归一化兜底（近似值映射合法值）；重试一次失败即降级（问用户/默认参数），绝不静默执行残缺调用。
- 架构级备选：胖复杂工具让模型输出结构化 JSON 而非 bind_tools（填数据心态比"调函数"稳，系统解析映射）；极端情况让模型自然语言描述意图 + 代码解析提取——让模型做擅长的（理解表达），让代码做擅长的（精确装配）。
- 加分应对：拆出太多新工具 → 按决策点数不按参数数拆；嵌套 schema 模型支持差 → 退回平铺只暴露必填 + 默认值（嵌套是优化项、瘦身是必选项）；修复死循环 → 限一次修复机会失败即降级（成本封顶，呼应题 17 重试预算）。
- 收尾金句：工具参数越多，模型越像填一份看不懂的表格——准确率提升不是把表格教给模型，是把表格撕了，只留它真正该答的那几栏。

**代码锚点**：graph/qa.py（双工具 2~3 参数 = 瘦 schema 范例）、experiments/multi_agent.py（make_plan 3 参数带 note 理由）、docs/interview-concepts.md（Function Calling 弹药）。

## 附录：项目事实速查（答细节题用）

- 部署形态：Postgres（postgresql+psycopg://…reading_agent）+ ChromaDB PersistentClient（rag/chroma_db）+ FastAPI + LangGraph。
- 入库链路：解析（txt/epub/pdf/docx，epub 走 BeautifulSoup、pdf 走 pypdf extract_text）→ 章节级分块（800/100）→ embedding → Chroma upsert（id=doc{doc_id}-{index}）。
- 去重键：documents.file_hash（文件字节 SHA-256）、content_hash（归一化文本 SHA-256），均 unique。
- QA 链路：cache_check（PG 双级缓存）→ retrieve（Chroma top_k=6、cosine ≥0.45、document_id 过滤）→ judge（空结果 → HITL）→ answer（LLM 双工具 final_answer/request_clarification）→ record（写 chat_messages）。
- 黄金集：15 题四类（正向/陷阱/信息不足/全库），指标 answerable_accuracy / unanswerable_recognition / citation_coverage / avg_latency；实测 15/15、延迟 1.28s。
- 生产 vs 实验：生产 = 单流水线状态机 + HITL；实验 = experiments/multi_agent.py supervisor-worker（无兜底）。
- 待补题号：2、3、5、6、11。

## 附录：项目补充场景路线图（面试弹药）

> 用途：被问"项目下一步做什么 / 还有什么场景 / 如果重做"时的弹药。每个场景标注它闭环的面试题。

- **第一梯队：补齐检索短板**（闭环题 4/25/提召回率）——①混合检索真落地：BM25 稀疏路 + 稠密 + RRF 融合（把"提升召回率"答案变事实）；②检索评测集：citations 的 chunk_id 弱标注 gold chunk → Recall@k（评测从端到端补到检索层，补题 18 过程质量短板）。
- **第二梯队：实验变生产**（闭环题 9/13/22）——跨文档对比问答：A/B 书同主题异同 → 并行检索 fan-out + 合流 synthesis（黄金集已有 cross-01、接口已支持 document_ids，缺并行检索）；supervisor 规划接生产做动态 doc_scope。多 agent 从实验代码变生产落地。
- **第三梯队：产品连续感**（闭环题 7/12/15）——阅读进度场景（记录读到哪、只答进度内内容，"别剧透"差异化）；用户画像记忆（常用书/偏好 → 个性化重排）；克制的多轮（增量摘要 + 最近 N 轮，兑现题 7 演进承诺）。
- **第四梯队：行业接口与内容形态**——MCP Server 封装（upload_book/ask_book/reindex，docs 已写扩展方向，做了 = "实现了"非"了解过"）；表格结构化（docx/pdf 表格 → Markdown 入库，比 OCR/VLM 便宜，题 10 短板先迈一步）。
- **工程可靠性线**（闭环题 17/23/3）——全局异常 handler、工具调用成功率/fallback 率指标、embedding collection 版本化（为将来换模型铺路）。
- 只挑三个的优先级：混合检索 + 检索评测集 → 跨文档对比问答 → MCP 封装。每个 1~2 周增量、可独立讲成 STAR。

## 更新日志
- 2026-09-04：初始存档（题 1/4/7/8/9/10/12/13 + 追加预演 1~6）。
- 2026-09-04：追加题 15（上下文膨胀保障成功率）。
- 2026-09-04：追加题 16（请求到回答全链路走查）。
- 2026-09-04：追加题 17（执行错误容错与持久化）。
- 2026-09-04：追加题 18（如何评测 agent）。
- 2026-09-04：追加题 20（ToolFactory 设计与新增工具步骤）。
- 2026-09-04：追加题 21（LangChain/LangGraph 理解）。
- 2026-09-04：追加题 22（agent 范式）。
- 2026-09-04：追加题 23（Function calling 流程与容错）。
- 2026-09-04：追加题 24（换模型是否更换提示词）。
- 2026-09-04：追加题 25（RAG 原理与模糊搜索对比）。
- 2026-09-04：追加题 26（上下文的范畴与边界）。
- 2026-09-04：追加题 14（skill 选择准确率，补空档）。
- 2026-09-04：追加题 5（query 改写，补空档）。
- 2026-09-04：追加题 2（模型技术选型，补空档）。
- 2026-09-04：追加题 3（Embedding 切换平滑迁移，补空档）。
- 2026-09-04：追加题 19（Skill 加载机制与系统 prompt 设计，补空档）。
- 2026-09-04：追加题 11（多参数工具调用准确率，补空档）。
- 2026-09-04：附录：项目补充场景路线图。
- 2026-09-04：落地路线图第一梯队——混合检索（BM25+稠密 RRF，retrieval_mode 默认 vector 保守开启）+ 检索评测工具（eval/build_retrieval_gold.py、eval/run_retrieval_eval.py）。
- 2026-09-07：MCP Server 封装完成（list_books/upload_book/ask_book/reindex_book，`uv run readingassistant-mcp` stdio）。real 检索 A/B：vector vs hybrid 在 golden 上 1.000 对等（gold 由引用弱标注自生成、存在天花板，无回归即达标）。
- 2026-09-07：落地路线图第二梯队——多文档对比问答：document_ids 多选 → 并行 fan-out 检索 → 合流 synthesis；引用带书名；eval/golden_multi_doc.json 跨书评测；单文档/缓存路径零回归。
