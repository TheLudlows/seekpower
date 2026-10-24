# 本地全文检索对照协议（执行前固定）

- anyseek: `6d28798a300bb947b2eb1d3e48e928d193103e17`, `cargo build --release --locked --features server`；不改生产代码/上限。Meilisearch Community v1.55.0 官方 linux-amd64，资产 SHA256 `7e33f5889235f14290074a19a6d3574bd9b10fb2ee8731eabb7dafd0714b0513`，非 Enterprise；保存 LICENSE 与 LICENSE-MIT。
- 同一 Linux KVM/overlay 宿主环境，4 CPU quota / 16 GiB cgroup 上限。每次仅一个服务，固定 CPU 0–1；负载端 CPU 2–3。无专用宿主、无单进程硬内存隔离，双方共享同一容器上限，不声称严格硬件隔离。固定本地 HTTP、Top-10、相同返回 title/text/docid；不含向量/模型。
- MIRACL 中文原始十分片全 SHA256/记录数校验。固定40个 dev query（seeded ID hash），纳入其全部相关和不相关标注文档，余量从全语料按 seeded doc-ID hash 取 distractors；1000/5000为嵌套子集。是有意保留相关文档的小语料诊断，不能与完整 MIRACL 分数比较或外推10万。只用官方 qrels 计算 nDCG@10 / Recall@10 / MRR@10；另列自然中英混合覆盖与无标注构造查询，不虚构相关性。
- 两种配置分别报告：`default` 保留双方实际 API 搜索/排名默认（anyseek 默认 UnicodeAlphanumeric、AND；Meili 默认容错/前缀/词项退让）；`restricted` anyseek显式 UnicodeAlphanumericZH、AND，Meili禁用 typoTolerance/prefixSearch且 matchingStrategy=all，双方显式搜索 title/text。后者也不称完全同语义：分词、繁简/折叠、排名、字段权重及句法仍不同。字段匹配/过滤/前缀/错字/繁简用独立少量功能样例核查。
- 构建计时从首批提交到全量可搜索并核对数量；双方统一每批50条。anyseek逐记录 durable+visible，Meili异步task须等完成并验证搜索；报告该差异。构建后显式 checkpoint（anyseek）单独计时；验证进程重开仍可查询，不用重开证明系统断电持久性。内存取 `/proc` 定时 RSS/HWM，磁盘同时报逻辑字节/分配字节，包含各自规范原文/日志，不只数索引文件。
- 先1000预检/完整实验，通过后才5000。保留原文，不截断或提高配额。预算拒绝逐条记录；若5000碰正式预算，不越过，报告最大已接受量且不把失败加载当5000性能结果。
- 每次冷启动=停止并重开服务，报告启动到健康及首轮查询。没有权限清 OS cache，不称真正冷缓存。热查询先两遍预热；并发1/4/8，各3轮、每轮5秒 closed-loop，记录每请求墙钟延迟、状态、响应大小、QPS/P50/P95/P99/错误率及样本数。负载端实现/CPU可能限制吞吐，不能把客户端饱和QPS解释为引擎极限。
- 写入期间查询使用全新隔离数据根，先装1000，再以固定2条/秒更新100个既有ID；并发4查询覆盖整个50秒窗口。每引擎都只1100操作以内，保留错误/预算拒绝；报告写入确认和可见等待，不吞错。以 restricted 配置执行，不重启性能优化专项。
- 原始响应/延迟/资源/配置保存在 `artifacts/benchmark/`；脚本、协议和报告在本目录。所有数据根仅属于本实验，不接触部署数据。不推送、不创建PR。

## 预检后的明确偏离

中文 `restricted` 在第385条写入（已接受384）返回资源预算错误，因此不运行其5000条或写入混合负载，不把部分索引冒充1000。保留该失败，生产实现与限制不动。写入期间查询改用双方 `default` 的独立1000条根；这只验证默认接口在该负载下的行为，不弥补中文质量或语义等价性。
