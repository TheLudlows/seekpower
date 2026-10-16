# Seekpower

Seekpower 是单机全文、向量与混合检索引擎。当前组成是 `seekpower` Rust 核心库、`seekpower_server` HTTP 服务和内嵌 Web UI。模型推理、文档解析与切块由调用方完成。

[完整功能清单](FEATURES.md) · [服务使用](seekpower_server/README.md) · [架构](ARCHITECTURE.md) · [迁移说明](MIGRATION.md) · [状态](STATUS.md) · [后续计划](SEEKPOWER_PLAN.md)

## 构建与启动

从本仓库构建核心库和 HTTP 服务。

```sh
cargo build --workspace
export MASTER_KEY_SECRET='replace-with-your-private-secret'
cargo run -p seekpower_server -- local_ip=127.0.0.1 local_port=8080
```

打开 `http://127.0.0.1:8080/`，将启动日志中的 `Master API key` 填入连接表单，即可创建索引、写入和检索。主密钥由 `MASTER_KEY_SECRET` 派生，重启复用其索引；不自动生成普通 key。需要独立数据作用域时，可用主密钥创建/撤销额外 key。完整流程见 [服务使用](seekpower_server/README.md)。

## 当前能力

| 领域 | 保留内容 |
| --- | --- |
| 全文 | BM25F/邻近度评分、布尔/短语、多字段、中文分词、停用词、N-gram |
| 向量 | 外部 JSON/base64 向量、多向量字段、F32/I8、三种距离、量化、全簇扫描与 IVF ANN |
| 混合 | 全文/向量独立召回后 RRF 融合 |
| 元数据与结果 | 全文/向量/混合元数据过滤、多字段排序、计数、分页、字段投影、双向遍历 |
| 数据与服务 | JSON 单条/批量写入、读取、更新、删除；索引生命周期；API key 鉴权与数据作用域 |
| Web UI | 密钥连接、索引创建/选择、字段 schema 配置、JSON 粘贴/文件写入与提交、三种检索模式、外部向量输入、分页、原文/链接预览、日期输入过滤与排序 |

完整接口、配置、外围功能及限制见 [FEATURES.md](FEATURES.md)。特别注意：HTTP 向量检索默认搜索全部簇，核心显式 ANN 仍受簇探测范围影响；UI 尚无完整管理页面；内部 DocID 不是稳定业务 ID；实时可见不等于崩溃可恢复。分布式与可靠性补强属于后续阶段。

## 外部向量示例

创建索引时使用以下 JSON（鉴权与路径见服务文档）：

```json
{
  "index_name": "external_demo",
  "schema": [
    {"field":"body","field_type":"Text"},
    {"field":"vector","field_type":"Json","index_vector":true}
  ],
  "vector":{"dimensions":3}
}
```

向 `/api/v1/index/{id}/doc` POST `{"body":"example text","vector":[1,0,0]}`，再 PATCH `/api/v1/index/{id}` 提交。向 `/api/v1/index/{id}/query` POST：

```json
{"query":"example","query_vector":[1,0,0]}
```

仅文本执行全文检索，仅 `query_vector` 执行向量检索，两者都有执行混合检索。空对象 `{}` 浏览文档；查询默认包含未提交写入，持久化仍需提交。写入必须包含所有已配置的向量字段。二进制向量字段仍以 base64 字符串放在 JSON 中，与已删除的 v2 二进制查询协议无关。

## 性能基线

Linux 下使用独立临时目录启动 release 服务，通过本机 HTTP 长连接测量；只需为压测脚本安装 NumPy：

```sh
cargo build --release -p seekpower_server
python3 -m pip install numpy
python3 scripts/performance_baseline.py --output /tmp/seekpower-baseline.json
```

默认固定种子生成 20,000 篇合成文档、128 维向量和 64 个查询，测全文/向量/混合检索，并发 1、4，每组预热后运行 5 秒、重复 3 次。使用默认计数和正文返回，Top-10；向量采用 HTTP 默认的全簇检索。每个连接由独立进程驱动，压测端与服务共享机器资源。

JSON 记录每轮 QPS、HTTP P50/P95/P99、服务端/压测端 CPU、服务端采样 RSS、索引磁盘占用，以及单次批量写入和提交耗时。汇总值为各轮指标的中位数；向量 Recall@10 对照独立 float64 全量余弦计算，全文核对匹配计数。请求失败会中止，不产生成功结果文件。

[首版实测数据](benchmarks/http-baseline-2026-10-15.json) 包含源码与二进制哈希、数据哈希、配置和环境。结果用于同环境版本对比：属于合成数据的热缓存、闭环负载，不是容量上限；未衡量真实语料的全文/混合相关性、冷缓存、过滤和混合读写。提交耗时表示 API 完成时间，不验证崩溃恢复或持久化保证。`--help` 可查看数据量、维度、时长和并发等覆盖参数；输出文件不会被覆盖。

真实语料使用完整 [BEIR SciFact](https://github.com/beir-cellar/beir)：5,183 篇文档、300 条测试查询。固定 `intfloat/multilingual-e5-small` 模型版本，CPU ONNX FP32 推理、384 维、最长 512 token、带掩码均值池化和 L2 归一化；查询/文档分别加 `query: `/`passage: ` 前缀。向量预先生成，耗时单独记录：

```sh
python3 -m pip install numpy==2.2.6 onnxruntime==1.23.2 tokenizers==0.22.2
python3 scripts/prepare_retrieval.py --cache /tmp/seekpower-model --output /tmp/seekpower-scifact
python3 scripts/performance_baseline.py --dataset /tmp/seekpower-scifact --output /tmp/seekpower-scifact-result.json
```

正文合并标题和文本，全文/混合请求显式使用 `operator=or`；其他 HTTP 配置不变。性能测 Top-10，相关性另取 Top-100，按人工标注计算 nDCG@10、Recall@100、MRR@10；最近邻 Recall@10 单独列出。[SciFact 实测数据](benchmarks/scifact-baseline-2026-10-15.json) 保留模型和输入哈希。SciFact 结果不代表百万级索引表现。模型、数据和向量保留在指定缓存/输出目录，不提交仓库。

MIRACL 中文使用固定版本的官方完整语料（4,934,368 段）、393 条开发集查询和 3,928 条正负相关性标注，自动选择 `UnicodeAlphanumericZH` 分词。完整语料准备、嵌入和压测命令：

```sh
python3 scripts/prepare_retrieval.py --dataset miracl-zh --cache /tmp/miracl-cache --output /tmp/miracl-full
python3 scripts/performance_baseline.py --dataset /tmp/miracl-full --commit-every 20 --output /tmp/miracl-full-result.json
```

准备程序按块生成向量并记录进度；中断后使用相同命令续跑，校验输入和模型一致性。完成的输出不会覆盖。压测使用磁盘请求缓存、内存映射向量和分块精确余弦参考，不会构造全量查询×文档矩阵；`--commit-every 20` 每 20 个批次提交一次以限制未提交数据，耗时计入写入阶段。服务端自身的资源需求仍随数据量增长。

资源有限时，显式加 `--documents 10000` 生成子集：保留全部开发集正负标注文档，并从全语料按固定 SHA-256 顺序补足背景文档。报告标记 `full_corpus=false`，这种筛选降低检索难度，不能与官方全量成绩比较。仅准备全文数据可用 `--text-only`，压测对应使用 `--modes lexical`。`--max-rss-mib` 可设置写入时的服务端内存保护阈值，超限失败不会输出成功报告。

本机实测分开保存：

- [493 万条全文结果（末尾分词修复前）](benchmarks/miracl-zh-full-lexical-2026-10-16.json)：4 核 CPU 亲和性、每 100 批提交一次，完整 393 条查询。该配置并发 4 的 QPS 约 145、P95 约 58.6 ms；nDCG@10 为 0.1467、Recall@10 为 20.52%、Recall@100 为 46.66%。
- [一万条子集三模式结果](benchmarks/miracl-zh-10k-baseline-2026-10-16.json)：默认 CPU 亲和性，保留全部 3,786 篇被标注文档并补足背景样本。该子集的纯向量 nDCG@10 为 0.7177，混合为 0.6391，不能据此推断全量结果。
- [未完成的尝试](benchmarks/miracl-zh-full-resource-attempt-2026-10-16.json)：默认多分片运行在约 96 万条触发主动 RSS 保护；第一次 4 核运行被执行环境中断。它们不参与成功结果统计。全量向量/混合尚未实测。

复现本次完整全文配置（将 `0-3` 换成机器实际可用的四个 CPU）：

```sh
python3 scripts/prepare_retrieval.py --dataset miracl-zh --text-only --cache /tmp/miracl-cache --output /tmp/miracl-text
taskset -c 0-3 python3 scripts/performance_baseline.py --dataset /tmp/miracl-text --modes lexical --commit-every 100 --max-rss-mib 6500 --output /tmp/miracl-text-result.json
```

人工标注指标包含 nDCG@10、Recall@10、Recall@100、MRR@10，与向量最近邻 Recall 分开记录。CI 只运行指标、分块参考检索和流式加载的轻量测试，不下载语料或执行性能门槛判断。

末尾分词修复后的独立简繁实验使用同一二进制、完整语料和上述四核配置，分别重建索引。两组均以 `body` 保存原文、仅索引不返回的 `search_body`；`identity` 原样输入，`t2s` 在客户端用固定 OpenCC 1.4.2 归一化文档和查询。未更换分词器或修改引擎默认值。

```sh
python3 -m pip install OpenCC==1.4.2
for mode in identity t2s; do
  taskset -c 0-3 python3 scripts/performance_baseline.py --dataset /tmp/miracl-text --modes lexical --commit-every 100 --max-rss-mib 6500 --text-normalization "$mode" --output "/tmp/miracl-$mode.json"
done
```

| 指标 | [原样对照](benchmarks/miracl-zh-fixed-identity-2026-10-16.json) | [繁转简](benchmarks/miracl-zh-fixed-t2s-2026-10-16.json) |
| --- | ---: | ---: |
| nDCG@10 | 0.1467 | 0.1618 |
| Recall@10 | 20.52% | 23.46% |
| Recall@100 | 46.66% | 53.72% |
| MRR@10 | 0.1689 | 0.1875 |
| 并发 1 QPS / P95 ms | 108.6 / 18.59 | 118.7 / 17.61 |
| 并发 4 QPS / P95 ms | 136.2 / 64.51 | 143.9 / 59.56 |
| 写入与提交秒数 | 545.34 | 679.88 |
| 逻辑索引 GiB / 服务端峰值 RSS GiB | 2.745 / 5.037 | 2.621 / 4.920 |

两组共 7,459 次计时请求，零错误。3,197,639 段（64.80%）发生转换，393 条原始查询均未改变，因此本数据集的差异来自文档侧归一化。Recall@100 增加 7.06 个百分点：74 条查询改善、36 条退化；nDCG@10 均值提升 10.32%，但 69 条改善、76 条退化，不能认为所有查询都受益。查询延迟存在单轮波动，写入各测一次，不据此宣称稳定的性能提升。

转换预先完成，不计入 HTTP 延迟或写入时间；完整文档转换累计 43.19 秒，查询转换 P95 为 0.00221 ms，整个输入准备阶段为 43.03 / 91.85 秒。它们是客户端测量，不代表接入引擎后的开销。两组都额外传输 `search_body`，写入吞吐不能直接与旧版单字段基线比较。原文返回已通过 HTTP 检查；OpenCC 配置、词典/实现哈希、逐查询指标均保存在 JSON 中。

独立分词器实验固定上述 `t2s`、全文字段和四核配置。对照使用 `3693bbf`，HMM 开启组使用 `54499af`，关闭组使用 `46dc886`：均为 `jieba-rs = 0.11.0`、普通 `cut` 模式、默认词典，后一组仅关闭 HMM。两者词典的 349,045 个词及词频完全相同；ASCII 片段继续原样保留。此实验分支临时替换 `UnicodeAlphanumericZH` 的后端，必须重建索引，不能与旧分词器混用已有索引。

| 指标 | [当前分词器](benchmarks/miracl-zh-tokenizer-reference-2026-10-16.json) | [jieba HMM 开](benchmarks/miracl-zh-tokenizer-jieba-2026-10-16.json) | [jieba HMM 关](benchmarks/miracl-zh-tokenizer-jieba-no-hmm-2026-10-16.json) |
| --- | ---: | ---: | ---: |
| nDCG@10 | 0.1618 | 0.1515 | 0.1606 |
| Recall@10 | 23.46% | 21.94% | 23.00% |
| Recall@100 | 53.72% | 52.26% | 54.12% |
| MRR@10 | 0.1875 | 0.1734 | 0.1855 |
| 并发 1 QPS / P95 ms | 109.3 / 21.41 | 95.1 / 23.35 | 119.9 / 16.96 |
| 并发 4 QPS / P95 ms | 134.2 / 61.75 | 139.6 / 60.56 | 137.3 / 62.95 |
| 写入与提交秒数 | 575.82 | 471.26 | 439.83 |
| 逻辑索引 GiB | 2.621 | 2.716 | 2.608 |
| 服务端峰值 RSS GiB | 4.893 | 4.962 | 4.803 |

完整 4,934,368 段、393 条查询，三组共 18 轮、10,774 次计时请求零错误，归一化请求体、查询和字段配置的哈希/内容一致。对照组逐查询质量与上一轮完全一致。HMM 开启组相对原分词器的 nDCG@10 有 45 条改善、65 条退化；Recall@100 有 16 条改善、22 条退化。对逐查询差值做配对 bootstrap（种子 20261016，10,000 次），nDCG@10 差值 95% 百分位区间为 [-0.01979, -0.00099]；Recall@100 为 [-0.03246, 0.00238]。本测试集不支持直接替换为该配置。

关闭 HMM 后，相对开启组，nDCG@10 增加 0.00908（55 条改善、34 条退化），Recall@100 增加 1.86 个百分点（18 条改善、6 条退化）；同一配对 bootstrap 方法的差值区间分别为 [0.00089, 0.01730]、[0.00722, 0.03221]。相对原分词器，nDCG@10 差值 -0.00125（41 条改善、34 条退化），Recall@100 差值 +0.00403（15 条改善、9 条退化），区间分别为 [-0.00660, 0.00421]、[-0.00869, 0.01724]，不能据此认定优于原分词器或证明等效。HMM 关闭组值得继续做集成验证，当前仍保持实验状态。

[三组同进程分词对照](benchmarks/miracl-zh-segmentation-hmm-ablation-2026-10-16.json)使用相同输入，一次预热、三轮轮换顺序计时：原分词器 / HMM 开 / HMM 关处理 10,000 段分别为 0.663 / 0.216 / 0.204 秒，393 条查询合计 0.796 / 0.280 / 0.313 ms。与下面历史计时存在环境波动，只比较本轮内三组；查询总耗时很短，不宣称关闭 HMM 稳定更快。HMM 开关改变 43 条查询切词；关闭后 390 条与原分词器相同，余下三条均涉及“哪种”的拆分。“魏博现”查询的 Recall@100 从 0 恢复为 1，“李荣浩”和“周深”查询从 0 变为 1/3，但两者前十名仍未命中。开关同时影响文档与查询，不能把质量变化全部归因于查询端。

[独立分词计时与查询切词对照](benchmarks/miracl-zh-segmentation-2026-10-16.json)：同一批已繁转简的前 10,000 段文本，分词中位耗时 1.728 / 0.633 秒；393 条查询合计 4.408 / 1.313 ms。排除初始化、大小写转换和片段切分，保留两组相同的 ASCII 旁路及字符串输出分配；一次预热、三轮交替顺序计时。这是局部分词成本，不是端到端延迟，也不是子集相关性评估。44 条查询切词变化；查询切词不变的 349 条也有质量变化，不能把全部差异归因于查询端 HMM。

复现完整检索实验：分别检出上述三个提交、各自构建 release，使用上一节的完整全文命令，三次均设 `--text-normalization t2s` 并使用不同输出路径。复现分词计时：在 `46dc886` 运行以下命令复现三组计时（将语料目录改为实际路径；`54499af` 为历史两组计时）；输出未附加的输入/词典哈希和环境记录可在已提交结果中核对。

```sh
python3 - <<'PYCODE' > /tmp/tokenizer-samples.jsonl
import itertools, json, pathlib, opencc
root = pathlib.Path('/tmp/miracl-text')
convert = opencc.OpenCC('t2s.json').convert
for kind, name, limit in [('query', 'queries.jsonl', 393), ('document', 'corpus.jsonl', 10000)]:
    with (root / name).open() as stream:
        for line in itertools.islice(stream, limit):
            row = json.loads(line)
            text = row['text'] if kind == 'query' else (row.get('title', '') + '\n' + row['text']).strip()
            print(json.dumps({'kind': kind, 'id': row['_id'], 'text': convert(text)}, ensure_ascii=False))
PYCODE
cargo build --release -p seekpower --example compare_zh_tokenizers
taskset -c 0-3 target/release/examples/compare_zh_tokenizers /tmp/tokenizer-samples.jsonl > /tmp/tokenizer-segmentation.json
```

写入各测一次，查询单轮有明显波动，不宣称稳定的整体性能提升。首次对照在约 190 万条后因执行环境连接中断而未完成，已另行完整补跑；中断轮次不计入结果。词项/位置、标点、中英混排、提交重开后的短语检索及 HTTP 原文返回检查通过；另发现现有 `中国 -北京` 排除行为在两个后端均不正确，本轮未修改查询语法。当前无高亮实现。jieba 默认词典间接引入 `zstd`，触发现有依赖裁剪 CI 检查；检查保留，实验不作为生产合入方案。

## 开发与文档

- [Rust 示例](docs/library-examples.md)、[元数据过滤与排序](FACETED_SEARCH.md)、[N-gram 配置](NGRAM_SEARCH.md)。本地执行 `cargo doc --workspace --no-deps` 生成与当前源码一致的 Rust 文档。
- [HTTP 请求示例](seekpower_server/test_api.rest)。运行 `target/debug/seekpower_server generate_openapi=true`，在可执行文件目录生成 `openapi.json` 和 `openapi.yml`；CI 也会提供生成文件。完整路由见功能清单。
- 运行 `cargo test --workspace --all-targets`；HTTP 冒烟使用 `python3 scripts/http_smoke.py --server target/debug/seekpower_server`。浏览器回归需安装 Python Playwright/Chromium，并设置 `SEEKPOWER_BROWSER_TEST=1`。
- 核心默认 features：`zh`（中文分词）、`vb`（结果诊断信息）；哈希实现固定使用 AHash。HTTP 对象与 OpenAPI 定义均在服务工程，核心通过 `log` 发出日志，由调用方配置输出。具体见 [Cargo.toml](seekpower/Cargo.toml)。
- [验证记录](VALIDATION.md) 保存实际测试和历史失败；[CHANGELOG](CHANGELOG.md) 保留上游历史，不作为当前功能列表。

本项目使用 [Apache-2.0 许可证](LICENSE)。
