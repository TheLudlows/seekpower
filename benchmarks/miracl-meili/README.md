# 本地检索 benchmark

先读 [报告](REPORT.md)、[实测表](RESULTS.md) 和 [实验协议](PROTOCOL.md)。这是有容量失败和语义差异的诊断，不是完整同义性能排名。

生产版本：`6d28798a300bb947b2eb1d3e48e928d193103e17`。所有脚本仅使用 Python 标准库，Linux `taskset`、`/proc`；本轮未测试Windows benchmark。脚本必须从本仓库布局运行，生成文件在 `artifacts/benchmark/`。同一时刻只运行一个实验，禁止与编译/其他服务重叠。

## 准备

1. 用独立工作分支保留该生产版本，检查无生产代码差异。已有语料位于 `miracl-corpus-v1.0-zh/`；缺失时参考该目录README与 `scripts/download_miracl_zh.py`，以manifest固定revision/哈希为准。
2. `python3 benchmarks/miracl-meili/fetch_inputs.py` 获取固定官方二进制、许可证、topics/qrels并校验。该下载辅助脚本语法检查通过；本轮下载使用了等价的官方URL与独立SHA校验，未为测试它重新下载大二进制。
3. `python3 benchmarks/miracl-meili/prepare_data.py` 验证全部语料并生成嵌套子集；扫描约493万条。选样manifest记录所有哈希，不能换语料而复用结果。
4. `cargo build --release --locked -p anyseek --features server`，等待退出。Meilisearch版本必须是v1.55.0，下载SHA见协议；固定 `anyseek_server` 二进制哈希记录在 `artifacts/benchmark/environment.json`。不要在实验中覆盖二进制。

## 执行与复算

每次 `--tag` 必须是新目录；若已经存在脚本拒绝覆盖。两个服务正式HTTP接口使用一次性本地密钥和localhost端口，独立数据根随原始证据保留。

```bash
python3 benchmarks/miracl-meili/run.py --engine anyseek --profile restricted --tag smoke-a-new --smoke
python3 benchmarks/miracl-meili/run.py --engine meili --profile restricted --tag smoke-m-new --smoke
python3 benchmarks/miracl-meili/probes.py
python3 benchmarks/miracl-meili/run.py --engine anyseek --profile restricted --n 1000 --tag a-restricted-new
python3 benchmarks/miracl-meili/run.py --engine meili --profile restricted --n 1000 --tag m-restricted-new
python3 benchmarks/miracl-meili/run.py --engine anyseek --profile default --n 1000 --tag a-default-new
python3 benchmarks/miracl-meili/run.py --engine meili --profile default --n 1000 --tag m-default-new
```

预检失败必须保留，不重试调高限制。确认1000实际预算后才做5000；`sequence.py`保存了本轮后半段的固定顺序，遇anyseek5000失败停止，不是通用自动调参器。`probes.py`固定输出目录，已存在时同样拒绝重跑；可在全新独立副本复现。

写入混合负载用新tag，`--n 1000 --profile default --write-mix`。本轮实际tag为 `anyseek-default-write1000` 和 `meili-default-write1000`，结束后 `verify_updates.py` 独立重开两者数据根并核对100份原文/更新标记。脚本不重试失败写入；其他tag需显式修改核验输入路径。

```bash
python3 benchmarks/miracl-meili/analyze.py
python3 -m py_compile benchmarks/miracl-meili/*.py
```

`analyze.py`从原始请求重算分位数和错误率，输出 `artifacts/benchmark/summary.json` 及本目录 `RESULTS.md`。不能从成功请求的分位数省略错误率，也不能把Meilisearch无对应checkpoint的占位耗时作为比较数据。

## 原始证据

- `artifacts/benchmark/environment.json`：宿主、cgroup、CPU绑定、两个二进制哈希。
- `data/selection-manifest.json`：来源revision、40查询选择、相关文档覆盖、10分片与子集哈希。
- `raw/<tag>/settings.json`、`events.jsonl`：实际配置、命令、提交及服务端响应。
- `run.json`、`quality.json`、`mixed-unjudged.json`、`warm-*.jsonl`、`during-writes.jsonl`、`resources.json`：汇总及逐请求/采样数据。
- `raw/*-verify-updates/verification.json`：成功确认与实际持久化逐条对照。
- `evidence-sha256.json`：输入/脚本/日志/统计哈希索引；排除可变数据库、下载二进制和哈希索引自身。原始数据库和二进制仍保留。

本次仅本地实验；这些产物未上传远端。报告中的下一步是建议，未开始生产修复或新的性能专项。
