# 工具安装&卸载
## 🔧 工具安装
✅ 环境要求

**Python 版本**：仅支持 Python **3.10** 或 **3.11**

不支持 Python 3.9 及以下，也不兼容 3.12 及以上版本

**推荐使用 Conda 管理环境**，以避免依赖冲突
```shell
conda create --name ais_bench python=3.10 -y
conda activate ais_bench
```

📦 安装方式（源码安装）

AISBench 当前仅提供源码安装方式，请确保安装环境联网：
```shell
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./ --use-pep517
```
该命令会自动安装核心依赖。
执行`ais_bench -h`，如果打印出AISBench评测工具的所有命令行的帮助信息，说明安装成功

⚙️ 服务化框架支持（可选）

若需评估服务化模型（如 vLLM、Triton 等），需额外安装相关依赖：
```shell
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```
🔗 Berkeley Function Calling Leaderboard (BFCL) 测评支持

```shell
pip3 install -r requirements/bfcl_dependencies.txt --no-deps
```

**重要提示**：由于 `bfcl_eval` 会自动安装 `pathlib` 库，而 Python 3.5+ 环境已内置该库，为避免版本冲突，请务必使用 `--no-deps` 参数跳过额外依赖的自动安装。

## ❌ 工具卸载
如需卸载 AISBench Benchmark，可执行以下命令：
```shell
pip3 uninstall ais_bench_benchmark
```