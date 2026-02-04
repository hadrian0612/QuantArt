# QuantArt_V1

基于 AkShare / AData / Baostock / Qlib 的 A 股 / 港股 / 美股 K 线数据与多策略选股小框架：数据拉取、增量更新、策略运行、结果汇总。

## 环境与依赖

### Python 版本

| 用途 | Python 版本 |
|------|-------------|
| **核心 + AkShare + AData** | **3.8+**（推荐 3.10 / 3.11 / 3.12） |
| **Qlib 增量脚本** | **仅 3.8～3.12**（pyqlib 暂无 3.13 wheel，需单独环境或跳过） |

若使用 Python 3.13，可正常跑选股、AkShare 与 AData 增量更新；仅 `incremental_qlib_update.py` 需在 3.11/3.12 的 venv 或 conda 中运行。

### 依赖包

**核心（选股 + 任意一种数据源）：**

- `pandas`
- `numpy`
- `scipy`
- `pyyaml`
- `tqdm`

**按数据源二选一或三选一：**

| 数据源 | 包名 | 说明 |
|--------|------|------|
| **AkShare** | `akshare` | 全量/增量 K 线、股票列表，支持 A/HK/US |
| **AData** | `adata` | 仅 A 股增量，与现有 CSV 格式兼容 |
| **Baostock** | `baostock` | 仅 A 股增量，免费、稳定，与现有 CSV 格式兼容 |
| **Qlib** | `pyqlib` | 仅 A 股增量，需 3.8～3.12 且需自备 Qlib 数据目录 |

**安装示例：**

```bash
# 最小可运行（选股 + AkShare 增量）
pip install pandas numpy scipy pyyaml tqdm akshare

# 若使用 AData 增量（A 股）
pip install adata

# 若使用 Baostock 增量（A 股）
pip install baostock

# 若使用 Qlib 增量（需 Python 3.8～3.12 环境）
pip install pyqlib
```

## 目录结构

```
QuantArt_V1/
├── config/              # 选股策略 YAML 配置（每个策略一个文件）
├── data/                # 行情数据
│   ├── A-shares/        # A 股日线 CSV（代码.csv）
│   ├── HK-stocks/       # 港股
│   └── US-stocks/       # 美股
├── logs/                # 运行日志
├── results/             # 选股结果（按时间戳分目录，含各策略 CSV + 汇总 CSV）
├── utils/
│   └── Selector.py      # 选股策略实现（BBIKDJ、SuperB1、PeakKDJ 等）
├── akshare_fetch_stocklist.py   # 拉取 A 股股票列表
├── akshare_fetch_stocklist_hk.py
├── akshare_fetch_stocklist_us.py
├── akshare_fetch_kline.py       # 按股票列表全量拉取 K 线（AkShare）
├── incremental_update_kline.py  # 增量更新 K 线（AkShare，A/HK/US）
├── incremental_adata_update.py   # 增量更新 K 线（AData，仅 A 股）
├── incremental_baostock_update.py # 增量更新 K 线（Baostock，仅 A 股）
├── incremental_qlib_update.py   # 增量更新 K 线（Qlib，仅 A 股，需 3.8～3.12）
└── select_stock.py              # 运行选股策略并写结果
```

## 使用流程

### 1. 股票列表（可选，若已有列表可跳过）

```bash
# A 股列表 → 默认 data/stocklist.csv
python akshare_fetch_stocklist.py
python akshare_fetch_stocklist_hk.py   # data/stocklist_hk.csv
python akshare_fetch_stocklist_us.py   # data/stocklist_us.csv
```

### 2. K 线数据

```bash
# 全量拉取（按列表里的代码拉日线到 data/A-shares 等，AkShare）
python akshare_fetch_kline.py --market A --stocklist ./data/stocklist.csv

# 增量更新（推荐日常使用，任选一种数据源，输出目录与 CSV 格式一致）
python incremental_update_kline.py --data-dir ./data --end today --default-start 20250101 --market A --workers 4
python incremental_adata_update.py --data-dir ./data --end today --default-start 20250101 --workers 4   # 仅 A 股
python incremental_baostock_update.py --data-dir ./data --end today --default-start 20250101 --workers 4 # 仅 A 股（Baostock）
# Qlib 需 Python 3.8～3.12 且已准备 qlib 数据目录：
# python incremental_qlib_update.py --data-dir ./data --end today --qlib-provider-uri "~/.qlib/qlib_data/cn_data"
```

### 3. 选股

```bash
# 使用 config 目录下全部策略，A 股，交易日取数据最新日期
python select_stock.py --data-dir ./data --market A

# 指定交易日
python select_stock.py --data-dir ./data --market A --date 2025-01-27

# 只跑指定配置文件
python select_stock.py --data-dir ./data --market A --config ./config/BBIKDJSelector_config.yml
```

选股结果会写入 `results/<时间戳>/`：

- 每个策略一个子目录，内含 `selected_stocks.csv`
- 汇总表 `summary_by_market_cap.csv`：所有被选中股票、按市值排序、备注被哪些策略选中

## 策略配置说明

`config/*.yml` 中每个文件对应一个策略，格式示例：

```yaml
class: BBIKDJSelector    # utils.Selector 中的类名
alias: 少妇战法          # 结果目录与汇总表中的策略名
activate: true
params:                  # 传给策略构造函数的参数
  j_threshold: 10
  bbi_min_window: 10
  # ...
```

未指定 `--config` 时，会扫描 `config/` 下所有 `*.yml`（文件名含 `AllSelectors` 的会被忽略），合并后依次运行。

## 注意事项

- 拉取数据时注意 AkShare 限频，脚本内带简单冷却与重试逻辑。
- 增量更新会按本地已有 CSV 的代码与日期范围补全，适合每日跑一次。
- 汇总 CSV 的市值列若为空，说明本地 K 线或股票列表中没有市值字段，排序会退化为“按写入顺序”。

## 写在最后

- 感谢z哥分享宝贵经验
- 感谢我的员工Cursor帮我Coding，每个月20刀，酷酷干活
- 独乐乐不如众乐乐，故而开源，有心人可以一起完善代码