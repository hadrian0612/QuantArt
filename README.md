# QuantArt_V1

基于 AkShare 的 A 股 / 港股 / 美股 K 线数据与多策略选股小框架：数据拉取、增量更新、策略运行、结果汇总。

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
├── akshare_fetch_kline.py       # 按股票列表全量拉取 K 线
├── incremental_update_kline.py  # 增量更新 K 线
└── select_stock.py              # 运行选股策略并写结果
```

## 依赖

- Python 3.8+
- `pandas`、`akshare`、`pyyaml`、`tqdm`

安装示例：`pip install pandas akshare pyyaml tqdm`

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
# 全量拉取（按列表里的代码拉日线到 data/A-shares 等）
python akshare_fetch_kline.py --market A --stocklist ./data/stocklist.csv

# 增量更新（推荐日常使用）
python incremental_update_kline.py --data-dir ./data --market A
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
