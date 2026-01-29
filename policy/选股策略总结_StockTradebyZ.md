# StockTradebyZ 选股策略总结

> 基于 **StockTradebyZ** 项目 `Selector.py`、`select_stock.py` 及 `configs.json` 整理。

---

## 一、项目与运行方式

- **选股入口**：`select_stock.py`，通过 `--config` 指定 JSON 配置（默认 `configs.json`）。
- **策略实现**：`Selector.py` 中多个 `*Selector` 类，统一接口 `select(date, data) -> List[str]`。
- **配置格式**：`configs.json` 的 `selectors` 数组，每项含 `class`、`alias`、`activate`、`params`。

---

## 二、通用基础设施

### 2.1 技术指标

| 指标 | 说明 |
|------|------|
| **KDJ** | `compute_kdj(df, n=9)`：K、D、J，RSV 基于 N 日高低收。 |
| **BBI** | `compute_bbi(df)`：多空指标，(MA3+MA6+MA12+MA24)/4。 |
| **RSV** | `compute_rsv(df, n)`：N 日 RSV，C 用 close 的 HHV，L 用 low 的 LLV。 |
| **DIF** | `compute_dif(df, fast=12, slow=26)`：MACD 的 DIF = EMA12 - EMA26。 |
| **知行线** | `compute_zx_lines(df)`：ZXDQ = EMA(EMA(C,10),10)；ZXDKX = (MA14+MA28+MA57+MA114)/4。 |

### 2.2 统一当日过滤 `passes_day_constraints_today`

所有战法**共用**的当日过滤：

1. 当日相对前一日涨跌幅 **< 2%**（绝对值）；
2. 当日振幅 `(High-Low)/Low` **< 7%**。

### 2.3 知行条件 `zx_condition_at_positions`

在指定日检查：

- **收盘 > 长期线 ZXDKX**（可关）；
- **短期线 ZXDQ > 长期线 ZXDKX**（可关）。

不同战法对「当日」或历史某日是否满足上述条件有不同要求。

### 2.4 其他工具

- **BBI 整体上升**：`bbi_deriv_uptrend`，在滑动窗口内对归一化 BBI 做一阶差分，按 `q_threshold` 分位数判断是否整体上升，允许一定回撤。
- **有效上穿 MA**：`last_valid_ma_cross_up(close, ma, lookback_n)`，查找最近一次 `close[T-1]<MA[T-1]` 且 `close[T]≥MA[T]` 的 T。
- **峰值检测**：`_find_peaks`，基于 `scipy.signal.find_peaks`，用于 Peaks+KDJ 类策略。

---

## 三、六大选股策略详解

### 3.1 BBIKDJSelector（少妇战法）

**思路**：BBI 趋势向上 + KDJ 超卖区 + 价格站上 MA60 且有过上穿 + MACD 多头 + 知行多头。

**过滤逻辑**：

1. 通过 **当日约束**（涨跌幅、振幅）；
2. 最近 `max_window` 根 K 线内，收盘价波动幅度 ≤ `price_range_pct`；
3. **BBI 整体上升**（`bbi_deriv_uptrend`，允许 `bbi_q_threshold` 比例回撤）；
4. **KDJ**：当日 `J < j_threshold` **或** J ≤ 近 `max_window` 根 J 的 `j_q_threshold` 分位；
5. 收盘 **> MA60**，且最近 `max_window` 内存在 **有效上穿 MA60**；
6. **DIF > 0**；
7. 当日满足 **知行条件**（收盘>长期线，短期>长期）。

**典型参数**（configs）：`j_threshold=-10`，`bbi_min_window=10`，`max_window=60`，`price_range_pct=1`，`bbi_q_threshold=0.2`，`j_q_threshold=0.10`。

---

### 3.2 SuperB1Selector（SuperB1 战法）

**思路**：历史上某日 `t_m` 满足 BBIKDJ，随后至今盘整，当日下跌且 J 极低，寻找「回调低吸」机会。

**过滤逻辑**：

1. 通过 **当日约束**；
2. 在最近 `lookback_n` 日内，存在一日 `t_m` 满足 **BBIKDJSelector** 的全部条件；
3. 在 `t_m` 当日满足 **知行条件**（收盘>长期线，短期>长期）；
4. **盘整**：`[t_m, date-1]` 收盘价波动率 ≤ `close_vol_pct`；
5. **当日下跌**：`(close_{date-1} - close_date) / close_{date-1}` ≥ `price_drop_pct`；
6. **J 极低**：`J < j_threshold` 或 J ≤ 近 `lookback_n` 日 J 的 `j_q_threshold` 分位；
7. 当日仅要求 **短期线 > 长期线**（不强制收盘>长期线）。

**典型参数**：`lookback_n=10`，`close_vol_pct=0.02`，`price_drop_pct=0.02`，`j_threshold=10`，`j_q_threshold=0.10`，且传入嵌套的 `B1_params` 用于 BBIKDJ。

---

### 3.3 BBIShortLongSelector（补票战法）

**思路**：BBI 上升 + 短长期 RSV 配合（短期先高后低再高，长期持续高位）+ DIF>0 + 知行多头。

**过滤逻辑**：

1. 通过 **当日约束**；
2. **BBI 整体上升**（`bbi_deriv_uptrend`）；
3. **RSV 短期/长期**：
   - `RSV_short`(n_short)，`RSV_long`(n_long)；
   - 最近 `m` 日：**长期 RSV 全部 ≥** `upper_rsv_threshold`；
   - 短期 RSV：先出现 ≥ `upper_rsv_threshold`，之后某日 **<** `lower_rsv_threshold`，且**最后一天**再次 ≥ `upper_rsv_threshold`；
4. **DIF > 0**；
5. 当日 **知行条件**（收盘>长期线，短期>长期）。

**典型参数**：`n_short=5`，`n_long=21`，`m=5`，`bbi_min_window=2`，`max_window=120`，`upper_rsv_threshold=75`，`lower_rsv_threshold=25`。

---

### 3.4 PeakKDJSelector（填坑战法）

**思路**：基于 **峰值**（oc_max = max(open,close)）结构 + KDJ 超卖 + 知行多头，捕捉「填坑」形态。

**过滤逻辑**：

1. 通过 **当日约束**；
2. 对 `oc_max` 做 **峰值检测**（distance=6, prominence=0.5），至少 2 个峰，且当日不是峰；
3. 取**最近两个峰** `peak_t`、`peak_{t-n}`：要求 `peak_t > peak_{t-n}`，且 `peak_{t-n}` 高于其与 `peak_t` 之间最低收盘价的 `(1+gap_threshold)` 倍；
4. 当日收盘相对 `peak_{t-n}` 的**波动率** ≤ `fluc_threshold`；
5. **KDJ**：J 绝对低或相对低（同 BBIKDJ 的 J 条件）；
6. 当日 **知行条件**（收盘>长期线，短期>长期）。

**典型参数**：`j_threshold=10`，`max_window=120`，`fluc_threshold=0.03`，`gap_threshold=0.2`，`j_q_threshold=0.10`。

---

### 3.5 MA60CrossVolumeWaveSelector（上穿 60 放量战法）

**思路**：J 低 + 近期有 MA60 有效上穿 + 上穿后「上涨波段」明显放量 + MA60 斜率向上 + 知行多头。

**过滤逻辑**：

1. 通过 **当日约束**；
2. **J 低**：`J < j_threshold` 或 J ≤ 近 `max_window` 根 J 的 `j_q_threshold` 分位；
3. 收盘 **> MA60**，且在 `lookback_n` 内存在 **有效上穿 MA60** 的 T；
4. 定义 **上涨波段**：`[T, Tmax]`，其中 Tmax 为 [T, today] 内最高价所在日；等长前置窗口为 `[T-wave_len, T-1]` 的截断；
5. **放量**：上涨波段日均成交量 ≥ 前置窗口日均成交量 × `vol_multiple`；
6. 最近 **`ma60_slope_days`** 个交易日 **MA60 线性回归斜率 > 0**；
7. 当日 **知行条件**（收盘>长期线，短期>长期）。

**典型参数**：`lookback_n=25`，`vol_multiple=1.8`，`j_threshold=15`，`ma60_slope_days=5`，`max_window=120`。

---

### 3.6 BigBullishVolumeSelector（暴力 K 战法）

**思路**：长阳 + 上影线不过长 + 明显放量 + 价格尚在 ZXDQ 下方（未远离短线），追强势启动。

**过滤逻辑**：

1. 当日 **收阳**（可选，`require_bullish_close`）：`close >= open`；
2. **长阳**：`(close - prev_close) / prev_close > up_pct_threshold`；
3. **上影线**：`(high - max(open,close)) / max(open,close) < upper_wick_pct_max`；
4. **放量**：当日成交量 **≥** 前 `vol_lookback_n` 日均量 × `vol_multiple`（可选忽略 0 成交量）；
5. **偏离**：`close < ZXDQ × close_lt_zxdq_mult`（尚未大幅脱离短线）。

**典型参数**：`up_pct_threshold=0.06`，`upper_wick_pct_max=0.02`，`vol_lookback_n=20`，`vol_multiple=2.5`，`require_bullish_close=true`，`close_lt_zxdq_mult=1.15`。

---

## 四、策略汇总表

| 策略类 | 别名 | 核心逻辑概要 |
|--------|------|--------------|
| **BBIKDJSelector** | 少妇战法 | BBI↑ + KDJ 超卖 + MA60 上穿 + DIF>0 + 知行 |
| **SuperB1Selector** | SuperB1 战法 | 历史 BBIKDJ 日 → 盘整 → 当日跌+J 极低 + 知行(松) |
| **BBIShortLongSelector** | 补票战法 | BBI↑ + RSV 短/长组合 + DIF>0 + 知行 |
| **PeakKDJSelector** | 填坑战法 | 双峰结构 + 波动约束 + KDJ 低 + 知行 |
| **MA60CrossVolumeWaveSelector** | 上穿 60 放量战法 | J 低 + MA60 上穿 + 上涨波段放量 + MA60 斜率↑ + 知行 |
| **BigBullishVolumeSelector** | 暴力 K 战法 | 长阳 + 缩上影 + 放量 + close<ZXDQ×系数 |

---

## 五、辅助模块（非选股策略）

- **SectorShift（SectorShift.py）**：按 **J 值 < 阈值** 选股，并统计**行业分布**（`compute_j_industry_distribution`），可导出 Excel。用于观察「超卖」股票的行业结构。
- **find_stock_by_price_concurrent**：按**历史价格**（收盘/最高/最低）及时间区间**并发**筛选股票，属工具型，非因子选股。

---

## 六、使用与扩展

- **运行选股**：  
  `python select_stock.py --data-dir ./data --config ./configs.json [--date YYYY-MM-DD] [--tickers all|code1,code2,...]`
- **启用/停用**：在 `configs.json` 中对应 selector 的 `"activate": true/false`。
- **扩展新策略**：在 `Selector.py` 中新增 `XxxSelector`，实现 `select(date, data) -> List[str]` 和内部 `_passes_filters(hist)`，在 `configs.json` 的 `selectors` 中增加一项即可。

---

## 七、小结

StockTradebyZ 的选股体系以**技术指标 + 形态/量价**为主，共用 **KDJ、BBI、DIF、知行线** 等指标，以及**统一当日过滤**与**知行条件**。六大战法分别侧重：趋势+超卖（少妇）、回调低吸（SuperB1）、RSV 节奏（补票）、峰谷形态（填坑）、均线放量（上穿 60 放量）、强势启动（暴力 K）。配置驱动、易扩展，可与 QuantArt_V1 的 `Selector`、`select_stock` 等模块对接或复用部分逻辑。
