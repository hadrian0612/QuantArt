from __future__ import annotations

import argparse
import datetime as dt
import logging
import random
import sys
import time
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import os

import pandas as pd
import akshare as ak
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --------------------------- 全局锁（用于美股数据抓取，避免 py_mini_racer 多线程冲突） --------------------------- #
# py_mini_racer 在多线程环境下不稳定，需要串行化美股接口调用
_us_stock_lock = threading.Lock()

# --------------------------- 全局日志配置 --------------------------- #
LOG_FILE = Path("./logs/fetch_akshare.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("fetch_from_stocklist_akshare")

# --------------------------- 限流/封禁处理配置 --------------------------- #
COOLDOWN_SECS = 600
BAN_PATTERNS = (
    "访问频繁", "请稍后", "超过频率", "频繁访问",
    "too many requests", "429",
    "forbidden", "403",
    "max retries exceeded",
    "连接超时", "timeout", "连接失败"
)

def _looks_like_ip_ban(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return any(pat in msg for pat in BAN_PATTERNS)

class RateLimitError(RuntimeError):
    """表示命中限流/封禁，需要长时间冷却后重试。"""
    pass

def _cool_sleep(base_seconds: int) -> None:
    jitter = random.uniform(0.9, 1.2)
    sleep_s = max(1, int(base_seconds * jitter))
    logger.warning("疑似被限流/封禁，进入冷却期 %d 秒...", sleep_s)
    time.sleep(sleep_s)

# --------------------------- 历史K线（AkShare 日线） --------------------------- #

def _get_kline_akshare(code: str, start: str, end: str, market: str = "A") -> pd.DataFrame:
    """
    使用 AkShare 获取股票日线数据
    
    Args:
        code: 股票代码
            - A股: 6位数字，如 "000001"
            - 港股: 5位数字，如 "00700"
            - 美股: 股票代码，如 "AAPL"
        start: 起始日期 YYYYMMDD
        end: 结束日期 YYYYMMDD
        market: 市场类型 "A"(A股), "HK"(港股), "US"(美股)
    
    Returns:
        DataFrame with columns: date, open, close, high, low, volume
    """
    try:
        if market == "A":
            # A股：使用 stock_zh_a_hist 接口（前复权）
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq"  # 前复权
            )
        elif market == "HK":
            # 港股：尝试多种API接口
            df = None
            # 方法1: stock_hk_hist
            try:
                df = ak.stock_hk_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
            except:
                pass
            # 方法2: stock_hk_daily
            if df is None or df.empty:
                try:
                    df = ak.stock_hk_daily(symbol=code, adjust="qfq", start_date=start, end_date=end)
                except:
                    pass
            # 方法3: stock_hk_hist_min_em (如果前两个都失败)
            if df is None or df.empty:
                try:
                    # 注意：这个接口可能需要不同的参数格式
                    df = ak.stock_hk_hist_min_em(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
                except:
                    pass
            if df is None:
                raise ValueError(f"无法获取港股 {code} 的数据，所有API接口均失败")
        elif market == "US":
            # 美股：使用 stock_us_daily 接口
            # 注意：akshare的美股接口内部使用 py_mini_racer，多线程不安全，需要加锁
            with _us_stock_lock:
                # 添加小延迟，避免请求过快
                time.sleep(0.1)
                try:
                    df = ak.stock_us_daily(symbol=code, adjust="qfq", start_date=start, end_date=end)
                except:
                    # 备用方法：尝试不带日期参数（获取全部数据后过滤）
                    try:
                        df_all = ak.stock_us_daily(symbol=code, adjust="qfq")
                        if df_all is not None and not df_all.empty:
                            # 过滤日期范围
                            if "date" in df_all.columns or "日期" in df_all.columns:
                                date_col = "date" if "date" in df_all.columns else "日期"
                                df_all[date_col] = pd.to_datetime(df_all[date_col])
                                start_dt = pd.to_datetime(start)
                                end_dt = pd.to_datetime(end)
                                df = df_all[(df_all[date_col] >= start_dt) & (df_all[date_col] <= end_dt)]
                            else:
                                df = df_all
                    except Exception as e:
                        raise ValueError(f"无法获取美股 {code} 的数据，API调用失败: {str(e)}")
        else:
            raise ValueError(f"不支持的市场类型: {market}")
    except Exception as e:
        if _looks_like_ip_ban(e):
            raise RateLimitError(str(e)) from e
        # 保留原始异常信息，让上层处理
        raise

    if df is None or df.empty:
        return pd.DataFrame()

    # AkShare 返回的列名可能是中文或英文，需要统一映射
    # A股列名：日期、开盘、收盘、最高、最低、成交量
    # 港股/美股可能使用英文列名：date, open, close, high, low, volume
    column_mapping = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        # 英文列名（如果已经是英文则保持不变）
        "date": "date",
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume"
    }
    
    # 检查并重命名列
    rename_dict = {}
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and old_col != new_col:
            rename_dict[old_col] = new_col
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # 检查必要的列是否存在
    required_cols = ["date", "open", "close", "high", "low", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"{code} ({market}) 数据缺少列: {missing_cols}，可用列: {df.columns.tolist()}")
        return pd.DataFrame()
    
    # 选择需要的列
    df = df[required_cols].copy()
    
    # 数据类型转换
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    return df.sort_values("date").reset_index(drop=True)

def validate(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("存在缺失日期！")
    if (df["date"] > pd.Timestamp.today()).any():
        raise ValueError("数据包含未来日期，可能抓取错误！")
    return df

# --------------------------- 读取 stocklist.csv & 过滤板块 --------------------------- #

def _filter_by_boards_stocklist(df: pd.DataFrame, exclude_boards: set[str]) -> pd.DataFrame:
    """
    exclude_boards 子集：{'gem','star','bj'}
    - gem  : 创业板 300/301（.SZ）
    - star : 科创板 688（.SH）
    - bj   : 北交所（.BJ 或 4/8 开头）
    """
    code = df["symbol"].astype(str)
    ts_code = df["ts_code"].astype(str).str.upper()
    mask = pd.Series(True, index=df.index)

    if "gem" in exclude_boards:
        mask &= ~code.str.startswith(("300", "301"))
    if "star" in exclude_boards:
        mask &= ~code.str.startswith(("688",))
    if "bj" in exclude_boards:
        mask &= ~(ts_code.str.endswith(".BJ") | code.str.startswith(("4", "8")))

    return df[mask].copy()

def load_codes_from_stocklist(stocklist_csv: Path, exclude_boards: set[str]) -> List[str]:
    """加载A股股票列表"""
    df = pd.read_csv(stocklist_csv)    
    df = _filter_by_boards_stocklist(df, exclude_boards)
    codes = df["symbol"].astype(str).str.zfill(6).tolist()
    codes = list(dict.fromkeys(codes))  # 去重保持顺序
    logger.info("从 %s 读取到 %d 只A股（排除板块：%s）",
                stocklist_csv, len(codes), ",".join(sorted(exclude_boards)) or "无")
    return codes

def load_codes_from_stocklist_hk(stocklist_csv: Path) -> List[str]:
    """加载港股股票列表"""
    df = pd.read_csv(stocklist_csv)
    if "symbol" in df.columns:
        codes = df["symbol"].astype(str).str.zfill(5).tolist()  # 港股5位代码
    else:
        codes = df.iloc[:, 0].astype(str).str.zfill(5).tolist()
    codes = list(dict.fromkeys(codes))  # 去重保持顺序
    logger.info("从 %s 读取到 %d 只港股", stocklist_csv, len(codes))
    return codes

def load_codes_from_stocklist_us(stocklist_csv: Path) -> List[str]:
    """加载美股股票列表"""
    df = pd.read_csv(stocklist_csv)
    if "symbol" in df.columns:
        codes = df["symbol"].astype(str).str.strip().tolist()  # 美股代码直接使用
    else:
        codes = df.iloc[:, 0].astype(str).str.strip().tolist()
    codes = list(dict.fromkeys(codes))  # 去重保持顺序
    logger.info("从 %s 读取到 %d 只美股", stocklist_csv, len(codes))
    return codes

# --------------------------- 单只抓取（全量覆盖保存） --------------------------- #
def fetch_one(
    code: str,
    start: str,
    end: str,
    out_dir: Path,
    market: str = "A",
):
    """
    抓取单只股票的K线数据
    
    Args:
        code: 股票代码
        start: 起始日期
        end: 结束日期
        out_dir: 输出目录
        market: 市场类型 "A"/"HK"/"US"
    """
    csv_path = out_dir / f"{code}.csv"

    for attempt in range(1, 4):
        try:
            new_df = _get_kline_akshare(code, start, end, market)
            if new_df.empty:
                logger.debug("%s (%s) 无数据，生成空表。", code, market)
                new_df = pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])
            new_df = validate(new_df)
            new_df.to_csv(csv_path, index=False)  # 直接覆盖保存
            break
        except Exception as e:
            if _looks_like_ip_ban(e):
                logger.error(f"{code} ({market}) 第 {attempt} 次抓取疑似被封禁，沉睡 {COOLDOWN_SECS} 秒")
                _cool_sleep(COOLDOWN_SECS)
            else:
                silent_seconds = 15 * attempt
                error_type = type(e).__name__
                # 尝试获取更详细的错误信息
                error_msg = str(e) if str(e) else ""
                if hasattr(e, 'args') and e.args:
                    # 尝试从异常参数中获取更多信息
                    error_details = " | ".join(str(arg) for arg in e.args if arg)
                    if error_details and error_details != error_msg:
                        error_msg = f"{error_msg} | {error_details}" if error_msg else error_details
                if not error_msg:
                    error_msg = repr(e)
                
                logger.info(f"{code} ({market}) 第 {attempt} 次抓取失败，{silent_seconds} 秒后重试：[{error_type}] {error_msg}")
                time.sleep(silent_seconds)
    else:
        logger.error("%s (%s) 三次抓取均失败，已跳过！", code, market)

# --------------------------- 主入口 --------------------------- #
def main():
    parser = argparse.ArgumentParser(description="从 stocklist.csv 读取股票池并用 AkShare 抓取日线K线（支持A股/港股/美股）")
    # 抓取范围
    parser.add_argument("--start", default="20240101", help="起始日期 YYYYMMDD 或 'today'")
    parser.add_argument("--end", default="today", help="结束日期 YYYYMMDD 或 'today'")
    # 股票清单与板块过滤
    parser.add_argument("--stocklist", type=Path, default=Path("./stocklist.csv"), help="A股股票清单CSV路径（需含 ts_code 或 symbol）")
    parser.add_argument("--stocklist_hk", type=Path, default=None, help="港股股票清单CSV路径（可选）")
    parser.add_argument("--stocklist_us", type=Path, default=None, help="美股股票清单CSV路径（可选）")
    parser.add_argument(
        "--exclude-boards",
        nargs="*",
        default=[],
        choices=["gem", "star", "bj"],
        help="排除板块，可多选：gem(创业板300/301) star(科创板688) bj(北交所.BJ/4/8)"
    )
    # 其它
    parser.add_argument("--out", default="./data", help="输出根目录（A股/HK-stocks/US-stocks会自动创建）")
    parser.add_argument("--workers", type=int, default=6, help="并发线程数")
    args = parser.parse_args()

    # ---------- AkShare 初始化（无需 Token） ---------- #
    logger.info("使用 AkShare 数据源（免费，无需 Token）")
    
    # ---------- 连接测试 ---------- #
    try:
        # 尝试获取一个常见股票的数据来测试连接
        # 使用最近30天的日期范围，确保包含交易日
        from datetime import datetime, timedelta
        test_end = datetime.now().strftime("%Y%m%d")
        test_start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        test_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date=test_start, end_date=test_end, adjust="qfq")
        if test_df is None or test_df.empty:
            logger.warning("连接测试：API 返回空数据，可能网络问题或数据源异常")
        else:
            logger.info("连接测试成功，AkShare 数据源可用（获取到 %d 条数据）", len(test_df))
    except Exception as e:
        error_msg = str(e) if str(e) else repr(e)
        logger.warning(f"连接测试异常：{error_msg}，程序将继续运行")

    # ---------- 日期解析 ---------- #
    start = dt.date.today().strftime("%Y%m%d") if str(args.start).lower() == "today" else args.start
    end = dt.date.today().strftime("%Y%m%d") if str(args.end).lower() == "today" else args.end

    base_out_dir = Path(args.out)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义各市场的输出目录
    out_dir_a = base_out_dir / "A-shares"
    out_dir_hk = base_out_dir / "HK-stocks"
    out_dir_us = base_out_dir / "US-stocks"
    
    # 创建各市场目录
    out_dir_a.mkdir(parents=True, exist_ok=True)
    if args.stocklist_hk:
        out_dir_hk.mkdir(parents=True, exist_ok=True)
    if args.stocklist_us:
        out_dir_us.mkdir(parents=True, exist_ok=True)

    # ---------- 处理A股 ---------- #
    exclude_boards = set(args.exclude_boards or [])
    codes_a = []
    if args.stocklist and args.stocklist.exists():
        codes_a = load_codes_from_stocklist(args.stocklist, exclude_boards)
    
    # ---------- 处理港股 ---------- #
    codes_hk = []
    if args.stocklist_hk and args.stocklist_hk.exists():
        codes_hk = load_codes_from_stocklist_hk(args.stocklist_hk)
    
    # ---------- 处理美股 ---------- #
    codes_us = []
    if args.stocklist_us and args.stocklist_us.exists():
        codes_us = load_codes_from_stocklist_us(args.stocklist_us)

    # 检查是否有股票需要抓取
    total_stocks = len(codes_a) + len(codes_hk) + len(codes_us)
    if total_stocks == 0:
        logger.error("没有股票需要抓取，请检查股票列表文件。")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("开始抓取股票数据 | 数据源:AkShare(日线,qfq) | 日期:%s → %s", start, end)
    logger.info("  A股: %d 只 (保存至: %s)", len(codes_a), out_dir_a)
    if codes_hk:
        logger.info("  港股: %d 只 (保存至: %s)", len(codes_hk), out_dir_hk)
    if codes_us:
        logger.info("  美股: %d 只 (保存至: %s)", len(codes_us), out_dir_us)
    logger.info("=" * 80)

    # ---------- 多线程抓取A股 ---------- #
    if codes_a:
        logger.info("开始抓取A股数据...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(fetch_one, code, start, end, out_dir_a, "A")
                for code in codes_a
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="A股下载进度"):
                pass
        logger.info("A股数据抓取完成，已保存至 %s", out_dir_a.resolve())

    # ---------- 多线程抓取港股 ---------- #
    if codes_hk:
        logger.info("开始抓取港股数据...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(fetch_one, code, start, end, out_dir_hk, "HK")
                for code in codes_hk
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="港股下载进度"):
                pass
        logger.info("港股数据抓取完成，已保存至 %s", out_dir_hk.resolve())

    # ---------- 多线程抓取美股 ---------- #
    # 注意：由于 py_mini_racer 多线程不稳定，美股数据抓取使用较低的并发数
    # 虽然已经加了锁，但降低并发数可以进一步减少内存压力
    if codes_us:
        logger.info("开始抓取美股数据...")
        logger.info("提示：美股接口使用 py_mini_racer，已启用线程锁保护，并发数已自动降低以提高稳定性")
        # 美股使用较低的并发数，避免 py_mini_racer 崩溃
        us_workers = min(2, args.workers)  # 最多2个并发，即使设置了更高的workers
        with ThreadPoolExecutor(max_workers=us_workers) as executor:
            futures = [
                executor.submit(fetch_one, code, start, end, out_dir_us, "US")
                for code in codes_us
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="美股下载进度"):
                pass
        logger.info("美股数据抓取完成，已保存至 %s", out_dir_us.resolve())

    logger.info("=" * 80)
    logger.info("全部任务完成！")
    logger.info("  A股: %s", out_dir_a.resolve())
    if codes_hk:
        logger.info("  港股: %s", out_dir_hk.resolve())
    if codes_us:
        logger.info("  美股: %s", out_dir_us.resolve())
    logger.info("=" * 80)

if __name__ == "__main__":
    main()

'''
使用示例：

# 只抓取A股
python akshare_fetch_kline.py \
    --start 20250101 \
    --end today \
    --stocklist ./data/stocklist.csv \
    --exclude-boards gem star bj \
    --out ./data \
    --workers 6

# 只抓取港股
python akshare_fetch_kline.py 
    --start 20250101 
    --end today 
    --stocklist_hk ./data/stocklist_hk.csv 
    --out ./data 
    --workers 6

# 只抓取美股
python akshare_fetch_kline.py 
    --start 20250101 
    --end today 
    --stocklist_us ./data/stocklist_us.csv 
    --out ./data 
    --workers 6

# 同时抓取A股、港股、美股
python akshare_fetch_kline.py \
    --start 20250101 \
    --end today \
    --stocklist ./data/stocklist.csv \
    --stocklist_hk ./data/stocklist_hk.csv \
    --stocklist_us ./data/stocklist_us.csv \
    --exclude-boards gem star bj \
    --out ./data \
    --workers 6



数据保存路径：
- A股: ./data/A-shares/
- 港股: ./data/HK-stocks/
- 美股: ./data/US-stocks/
'''