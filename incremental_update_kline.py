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
from typing import List, Optional, Tuple
import os

import pandas as pd
import akshare as ak
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --------------------------- 全局锁（用于美股数据抓取，避免 py_mini_racer 多线程冲突） --------------------------- #
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
logger = logging.getLogger("incremental_update_kline")

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
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq"  # 前复权
            )
        elif market == "HK":
            df = None
            try:
                df = ak.stock_hk_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
            except:
                pass
            if df is None or df.empty:
                try:
                    df = ak.stock_hk_daily(symbol=code, adjust="qfq", start_date=start, end_date=end)
                except:
                    pass
            if df is None or df.empty:
                try:
                    df = ak.stock_hk_hist_min_em(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
                except:
                    pass
            if df is None:
                raise ValueError(f"无法获取港股 {code} 的数据，所有API接口均失败")
        elif market == "US":
            with _us_stock_lock:
                time.sleep(0.1)
                try:
                    df = ak.stock_us_daily(symbol=code, adjust="qfq", start_date=start, end_date=end)
                except:
                    try:
                        df_all = ak.stock_us_daily(symbol=code, adjust="qfq")
                        if df_all is not None and not df_all.empty:
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
        raise

    if df is None or df.empty:
        return pd.DataFrame()

    # 统一列名映射
    column_mapping = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "date": "date",
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume"
    }
    
    rename_dict = {}
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and old_col != new_col:
            rename_dict[old_col] = new_col
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    required_cols = ["date", "open", "close", "high", "low", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"{code} ({market}) 数据缺少列: {missing_cols}，可用列: {df.columns.tolist()}")
        return pd.DataFrame()
    
    df = df[required_cols].copy()
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

# --------------------------- 增量更新核心逻辑 --------------------------- #

def get_last_date_from_csv(csv_path: Path) -> Optional[pd.Timestamp]:
    """
    从CSV文件中读取最后一条记录的日期
    
    Returns:
        最后日期，如果文件不存在、为空或只有表头则返回 None
    """
    if not csv_path.exists():
        return None
    
    try:
        # 先检查文件大小，如果只有表头（很小），直接返回None
        file_size = csv_path.stat().st_size
        if file_size < 100:  # 只有表头的文件通常很小
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 如果只有1行（表头）或第2行为空，认为是空文件
                if len(lines) <= 1 or (len(lines) == 2 and not lines[1].strip()):
                    return None
        
        df = pd.read_csv(csv_path, parse_dates=["date"])
        # 检查是否有有效数据行（排除只有表头的情况）
        if df.empty or len(df) == 0:
            return None
        if "date" not in df.columns:
            return None
        # 检查date列是否有有效值（非NaN）
        valid_dates = df["date"].dropna()
        if valid_dates.empty:
            return None
        
        df = df.sort_values("date")
        last_date = pd.to_datetime(df["date"].iloc[-1])
        # 确保日期有效
        if pd.isna(last_date):
            return None
        return last_date
    except Exception as e:
        logger.warning(f"读取 {csv_path} 失败: {e}")
        return None

def detect_market_from_code(code: str, data_dir: Path) -> Optional[str]:
    """
    根据数据目录结构自动检测市场类型
    检查 code.csv 在哪个子目录下：A-shares, HK-stocks, US-stocks
    """
    for market, subdir in [("A", "A-shares"), ("HK", "HK-stocks"), ("US", "US-stocks")]:
        csv_path = data_dir / subdir / f"{code}.csv"
        if csv_path.exists():
            return market
    return None

def _is_connection_error(e: Exception) -> bool:
    """判断是否为连接错误"""
    error_type = type(e).__name__
    error_msg = str(e).lower()
    return (
        error_type == "ConnectionError" or
        "connection" in error_msg or
        "remote" in error_msg or
        "disconnected" in error_msg or
        "reset" in error_msg
    )

def update_one_incremental(
    code: str,
    csv_path: Path,
    end_date: str,
    market: str,
    default_start_date: Optional[str] = None,
    is_retry: bool = False,
) -> bool:
    """
    增量更新单只股票的K线数据
    
    Args:
        code: 股票代码
        csv_path: CSV文件路径
        end_date: 结束日期 YYYYMMDD
        market: 市场类型 "A"/"HK"/"US"
        default_start_date: 默认开始日期 YYYYMMDD（当文件为空时使用，None则不处理空文件）
        is_retry: 是否为延迟重试（用于区分首次尝试和最后统一重试）
    
    Returns:
        bool: True表示成功，False表示失败（需要延迟重试）
    """
    # 读取现有数据的最后日期
    last_date = get_last_date_from_csv(csv_path)
    
    if last_date is None:
        # 文件为空或不存在，使用默认开始日期
        if default_start_date is None:
            logger.warning(f"{code} ({market}) 文件不存在或为空，跳过增量更新（请使用全量抓取脚本或设置 --default-start）")
            return
        else:
            start_date = default_start_date
            logger.info(f"{code} ({market}) 文件为空，使用默认开始日期: {start_date} → {end_date}")
    else:
        # 计算增量起始日期（最后日期+1天）
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        end_dt = pd.to_datetime(end_date)
        
        # 如果最后日期已经 >= 结束日期，无需更新
        if last_date >= end_dt:
            logger.debug(f"{code} ({market}) 数据已是最新，最后日期: {last_date.date()}")
            return
        
        # 如果起始日期 > 结束日期，也无需更新
        if pd.to_datetime(start_date) > end_dt:
            return
        logger.info(f"{code} ({market}) 增量更新: {start_date} → {end_date} (最后日期: {last_date.date()})")
    
    # 连接错误：短延时重试（最多3次）
    # 其他错误：如果是首次尝试，记录后返回False；如果是延迟重试，直接失败
    max_attempts = 3
    
    for attempt in range(1, max_attempts + 1):
        try:
            # 获取增量数据
            new_df = _get_kline_akshare(code, start_date, end_date, market)
            
            if new_df.empty:
                logger.debug(f"{code} ({market}) 无新数据")
                return True
            
            new_df = validate(new_df)
            
            # 合并数据（去重，保留最新）
            if last_date is None:
                # 空文件，直接保存新数据
                combined_df = new_df
            else:
                # 读取现有数据并合并
                existing_df = pd.read_csv(csv_path, parse_dates=["date"])
                existing_df = validate(existing_df)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset="date", keep="last")
                combined_df = combined_df.sort_values("date").reset_index(drop=True)
            
            # 保存
            combined_df.to_csv(csv_path, index=False)
            logger.info(f"{code} ({market}) 更新成功，新增 {len(new_df)} 条，总计 {len(combined_df)} 条")
            return True
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e) if str(e) else ""
            if hasattr(e, 'args') and e.args:
                error_details = " | ".join(str(arg) for arg in e.args if arg)
                if error_details and error_details != error_msg:
                    error_msg = f"{error_msg} | {error_details}" if error_msg else error_details
            if not error_msg:
                error_msg = repr(e)
            
            # 判断错误类型
            if _looks_like_ip_ban(e):
                # IP封禁：长延时
                logger.error(f"{code} ({market}) 第 {attempt} 次抓取疑似被封禁，沉睡 {COOLDOWN_SECS} 秒")
                if attempt < max_attempts:
                    _cool_sleep(COOLDOWN_SECS)
                else:
                    logger.error(f"{code} ({market}) 三次抓取均失败（疑似封禁），已跳过！")
                    return False
            elif _is_connection_error(e):
                # 连接错误：短延时重试
                if attempt < max_attempts:
                    short_delay = 2  # 连接错误短延时2秒
                    logger.info(f"{code} ({market}) 第 {attempt} 次抓取失败（连接错误），{short_delay} 秒后重试：[{error_type}] {error_msg}")
                    time.sleep(short_delay)
                else:
                    logger.warning(f"{code} ({market}) 连接错误，3次重试均失败，将延迟重试")
                    return False
            else:
                # 其他错误：如果是首次尝试，记录后返回False（延迟重试）
                # 如果是延迟重试，直接失败
                if is_retry:
                    logger.error(f"{code} ({market}) 延迟重试失败，已跳过：[{error_type}] {error_msg}")
                    return False
                else:
                    logger.warning(f"{code} ({market}) 第 {attempt} 次抓取失败（非连接错误），将延迟重试：[{error_type}] {error_msg}")
                    return False
    
    # 理论上不会到这里
    return False

# --------------------------- 主入口 --------------------------- #

def main():
    parser = argparse.ArgumentParser(description="增量更新已存在的K线数据（从最后日期到指定结束日期）")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="数据根目录（包含 A-shares/HK-stocks/US-stocks 子目录）")
    parser.add_argument("--end", default="today", help="结束日期 YYYYMMDD 或 'today'（默认：今天）")
    parser.add_argument("--default-start", default="20250101", help="默认开始日期 YYYYMMDD（当文件为空时使用，默认：20250101）")
    parser.add_argument("--market", choices=["A", "HK", "US", "all"], default="all", 
                        help="市场类型：A(仅A股), HK(仅港股), US(仅美股), all(全部，默认)")
    parser.add_argument("--workers", type=int, default=6, help="并发线程数（默认：6）")
    args = parser.parse_args()
    
    # ---------- 日期解析 ---------- #
    end_date = dt.date.today().strftime("%Y%m%d") if str(args.end).lower() == "today" else args.end
    default_start_date = args.default_start if args.default_start else None
    
    data_dir = args.data_dir
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        sys.exit(1)
    
    # ---------- 确定要更新的市场 ---------- #
    markets_to_update = []
    if args.market == "all":
        markets_to_update = ["A", "HK", "US"]
    else:
        markets_to_update = [args.market]
    
    # ---------- 收集需要更新的股票代码 ---------- #
    tasks: List[Tuple[str, Path, str]] = []  # (code, csv_path, market)
    
    for market in markets_to_update:
        if market == "A":
            subdir = "A-shares"
        elif market == "HK":
            subdir = "HK-stocks"
        elif market == "US":
            subdir = "US-stocks"
        else:
            continue
        
        market_dir = data_dir / subdir
        if not market_dir.exists():
            logger.info(f"市场目录不存在，跳过: {market_dir}")
            continue
        
        # 扫描该目录下的所有CSV文件
        csv_files = list(market_dir.glob("*.csv"))
        logger.info(f"在 {subdir} 目录下找到 {len(csv_files)} 个CSV文件")
        
        for csv_path in csv_files:
            code = csv_path.stem
            tasks.append((code, csv_path, market))
    
    if not tasks:
        logger.warning("没有找到需要更新的股票数据文件")
        sys.exit(0)
    
    logger.info("=" * 80)
    logger.info("开始增量更新K线数据 | 数据源:AkShare(日线,qfq) | 结束日期:%s", end_date)
    if default_start_date:
        logger.info("  空文件默认开始日期: %s", default_start_date)
    logger.info("  待更新股票数: %d", len(tasks))
    logger.info("=" * 80)
    
    # ---------- 多线程增量更新 ---------- #
    # 按优先级分离：A股 → 港股 → 美股
    a_tasks = [(code, path, mkt) for code, path, mkt in tasks if mkt == "A"]
    hk_tasks = [(code, path, mkt) for code, path, mkt in tasks if mkt == "HK"]
    us_tasks = [(code, path, mkt) for code, path, mkt in tasks if mkt == "US"]
    
    # 收集失败的任务，用于最后统一重试
    failed_tasks = []
    
    def _process_market(market_tasks, market_name, workers, is_retry=False):
        """处理单个市场的更新，返回失败的任务列表"""
        if not market_tasks:
            return []
        
        logger.info(f"开始更新{market_name}数据...")
        market_failed = []
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(update_one_incremental, code, csv_path, end_date, market, default_start_date, is_retry): (code, csv_path, market)
                for code, csv_path, market in market_tasks
            }
            
            pbar = tqdm(total=len(market_tasks), desc=f"{market_name}更新进度")
            try:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        success = future.result()
                        if not success:
                            # 任务失败，记录用于延迟重试（仅首次尝试时）
                            if not is_retry:
                                market_failed.append(task)
                    except Exception as e:
                        # 异常情况也记录为失败（仅首次尝试时）
                        if not is_retry:
                            logger.error(f"{task[0]} ({task[2]}) 更新异常: {e}")
                            market_failed.append(task)
                    pbar.update(1)
            finally:
                pbar.close()
        
        if not market_failed:
            logger.info(f"{market_name}数据更新完成")
        elif not is_retry:
            logger.warning(f"{market_name}数据更新完成，{len(market_failed)} 个任务失败，将在最后统一重试")
        
        return market_failed
    
    # 1. 先处理A股（优先级最高）
    failed_tasks.extend(_process_market(a_tasks, "A股", args.workers, is_retry=False))
    
    # 2. 再处理港股（优先级第二）
    failed_tasks.extend(_process_market(hk_tasks, "港股", args.workers, is_retry=False))
    
    # 3. 最后处理美股（优先级最低，使用较低并发）
    if us_tasks:
        logger.info("提示：美股接口使用 py_mini_racer，已启用线程锁保护，并发数已自动降低")
        us_workers = min(2, args.workers)
        failed_tasks.extend(_process_market(us_tasks, "美股", us_workers, is_retry=False))
    
    # 4. 统一重试失败的任务（仅一次）
    if failed_tasks:
        logger.info("=" * 80)
        logger.info(f"开始统一重试失败的 {len(failed_tasks)} 个任务...")
        
        # 按市场分组重试
        retry_a = [(code, path, mkt) for code, path, mkt in failed_tasks if mkt == "A"]
        retry_hk = [(code, path, mkt) for code, path, mkt in failed_tasks if mkt == "HK"]
        retry_us = [(code, path, mkt) for code, path, mkt in failed_tasks if mkt == "US"]
        
        retry_failed = []
        
        if retry_a:
            logger.info(f"重试A股失败任务: {len(retry_a)} 个")
            retry_failed.extend(_process_market(retry_a, "A股重试", args.workers, is_retry=True))
        
        if retry_hk:
            logger.info(f"重试港股失败任务: {len(retry_hk)} 个")
            retry_failed.extend(_process_market(retry_hk, "港股重试", args.workers, is_retry=True))
        
        if retry_us:
            logger.info(f"重试美股失败任务: {len(retry_us)} 个")
            us_workers = min(2, args.workers)
            retry_failed.extend(_process_market(retry_us, "美股重试", us_workers, is_retry=True))
        
        if retry_failed:
            logger.warning(f"重试后仍有 {len(retry_failed)} 个任务失败，已跳过")
            for code, _, market in retry_failed:
                logger.warning(f"  - {code} ({market})")
    
    logger.info("=" * 80)
    logger.info("全部增量更新任务完成！")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()

'''
使用示例：

# 更新所有市场的K线数据到今天（空文件从20250101开始）
python incremental_update_kline.py --data-dir ./data --end today --default-start 20250101 --workers 6

# 只更新A股数据到指定日期
python incremental_update_kline.py --data-dir ./data --end 20250115 --market A --default-start 20250101 --workers 6

# 只更新港股数据（使用自定义默认开始日期）
python incremental_update_kline.py --data-dir ./data --end today --market HK --default-start 20250101 --workers 6

# 不处理空文件（不设置 --default-start 或设置为空）
python incremental_update_kline.py --data-dir ./data --end today --workers 6

说明：
- 脚本会自动检测每个CSV文件的最后日期
- 从最后日期+1天开始增量更新到指定结束日期
- 如果数据已是最新（最后日期 >= 结束日期），则跳过
- 对于空文件（只有表头或完全为空），如果设置了 --default-start，则从该日期开始抓取
- 支持A股、港股、美股三种市场
- 数据目录结构：
  ./data/
    ├── A-shares/    (A股CSV文件)
    ├── HK-stocks/   (港股CSV文件)
    └── US-stocks/   (美股CSV文件)
'''
