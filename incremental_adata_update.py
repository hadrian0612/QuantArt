from __future__ import annotations

import argparse
import datetime as dt
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

import adata


# --------------------------- 日志配置 --------------------------- #
LOG_FILE = Path("./logs/fetch_adata.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("incremental_adata_update")


# --------------------------- 工具函数：CSV 读写 & 校验 --------------------------- #

def get_last_date_from_csv(csv_path: Path) -> Optional[pd.Timestamp]:
    """
    从 CSV 文件中读取最后一条记录的日期（与 incremental_update_kline.py 保持一致）
    """
    if not csv_path.exists():
        return None

    try:
        file_size = csv_path.stat().st_size
        if file_size < 100:
            with csv_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) <= 1 or (len(lines) == 2 and not lines[1].strip()):
                    return None

        df = pd.read_csv(csv_path, parse_dates=["date"])
        if df.empty or "date" not in df.columns:
            return None

        valid_dates = df["date"].dropna()
        if valid_dates.empty:
            return None

        df = df.sort_values("date")
        last_date = pd.to_datetime(df["date"].iloc[-1])
        if pd.isna(last_date):
            return None
        return last_date
    except Exception as e:
        logger.warning("读取 %s 失败: %s", csv_path, e)
        return None


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    与 incremental_update_kline.py 相同的校验逻辑：按 date 去重排序，不允许未来日期
    """
    if df is None or df.empty:
        return df
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("存在缺失日期！")
    if (df["date"] > pd.Timestamp.today()).any():
        raise ValueError("数据包含未来日期，可能抓取错误！")
    return df


# --------------------------- AData 日线获取 --------------------------- #

def _get_kline_adata(code: str, start: str, end: str) -> pd.DataFrame:
    """
    使用 AData 获取 A 股日线数据，返回与现有 CSV 兼容的格式：
    columns = [date, open, close, high, low, volume]

    Args:
        code: 6 位股票代码，如 "000001"
        start: 起始日期 YYYYMMDD
        end:   结束日期 YYYYMMDD
    """
    # AData 使用 YYYY-MM-DD 格式
    start_str = dt.datetime.strptime(start, "%Y%m%d").strftime("%Y-%m-%d")
    end_str = dt.datetime.strptime(end, "%Y%m%d").strftime("%Y-%m-%d")

    # k_type=1 表示日 K 线（按 AData 文档）
    df = adata.stock.market.get_market(
        stock_code=code,
        k_type=1,
        start_date=start_str,
        end_date=end_str,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # 统一列名映射
    column_mapping = {
        "trade_date": "date",
        "date": "date",
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
        # 兼容可能出现的中文列名
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "日期": "date",
    }

    rename_dict = {}
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and old_col != new_col:
            rename_dict[old_col] = new_col

    if rename_dict:
        df = df.rename(columns=rename_dict)

    required_cols = ["date", "open", "close", "high", "low", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning("%s (A) AData 返回数据缺少列: %s，可用列: %s", code, missing, df.columns.tolist())
        return pd.DataFrame()

    df = df[required_cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


# --------------------------- 单只股票增量更新 --------------------------- #

def update_one_incremental(
    code: str,
    csv_path: Path,
    end_date: str,
    default_start_date: Optional[str] = None,
    request_delay: float = 0.5,
) -> bool:
    """
    使用 AData 对单只 A 股股票做增量更新。

    返回：
        True  表示成功（包括已最新或无新数据）
        False 表示本轮更新失败（可用于统一重试）
    """
    last_date = get_last_date_from_csv(csv_path)

    if last_date is None:
        if default_start_date is None:
            logger.warning(
                "%s (A) 文件不存在或为空，且未设置 --default-start，跳过（建议先用全量脚本或设置默认开始日期）",
                code,
            )
            return True
        start_date = default_start_date
        logger.info("%s (A) 文件为空，使用默认开始日期: %s → %s", code, start_date, end_date)
    else:
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        end_dt = pd.to_datetime(end_date)

        if last_date >= end_dt:
            logger.debug("%s (A) 数据已是最新，最后日期: %s", code, last_date.date())
            return True

        if pd.to_datetime(start_date) > end_dt:
            return True

        logger.info("%s (A) 增量更新: %s → %s (最后日期: %s)", code, start_date, end_date, last_date.date())

    max_attempts = 3
    end_date_is_today = pd.to_datetime(end_date).date() == dt.date.today()

    for attempt in range(1, max_attempts + 1):
        try:
            if request_delay > 0:
                time.sleep(random.uniform(request_delay * 0.5, request_delay * 1.5))

            new_df = _get_kline_adata(code, start_date, end_date)

            # 如果请求 today 且返回空，并且本地已有历史数据，则尝试改用昨天
            if new_df.empty and end_date_is_today and last_date is not None:
                yesterday = (dt.date.today() - dt.timedelta(days=1)).strftime("%Y%m%d")
                if pd.to_datetime(start_date) <= pd.to_datetime(yesterday):
                    logger.info("%s (A) 今日数据暂未出，改用结束日期 %s 重试", code, yesterday)
                    if request_delay > 0:
                        time.sleep(random.uniform(request_delay * 0.5, request_delay * 1.5))
                    new_df = _get_kline_adata(code, start_date, yesterday)

            if new_df.empty:
                logger.debug("%s (A) 无新数据", code)
                return True

            new_df = validate(new_df)

            if last_date is None:
                combined_df = new_df
            else:
                existing_df = pd.read_csv(csv_path, parse_dates=["date"])
                existing_df = validate(existing_df)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset="date", keep="last")
                combined_df = combined_df.sort_values("date").reset_index(drop=True)

            combined_df.to_csv(csv_path, index=False)
            logger.info("%s (A) 更新成功，新增 %d 条，总计 %d 条", code, len(new_df), len(combined_df))
            return True

        except Exception as e:
            logger.warning("%s (A) 第 %d 次更新失败：%s", code, attempt, e)
            if attempt < max_attempts:
                time.sleep(3)
            else:
                logger.error("%s (A) 多次更新失败，留待统一重试。错误：%s", code, e)
                return False

    return False


# --------------------------- 主入口 --------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="使用 AData 对已存在的 A 股 K 线 CSV 做增量更新（从最后日期到指定结束日期）"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="数据根目录（包含 A-shares 子目录，路径和文件格式与 incremental_update_kline.py 兼容）",
    )
    parser.add_argument(
        "--end",
        default="today",
        help="结束日期 YYYYMMDD 或 'today'（默认：今天）",
    )
    parser.add_argument(
        "--default-start",
        default="20250101",
        help="默认开始日期 YYYYMMDD（当文件为空时使用，默认：20250101）",
    )
    parser.add_argument(
        "--market",
        choices=["A", "HK", "US", "all"],
        default="A",
        help="市场类型（当前脚本仅实际更新 A 股，其他市场会忽略，仅为参数兼容）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并发线程数（默认：4）",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.5,
        metavar="SEC",
        help="每次请求前的随机延时基数（秒），实际延时为该值的 0.5~1.5 倍（默认：0.5，可减少限流）",
    )
    args = parser.parse_args()

    end_date = dt.date.today().strftime("%Y%m%d") if str(args.end).lower() == "today" else args.end
    default_start_date = args.default_start if args.default_start else None

    data_dir = args.data_dir
    if not data_dir.exists():
        logger.error("数据目录不存在: %s", data_dir)
        sys.exit(1)

    # 仅实际处理 A 股，其余市场参数仅用于与旧脚本兼容
    if args.market in {"HK", "US", "all"}:
        logger.warning(
            "AData 增量脚本当前仅支持 A 股（A-shares 目录），其他市场将被忽略；如需港股/美股请继续使用 AkShare 版本脚本。"
        )

    market_dir = data_dir / "A-shares"
    if not market_dir.exists():
        logger.warning("A-shares 目录不存在: %s", market_dir)
        sys.exit(0)

    csv_files = list(market_dir.glob("*.csv"))
    tasks: List[Tuple[str, Path]] = [(fp.stem, fp) for fp in csv_files]

    if not tasks:
        logger.warning("A-shares 目录下没有找到任何 CSV 文件")
        sys.exit(0)

    logger.info("=" * 80)
    logger.info("开始增量更新 A 股 K 线 | 数据源: AData | 结束日期: %s", end_date)
    if default_start_date:
        logger.info("  空文件默认开始日期: %s", default_start_date)
    logger.info("  待更新股票数: %d", len(tasks))
    logger.info("=" * 80)

    failed_tasks: List[Tuple[str, Path]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(
                update_one_incremental,
                code,
                csv_path,
                end_date,
                default_start_date,
                args.request_delay,
            ): (code, csv_path)
            for code, csv_path in tasks
        }

        pbar = tqdm(total=len(tasks), desc="AData 更新进度")
        try:
            for future in as_completed(future_to_task):
                code, csv_path = future_to_task[future]
                try:
                    ok = future.result()
                    if not ok:
                        failed_tasks.append((code, csv_path))
                except Exception as e:
                    logger.error("%s (A) 更新异常：%s", code, e)
                    failed_tasks.append((code, csv_path))
                pbar.update(1)
        finally:
            pbar.close()

    if failed_tasks:
        logger.warning("首次更新后仍有 %d 只股票失败，将尝试统一重试一次", len(failed_tasks))

        retry_failed: List[Tuple[str, Path]] = []
        with ThreadPoolExecutor(max_workers=min(2, args.workers)) as executor:
            future_to_task = {
                executor.submit(
                    update_one_incremental,
                    code,
                    csv_path,
                    end_date,
                    default_start_date,
                    args.request_delay,
                ): (code, csv_path)
                for code, csv_path in failed_tasks
            }

            pbar = tqdm(total=len(failed_tasks), desc="AData 重试进度")
            try:
                for future in as_completed(future_to_task):
                    code, csv_path = future_to_task[future]
                    try:
                        ok = future.result()
                        if not ok:
                            retry_failed.append((code, csv_path))
                    except Exception as e:
                        logger.error("%s (A) 重试仍然失败：%s", code, e)
                        retry_failed.append((code, csv_path))
                    pbar.update(1)
            finally:
                pbar.close()

        if retry_failed:
            logger.warning("重试后仍有 %d 只股票失败，已跳过：", len(retry_failed))
            for code, _ in retry_failed:
                logger.warning("  - %s (A)", code)

    logger.info("=" * 80)
    logger.info("AData 增量更新任务完成！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

"""
使用示例（与 incremental_update_kline.py 输出目录和 CSV 格式兼容）：

# 使用 AData 将 A 股日线更新到今天
python incremental_adata_update.py --data-dir ./data --end today --default-start 20250101 --workers 4 --request-delay 0.5

# 只更新 A 股（本脚本仅支持 A 股；港股/美股仍请使用 AkShare 版本）
python incremental_adata_update.py --data-dir ./data --market A --end today
"""

