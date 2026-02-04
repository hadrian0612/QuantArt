"""
单独拉取 002230 的 A 股日线数据，写入 data/A-shares/002230.csv（格式与项目其他 CSV 一致）
数据源与 akshare_fetch_kline.py 一致：AkShare stock_zh_a_hist(前复权)，失败时回退 AData。
"""
from pathlib import Path
import datetime as dt
import time

import pandas as pd
import akshare as ak

try:
    import adata
except ImportError:
    adata = None

CODE = "002230"
OUT_CSV = Path(__file__).resolve().parent / "data" / "A-shares" / f"{CODE}.csv"
# 与 akshare_fetch_kline.py 的 fetch_one 一致：多轮重试，间隔 15*attempt 秒
MAX_ATTEMPTS = 4


def fetch_with_akshare(start_str: str, end_str: str) -> pd.DataFrame:
    """使用与 akshare_fetch_kline 相同的接口与重试策略拉取日线"""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"[AkShare] 拉取 {CODE} 日线 {start_str} ~ {end_str} ... (第 {attempt}/{MAX_ATTEMPTS} 次)")
            df = ak.stock_zh_a_hist(
                symbol=CODE,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust="qfq",
            )
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            if attempt < MAX_ATTEMPTS:
                delay = 15 * attempt
                print(f"  [AkShare] 失败，{delay} 秒后重试: {e}")
                time.sleep(delay)
            else:
                print(f"  [AkShare] 多次重试仍失败: {e}")
    return pd.DataFrame()


def fetch_with_adata(start_str: str, end_str: str) -> pd.DataFrame:
    """回退：使用 AData 获取日线（仅 A 股）"""
    if adata is None:
        print("[AData] 未安装 adata，无法使用 AData 回退")
        return pd.DataFrame()

    start_dt = dt.datetime.strptime(start_str, "%Y%m%d").strftime("%Y-%m-%d")
    end_dt = dt.datetime.strptime(end_str, "%Y%m%d").strftime("%Y-%m-%d")
    end_date = pd.to_datetime(end_str).date()

    # 方式1：带 end_date 请求（部分版本支持）
    print(f"[AData] 拉取 {CODE} 日线 {start_dt} ~ {end_dt} ...")
    try:
        df = adata.stock.market.get_market(
            stock_code=CODE,
            k_type=1,
            start_date=start_dt,
            end_date=end_dt,
        )
        if df is not None and not df.empty:
            return _adata_filter_dates(df, end_date)
    except Exception as e:
        print(f"[AData] 带 end_date 请求异常: {e}")

    # 方式2：只传 start_date（官方示例用法），取回后本地截断到 end
    print("[AData] 改为仅传 start_date 拉取，本地截断日期 ...")
    try:
        df = adata.stock.market.get_market(
            stock_code=CODE,
            k_type=1,
            start_date=start_dt,
        )
        if df is not None and not df.empty:
            return _adata_filter_dates(df, end_date)
    except Exception as e:
        print(f"[AData] 仅 start_date 请求异常: {e}")

    # 方式3：若结束日是今天，再试一次用“昨天”为界的短区间
    if end_dt == dt.date.today().strftime("%Y-%m-%d"):
        yesterday = (dt.date.today() - dt.timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"[AData] 尝试最近区间 {yesterday} ~ {yesterday} ...")
        try:
            df = adata.stock.market.get_market(
                stock_code=CODE,
                k_type=1,
                start_date=yesterday,
                end_date=yesterday,
            )
            if df is not None and not df.empty:
                return _adata_filter_dates(df, end_date)
        except Exception as e:
            print(f"[AData] 单日请求异常: {e}")

    print("[AData] 无可用数据")
    return pd.DataFrame()


def _adata_filter_dates(df: pd.DataFrame, end_date) -> pd.DataFrame:
    """将 AData 返回的 df 按 end_date 截断（列名可能为 trade_date 或 date）"""
    date_col = "trade_date" if "trade_date" in df.columns else "date"
    if date_col not in df.columns:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col].dt.date <= end_date]
    return df


def main():
    # 日期范围：2025-01-01 至今天
    end = dt.date.today()
    start = dt.date(2025, 1, 1)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    # 1) 先尝试 AkShare
    df = fetch_with_akshare(start_str, end_str)

    # 2) 若 AkShare 失败或返回空，再尝试 AData
    if df is None or df.empty:
        print("[INFO] AkShare 无数据或失败，尝试使用 AData 回退...")
        df = fetch_with_adata(start_str, end_str)

    if df is None or df.empty:
        if OUT_CSV.exists() and OUT_CSV.stat().st_size > 100:
            try:
                existing = pd.read_csv(OUT_CSV, parse_dates=["date"])
                n = len(existing) if not existing.empty else 0
                print(f"未获取到新数据（AkShare/AData 均失败或返回空），已保留本地 {OUT_CSV.name} 原有 {n} 条，未覆盖。")
            except Exception:
                print("未获取到任何数据（AkShare / AData 均失败或返回空）。")
        else:
            print("未获取到任何数据（AkShare / AData 均失败或返回空）。")
        return

    # 统一列名（与 incremental_update_kline / incremental_adata_update 一致）
    col_map = {
        "日期": "date",
        "trade_date": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
    }
    for old, new in col_map.items():
        if old in df.columns and old != new:
            df = df.rename(columns={old: new})

    required = ["date", "open", "close", "high", "low", "volume"]
    if not all(c in df.columns for c in required):
        print("返回数据缺少必要列:", df.columns.tolist())
        return

    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"已写入 {len(df)} 条 → {OUT_CSV}")


if __name__ == "__main__":
    main()
