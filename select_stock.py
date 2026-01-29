# 选股策略

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List
import yaml

import pandas as pd

# 导入Selector模块
from utils.Selector import BBIKDJSelector

# 本次脚本运行的统一时间戳：年月时分（例如：202501271429）
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M")

# ---------- 日志配置 ---------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/select_results_{TIMESTAMP}.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("select")


# ---------- 工具函数 ---------- #

def load_data(data_dir: Path, codes: Iterable[str], market: str = "A") -> Dict[str, pd.DataFrame]:
    """
    加载股票数据
    
    Args:
        data_dir: 数据根目录
        codes: 股票代码列表
        market: 市场类型 "A"(A股), "HK"(港股), "US"(美股)
    
    Returns:
        股票代码到DataFrame的字典
    """
    # 根据市场类型确定子目录
    if market == "A":
        subdir = "A-shares"
    elif market == "HK":
        subdir = "HK-stocks"
    elif market == "US":
        subdir = "US-stocks"
    else:
        raise ValueError(f"不支持的市场类型: {market}")
    
    market_dir = data_dir / subdir
    if not market_dir.exists():
        logger.warning(f"市场目录不存在: {market_dir}")
        return {}
    
    frames: Dict[str, pd.DataFrame] = {}
    for code in codes:
        fp = market_dir / f"{code}.csv"
        if not fp.exists():
            logger.debug(f"{code} 不存在，跳过")
            continue
        try:
            df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")
            if not df.empty:
                frames[code] = df
        except Exception as e:
            logger.warning(f"读取 {code} 失败: {e}")
            continue
    return frames


def load_config(cfg_path: Path) -> List[Dict[str, Any]]:
    """
    加载YAML配置文件
    
    支持两种格式：
    1. 单个策略配置：直接包含 class、alias、activate、params 字段
    2. 多个策略配置：包含 selectors 列表，或直接是列表
    """
    if not cfg_path.exists():
        logger.error(f"配置文件 {cfg_path} 不存在")
        sys.exit(1)
    
    with cfg_path.open(encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)
    
    if cfg_raw is None:
        logger.error("配置文件为空")
        sys.exit(1)
    
    # 判断是单个策略配置还是多个策略配置
    if isinstance(cfg_raw, list):
        # 直接是列表（多个策略）
        cfgs = cfg_raw
    elif isinstance(cfg_raw, dict):
        if "selectors" in cfg_raw:
            # 包含 selectors 键（多个策略）
            cfgs = cfg_raw["selectors"]
        elif "class" in cfg_raw:
            # 单个策略配置（直接包含 class 字段）
            cfgs = [cfg_raw]
        else:
            logger.error("配置文件格式错误：单个策略需要 'class' 字段，多个策略需要 'selectors' 键或直接是列表")
            sys.exit(1)
    else:
        logger.error("配置文件格式错误")
        sys.exit(1)
    
    if not cfgs:
        logger.error("配置文件中未定义任何 Selector")
        sys.exit(1)
    
    return cfgs


def load_all_configs(cfg_dir: Path) -> List[Dict[str, Any]]:
    """
    扫描目录下所有 YAML 配置文件并合并为一个 Selector 配置列表。
    - 支持单策略文件（直接包含 class/alias/activate/params）
    - 也支持带 selectors 列表的多策略文件
    - 默认会跳过文件名包含 'AllSelectors' 的聚合配置，避免与单策略文件重复
    """
    if not cfg_dir.exists() or not cfg_dir.is_dir():
        logger.error(f"配置目录 {cfg_dir} 不存在或不是目录")
        sys.exit(1)

    all_cfgs: List[Dict[str, Any]] = []
    for fp in sorted(cfg_dir.glob("*.yml")):
        # 跳过聚合配置，防止和拆分后的单策略文件重复运行
        if "AllSelectors" in fp.name:
            logger.info(f"忽略聚合配置文件: {fp.name}")
            continue
        try:
            cfgs = load_config(fp)
        except SystemExit:
            # 单个文件出错时仅记录日志并跳过，不中断整个流程
            logger.error(f"跳过配置文件 {fp}: 解析失败")
            continue
        except Exception as e:
            logger.error(f"跳过配置文件 {fp}: {e}")
            continue
        all_cfgs.extend(cfgs)

    if not all_cfgs:
        logger.error(f"在目录 {cfg_dir} 中未找到任何有效的 Selector 配置")
        sys.exit(1)

    return all_cfgs


def instantiate_selector(cfg: Dict[str, Any]):
    """动态加载Selector类并实例化"""
    cls_name: str = cfg.get("class")
    if not cls_name:
        raise ValueError("缺少 class 字段")
    
    # 从utils.Selector模块导入
    try:
        from utils import Selector
        cls = getattr(Selector, cls_name)
    except AttributeError as e:
        raise ImportError(f"无法加载 Selector.{cls_name}: {e}") from e
    
    params = cfg.get("params", {})
    alias = cfg.get("alias", cls_name)
    return alias, cls(**params)


def save_results(picks: List[str], selector_alias: str, trade_date: pd.Timestamp, results_dir: Path = Path("./results")):
    """
    保存选股结果到 results/年月时分/Selector名称 目录下
    
    Args:
        picks: 选中的股票代码列表（可能包含市场前缀，如 A_000001）
        selector_alias: Selector的别名
        trade_date: 交易日期
        results_dir: 结果根目录，默认为 ./results
    """
    # 创建目录：results/年月时分/Selector名称
    save_dir = results_dir / TIMESTAMP / selector_alias
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 按市场分组整理结果
    results_data = []
    for pick in picks:
        if pick.startswith("A_"):
            market = "A"
            code = pick[2:]
        elif pick.startswith("HK_"):
            market = "HK"
            code = pick[3:]
        elif pick.startswith("US_"):
            market = "US"
            code = pick[3:]
        else:
            market = "Unknown"
            code = pick
        
        results_data.append({
            "market": market,
            "code": code,
            "full_code": pick,
            "trade_date": trade_date.strftime("%Y-%m-%d"),
            "select_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # 保存为CSV文件
    if results_data:
        df_results = pd.DataFrame(results_data)
        csv_path = save_dir / "selected_stocks.csv"
        df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"选股结果已保存到: {csv_path}")
        logger.info(f"共保存 {len(results_data)} 只股票")
    else:
        # 即使没有选中的股票，也创建一个空文件记录
        csv_path = save_dir / "selected_stocks.csv"
        pd.DataFrame(columns=["market", "code", "full_code", "trade_date", "select_time"]).to_csv(
            csv_path, index=False, encoding="utf-8-sig"
        )
        logger.info(f"无选中股票，空结果已保存到: {csv_path}")
    
    return save_dir


# ---------- 主函数 ---------- #

def main():
    parser = argparse.ArgumentParser(description="运行选股策略")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="数据根目录（包含A-shares/HK-stocks/US-stocks子目录）")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="单个 Selector 配置文件路径（支持单个策略或多个策略配置）；为空时自动加载 config 目录下所有配置",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("./config"),
        help="当未指定 --config 时，从该目录下自动加载所有配置文件",
    )
    parser.add_argument("--date", help="交易日 YYYY-MM-DD；缺省=数据最新日期")
    parser.add_argument("--market", choices=["A", "HK", "US", "all"], default="A", help="市场类型：A(仅A股), HK(仅港股), US(仅美股), all(全部)")
    parser.add_argument("--tickers", default="all", help="'all' 或逗号分隔股票代码列表（仅用于指定市场）")
    args = parser.parse_args()
    
    # --- 加载行情数据 ---
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"数据目录 {data_dir} 不存在")
        sys.exit(1)
    
    # 确定要处理的市场
    markets_to_process = []
    if args.market == "all":
        markets_to_process = ["A", "HK", "US"]
    else:
        markets_to_process = [args.market]
    
    # 加载所有市场的数据
    all_data: Dict[str, pd.DataFrame] = {}
    for market in markets_to_process:
        if args.tickers.lower() == "all":
            # 扫描该市场目录下的所有CSV文件
            if market == "A":
                subdir = "A-shares"
            elif market == "HK":
                subdir = "HK-stocks"
            elif market == "US":
                subdir = "US-stocks"
            else:
                continue
            
            market_dir = data_dir / subdir
            if market_dir.exists():
                codes = [f.stem for f in market_dir.glob("*.csv")]
            else:
                codes = []
        else:
            codes = [c.strip() for c in args.tickers.split(",") if c.strip()]
        
        if not codes:
            logger.warning(f"{market} 市场没有找到股票代码")
            continue
        
        market_data = load_data(data_dir, codes, market)
        if market_data:
            # 为代码添加市场前缀，避免不同市场代码冲突
            for code, df in market_data.items():
                all_data[f"{market}_{code}"] = df
            logger.info(f"加载 {market} 市场数据: {len(market_data)} 只股票")
    
    if not all_data:
        logger.error("未能加载任何行情数据")
        sys.exit(1)
    
    # --- 确定交易日期 ---
    if args.date:
        trade_date = pd.to_datetime(args.date)
    else:
        # 收集所有有效的最大日期
        valid_dates = []
        for df in all_data.values():
            if df.empty or "date" not in df.columns:
                continue
            valid_date_series = df["date"].dropna()
            if not valid_date_series.empty:
                max_date = valid_date_series.max()
                if pd.notna(max_date):
                    valid_dates.append(pd.Timestamp(max_date))
        
        if not valid_dates:
            logger.error("所有数据文件都没有有效的日期数据")
            sys.exit(1)
        
        trade_date = max(valid_dates)
    
    if not args.date:
        logger.info(f"未指定 --date，使用最近日期 {trade_date.date()}")
    
    # --- 加载Selector配置 ---
    if args.config is not None:
        # 兼容旧用法：显式指定单个配置文件
        selector_cfgs = load_config(Path(args.config))
    else:
        # 新用法：未指定 --config 时，自动加载 config 目录下所有配置
        selector_cfgs = load_all_configs(Path(args.config_dir))
    
    # --- 逐个Selector运行 ---
    # 用于汇总所有策略的选股结果：key 为 full_code，value 为信息字典
    combined_results: Dict[str, Dict[str, Any]] = {}

    for cfg in selector_cfgs:
        if cfg.get("activate", True) is False:
            logger.info(f"跳过未激活的策略: {cfg.get('alias', cfg.get('class'))}")
            continue
        
        try:
            alias, selector = instantiate_selector(cfg)
        except Exception as e:
            logger.error(f"跳过配置 {cfg.get('alias', cfg.get('class'))}：{e}")
            continue
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"运行选股策略: {alias}")
        logger.info(f"交易日: {trade_date.date()}")
        logger.info(f"股票池: {len(all_data)} 只")
        
        picks = selector.select(trade_date, all_data)
        
        # 输出结果
        logger.info("")
        logger.info("============== 选股结果 ==============")
        logger.info(f"符合条件股票数: {len(picks)}")
        if picks:
            # 按市场分组显示
            picks_by_market: Dict[str, List[str]] = {"A": [], "HK": [], "US": []}
            for pick in picks:
                if pick.startswith("A_"):
                    picks_by_market["A"].append(pick[2:])
                elif pick.startswith("HK_"):
                    picks_by_market["HK"].append(pick[3:])
                elif pick.startswith("US_"):
                    picks_by_market["US"].append(pick[3:])
            
            for market, codes in picks_by_market.items():
                if codes:
                    logger.info(f"{market}股 ({len(codes)}只): {', '.join(codes)}")
        else:
            logger.info("无符合条件股票")
        logger.info("=" * 80)
        
        # 保存选股结果到文件
        save_results(picks, alias, trade_date)

        # 汇总结果：记录每只股票被哪些策略选中，并附带市值等信息
        for pick in picks:
            # 解析市场和纯代码
            if pick.startswith("A_"):
                market = "A"
                code = pick[2:]
            elif pick.startswith("HK_"):
                market = "HK"
                code = pick[3:]
            elif pick.startswith("US_"):
                market = "US"
                code = pick[3:]
            else:
                market = "Unknown"
                code = pick

            if pick not in combined_results:
                # 从行情数据中获取市值（若有）
                mcap = None
                df = all_data.get(pick)
                if df is not None and not df.empty:
                    df_valid = df[df["date"] <= trade_date].sort_values("date")
                    if not df_valid.empty:
                        last_row = df_valid.iloc[-1]
                        # 尝试一系列常见市值字段名
                        for col in ["market_cap", "total_mv", "总市值", "市值", "total_market_cap"]:
                            if col in last_row.index and pd.notna(last_row[col]):
                                try:
                                    mcap = float(last_row[col])
                                except Exception:
                                    mcap = None
                                break

                combined_results[pick] = {
                    "market": market,
                    "code": code,
                    "full_code": pick,
                    "trade_date": trade_date.strftime("%Y-%m-%d"),
                    "market_cap": mcap,
                    "strategies": set(),
                }

            # 记录该股票被哪个策略命中
            strategies_set = combined_results[pick]["strategies"]
            if isinstance(strategies_set, set):
                strategies_set.add(alias)
            else:
                # 理论上不会发生，只是防御性处理
                combined_results[pick]["strategies"] = {alias}

    # --- 生成汇总 CSV ---
    if combined_results:
        summary_rows: List[Dict[str, Any]] = []
        for info in combined_results.values():
            row = dict(info)
            # 将策略集合转换为逗号分隔的字符串，便于查看
            strategies = info.get("strategies", set())
            if isinstance(strategies, set):
                row["strategies"] = ",".join(sorted(strategies))
            else:
                row["strategies"] = str(strategies)
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        if "market_cap" in df_summary.columns:
            df_summary = df_summary.sort_values(
                by="market_cap", ascending=False, na_position="last"
            )

        # 汇总文件放在本次运行的 results/TIMESTAMP 目录下
        summary_root = Path("./results") / TIMESTAMP
        summary_root.mkdir(parents=True, exist_ok=True)
        summary_path = summary_root / "summary_by_market_cap.csv"
        df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        logger.info(f"所有策略汇总结果已保存到: {summary_path}")


if __name__ == "__main__":
    main()

'''
# 运行少妇战法选股（A股，使用默认配置）
python select_stock.py --data-dir ./data --market A

# 指定日期
python select_stock.py --data-dir ./data --market A --date 2025-01-26

# 多市场
python select_stock.py --data-dir ./data --market all

# 指定股票代码
python select_stock.py --data-dir ./data --market A --tickers 000001,000002

# 使用默认配置文件（BBIKDJSelector_config.yml）
python select_stock.py --data-dir ./data --market A

# 指定其他配置文件
python select_stock.py --data-dir ./data --market A --config ./config/OtherSelector_config.yml
'''