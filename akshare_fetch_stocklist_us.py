# 获取所有美股股票列表
# 使用akshare获取美股列表，包含ts_code, symbol, name等字段

import akshare as ak
import pandas as pd
from pathlib import Path
import time
from typing import Callable, Optional


def _call_with_retry(
    fn: Callable[[], pd.DataFrame],
    *,
    name: str,
    max_retries: int = 5,
    base_sleep: float = 1.0,
    max_sleep: float = 20.0,
) -> pd.DataFrame:
    """
    对不稳定的网络请求做重试（指数退避）。
    """
    last_err: Optional[BaseException] = None
    for i in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i >= max_retries:
                break
            sleep_s = min(max_sleep, base_sleep * (2 ** (i - 1)))
            print(f"{name} 第{i}次失败: {e}；{sleep_s:.1f}s 后重试...")
            time.sleep(sleep_s)
    raise RuntimeError(f"{name} 重试 {max_retries} 次仍失败") from last_err

def fetch_stocklist_us(output_path: str = "stocklist_us.csv", *, use_cache_on_fail: bool = True):
    """
    获取所有美股股票列表并保存为CSV文件
    
    Args:
        output_path: 输出CSV文件路径，默认为stocklist_us.csv
    """
    print("正在获取美股股票列表...")
    try:
        print(f"akshare版本: {getattr(ak, '__version__', 'unknown')}")
    except Exception:
        pass

    # 提示：如果你在公司/校园网环境，建议先在PowerShell里配置代理再运行：
    # $env:HTTP_PROXY="http://127.0.0.1:7890"; $env:HTTPS_PROXY="http://127.0.0.1:7890"
    # AkShare底层一般会遵循这些环境变量。
    
    # 方法1: 使用get_us_stock_name()获取美股列表
    try:
        print("使用get_us_stock_name()接口获取美股列表...")
        stock_list = _call_with_retry(lambda: ak.get_us_stock_name(), name="get_us_stock_name()")
        
        if stock_list is not None and not stock_list.empty:
            print(f"获取到 {len(stock_list)} 只美股")
            
            result_list = []
            
            # get_us_stock_name() 返回的列通常是: name, cname, symbol
            # 检查实际返回的列名
            print(f"数据列: {stock_list.columns.tolist()}")
            
            # 查找对应的列
            symbol_col = None
            name_en_col = None
            name_cn_col = None
            name_cn = ""
            
            for col in stock_list.columns:
                col_lower = col.lower()
                if 'symbol' in col_lower or '代码' in col or 'code' in col_lower:
                    symbol_col = col
                elif ('name' in col_lower and 'cname' not in col_lower) or '英文名称' in col:
                    name_en_col = col
                elif 'cname' in col_lower or '中文名称' in col or '中文' in col:
                    name_cn_col = col
            
            # 如果找不到symbol列，尝试使用第一列
            if not symbol_col and len(stock_list.columns) > 0:
                symbol_col = stock_list.columns[0]
            
            # 如果找不到name列，尝试使用第二列
            if not name_en_col and len(stock_list.columns) > 1:
                name_en_col = stock_list.columns[1]
            
            print(f"使用列: symbol={symbol_col}, name_en={name_en_col}, name_cn={name_cn_col}")
            
            for idx, row in stock_list.iterrows():
                symbol = str(row[symbol_col]).strip() if symbol_col else ""
                
                # 获取英文名称
                if name_en_col:
                    name_en = str(row[name_en_col]).strip()
                else:
                    name_en = ""
                
                # 获取中文名称（如果有）
                if name_cn_col:
                    name_cn = str(row[name_cn_col]).strip()
                    # 优先使用中文名称，如果没有则使用英文名称
                    name = name_cn if name_cn and name_cn != 'nan' else name_en
                else:
                    name = name_en
                
                if symbol:
                    # 美股代码通常直接使用，如 "AAPL", "MSFT"
                    # ts_code格式可以保持原样或添加交易所后缀（如果需要）
                    ts_code = symbol  # 美股通常不需要后缀，或使用 .US
                    
                    result_list.append({
                        'ts_code': ts_code,
                        'symbol': symbol,
                        'name': name,
                        'name_en': name_en,
                        'name_cn': name_cn if name_cn_col else '',
                        'market': 'US'  # 标记为美股市场
                    })
            
            # 创建DataFrame并保存
            result_df = pd.DataFrame(result_list)
            result_df = result_df.sort_values('symbol')
            
            # 保存为CSV
            output_path = Path(output_path)
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"\n成功保存 {len(result_df)} 只美股信息到: {output_path}")
            print(f"字段: {', '.join(result_df.columns.tolist())}")
            print(f"\n前5条数据预览:")
            print(result_df.head().to_string(index=False))
            
            return result_df
        else:
            print("get_us_stock_name()返回空数据")
    except Exception as e:
        print(f"get_us_stock_name()接口失败: {e}")
        print("尝试其他方法...")
    
    # 方法2: 尝试使用stock_us_spot_em()获取实时行情
    try:
        print("尝试使用stock_us_spot_em()接口获取美股列表...")
        stock_spot = _call_with_retry(lambda: ak.stock_us_spot_em(), name="stock_us_spot_em()")
        
        if stock_spot is not None and not stock_spot.empty:
            print(f"从实时行情接口获取到 {len(stock_spot)} 只美股")
            
            result_list = []
            
            # 查找代码和名称列
            symbol_col = None
            name_col = None
            
            for col in stock_spot.columns:
                col_lower = col.lower()
                if 'symbol' in col_lower or '代码' in col or 'code' in col_lower or '股票代码' in col:
                    symbol_col = col
                if 'name' in col_lower or '名称' in col or '简称' in col or '股票名称' in col:
                    name_col = col
            
            if symbol_col and name_col:
                print(f"使用列: symbol={symbol_col}, name={name_col}")
                
                for idx, row in stock_spot.iterrows():
                    symbol = str(row[symbol_col]).strip()
                    name = str(row[name_col]).strip()
                    
                    if symbol:
                        ts_code = symbol
                        
                        result_list.append({
                            'ts_code': ts_code,
                            'symbol': symbol,
                            'name': name,
                            'name_en': name,
                            'name_cn': '',
                            'market': 'US'
                        })
                
                result_df = pd.DataFrame(result_list)
                result_df = result_df.sort_values('symbol')
                
                output_path = Path(output_path)
                result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                
                print(f"\n成功保存 {len(result_df)} 只美股信息到: {output_path}")
                print(f"字段: {', '.join(result_df.columns.tolist())}")
                print(f"\n前5条数据预览:")
                print(result_df.head().to_string(index=False))
                
                return result_df
            else:
                print(f"警告: 未找到代码或名称列，可用列: {stock_spot.columns.tolist()}")
    except Exception as e:
        print(f"stock_us_spot_em()接口失败: {e}")
    
    # 兜底：读取本地缓存（如果存在）
    if use_cache_on_fail:
        cache_path = Path(output_path)
        if cache_path.exists():
            try:
                cached = pd.read_csv(cache_path, dtype=str)
                if not cached.empty:
                    print(f"警告：在线获取失败，已改用本地缓存: {cache_path}（{len(cached)}行）")
                    return cached
            except Exception as e:
                print(f"读取本地缓存失败: {cache_path}，原因: {e}")

    print("错误：未能获取美股列表。常见原因：网络/代理未配置、目标站点限流/封禁、AkShare数据源临时不可用。")
    return pd.DataFrame()

if __name__ == "__main__":
    # 可以指定输出路径
    import sys
    output_file = sys.argv[1] if len(sys.argv) > 1 else "./data/stocklist_us.csv"
    
    fetch_stocklist_us(output_file)
