# 获取所有港股股票列表
# 使用akshare获取港股列表，包含ts_code, symbol, name等字段

import akshare as ak
import pandas as pd
from pathlib import Path
import time

def fetch_stocklist_hk(output_path: str = "stocklist_hk.csv"):
    """
    获取所有港股股票列表并保存为CSV文件
    
    Args:
        output_path: 输出CSV文件路径，默认为stocklist_hk.csv
    """
    print("正在获取港股股票列表...")
    
    # 方法1: 尝试使用stock_hk_spot_em()获取实时行情
    try:
        print("尝试使用实时行情接口获取港股列表...")
        stock_spot = ak.stock_hk_spot_em()
        if stock_spot is not None and not stock_spot.empty:
            print(f"从实时行情接口获取到 {len(stock_spot)} 只港股")
            
            # 检查列名并准备数据
            result_list = []
            
            # 常见的列名映射
            code_col = None
            name_col = None
            
            # 查找代码列
            for col in stock_spot.columns:
                if '代码' in col or 'code' in col.lower() or 'symbol' in col.lower():
                    code_col = col
                if '名称' in col or 'name' in col.lower() or '简称' in col:
                    name_col = col
            
            if code_col and name_col:
                print(f"使用列: 代码={code_col}, 名称={name_col}")
                
                for idx, row in stock_spot.iterrows():
                    code = str(row[code_col]).strip()
                    name = str(row[name_col]).strip()
                    
                    # 港股代码格式通常是5位数字，如 "00700"
                    # 生成ts_code格式，港股通常以 .HK 结尾
                    if code:
                        # 确保代码是5位数字
                        code = code.zfill(5)
                        ts_code = f"{code}.HK"
                    else:
                        continue
                    
                    result_list.append({
                        'ts_code': ts_code,
                        'symbol': code,
                        'name': name,
                        'market': 'HK'  # 标记为港股市场
                    })
                
                # 创建DataFrame并保存
                result_df = pd.DataFrame(result_list)
                result_df = result_df.sort_values('symbol')
                
                # 保存为CSV
                output_path = Path(output_path)
                result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                
                print(f"\n成功保存 {len(result_df)} 只港股信息到: {output_path}")
                print(f"字段: {', '.join(result_df.columns.tolist())}")
                print(f"\n前5条数据预览:")
                print(result_df.head().to_string(index=False))
                
                return result_df
            else:
                print(f"警告: 未找到代码或名称列，可用列: {stock_spot.columns.tolist()}")
                # 如果找不到标准列，尝试使用前两列
                if len(stock_spot.columns) >= 2:
                    code_col = stock_spot.columns[0]
                    name_col = stock_spot.columns[1]
                    print(f"尝试使用前两列: {code_col}, {name_col}")
                    
                    result_list = []
                    for idx, row in stock_spot.iterrows():
                        code = str(row[code_col]).strip()
                        name = str(row[name_col]).strip()
                        
                        if code:
                            code = code.zfill(5)
                            ts_code = f"{code}.HK"
                            
                            result_list.append({
                                'ts_code': ts_code,
                                'symbol': code,
                                'name': name,
                                'market': 'HK'
                            })
                    
                    result_df = pd.DataFrame(result_list)
                    result_df = result_df.sort_values('symbol')
                    
                    output_path = Path(output_path)
                    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    
                    print(f"\n成功保存 {len(result_df)} 只港股信息到: {output_path}")
                    print(f"字段: {', '.join(result_df.columns.tolist())}")
                    print(f"\n前5条数据预览:")
                    print(result_df.head().to_string(index=False))
                    
                    return result_df
        else:
            print("实时行情接口返回空数据")
    except Exception as e:
        print(f"实时行情接口失败: {e}")
        print("尝试其他方法...")
    
    # 方法2: 尝试使用stock_hk_spot()接口
    try:
        print("尝试使用stock_hk_spot()接口...")
        stock_list = ak.stock_hk_spot()
        if stock_list is not None and not stock_list.empty:
            print(f"获取到 {len(stock_list)} 只港股")
            
            result_list = []
            for idx, row in stock_list.iterrows():
                # 根据实际返回的列名调整
                code = str(row.iloc[0]).strip() if len(row) > 0 else ""
                name = str(row.iloc[1]).strip() if len(row) > 1 else ""
                
                if code:
                    code = code.zfill(5)
                    ts_code = f"{code}.HK"
                    
                    result_list.append({
                        'ts_code': ts_code,
                        'symbol': code,
                        'name': name,
                        'market': 'HK'
                    })
            
            result_df = pd.DataFrame(result_list)
            result_df = result_df.sort_values('symbol')
            
            output_path = Path(output_path)
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"\n成功保存 {len(result_df)} 只港股信息到: {output_path}")
            print(f"字段: {', '.join(result_df.columns.tolist())}")
            print(f"\n前5条数据预览:")
            print(result_df.head().to_string(index=False))
            
            return result_df
    except Exception as e:
        print(f"stock_hk_spot()接口失败: {e}")
    
    print("错误：未能获取港股列表，请检查网络连接或API是否可用")
    return pd.DataFrame()

if __name__ == "__main__":
    # 可以指定输出路径
    import sys
    output_file = sys.argv[1] if len(sys.argv) > 1 else "./data/stocklist_hk.csv"
    
    fetch_stocklist_hk(output_file)
