# 获取所有A股股票列表
# 使用akshare获取股票列表，包含ts_code, symbol, name, area, industry字段

import akshare as ak
import pandas as pd
from pathlib import Path
import time

def fetch_stocklist(output_path: str = "stocklist.csv"):
    """
    获取所有A股股票列表并保存为CSV文件
    
    Args:
        output_path: 输出CSV文件路径，默认为stocklist.csv
    """
    print("正在获取A股股票列表...")
    
    # 方法1: 尝试使用stock_zh_a_spot_em()获取实时行情（可能包含更多信息）
    try:
        print("尝试使用实时行情接口获取股票列表...")
        stock_spot = ak.stock_zh_a_spot_em()
        if stock_spot is not None and not stock_spot.empty:
            print(f"从实时行情接口获取到 {len(stock_spot)} 只股票")
            # 检查是否有地区和行业字段
            if '代码' in stock_spot.columns and '名称' in stock_spot.columns:
                stock_list = stock_spot[['代码', '名称']].copy()
                stock_list.columns = ['code', 'name']
            else:
                stock_list = None
        else:
            stock_list = None
    except Exception as e:
        print(f"实时行情接口失败: {e}")
        stock_list = None
    
    # 方法2: 如果方法1失败，使用stock_info_a_code_name()
    if stock_list is None or stock_list.empty:
        print("使用stock_info_a_code_name()接口获取股票列表...")
        stock_list = ak.stock_info_a_code_name()
        if stock_list is None or stock_list.empty:
            print("错误：未能获取股票列表")
            return
    
    print(f"获取到 {len(stock_list)} 只股票")
    
    # 准备结果DataFrame
    result_list = []
    
    print("正在获取每只股票的详细信息（地区、行业）...")
    total = len(stock_list)
    
    for idx, row in stock_list.iterrows():
        code = str(row['code']).zfill(6)  # 确保是6位代码
        name = row['name']
        
        # 根据代码判断交易所，生成ts_code
        if code.startswith(('60', '68')):
            ts_code = f"{code}.SH"
            default_area = "上海"
        elif code.startswith(('00', '30', '301')):
            ts_code = f"{code}.SZ"
            default_area = "深圳"
        elif code.startswith(('4', '8')):
            ts_code = f"{code}.BJ"
            default_area = "北京"
        else:
            ts_code = f"{code}.SZ"  # 默认深圳
            default_area = "深圳"
        
        # 初始化地区和行业
        area = default_area
        industry = "未知"
        
        try:
            # 获取个股详细信息
            stock_info = ak.stock_individual_info_em(symbol=code)
            
            if stock_info is not None and not stock_info.empty:
                # 将信息转换为字典便于查找
                info_dict = dict(zip(stock_info.iloc[:, 0], stock_info.iloc[:, 1]))
                
                # 提取地区信息
                area_keywords = ['所属地区', '地区', '注册地址', '办公地址', '省份']
                for keyword in area_keywords:
                    if keyword in info_dict:
                        area_str = str(info_dict[keyword]).strip()
                        if area_str and area_str != 'nan':
                            # 提取省份或城市
                            if '省' in area_str:
                                area = area_str.split('省')[0] + '省'
                            elif '市' in area_str:
                                area = area_str.split('市')[0] + '市'
                            elif '自治区' in area_str:
                                area = area_str.split('自治区')[0] + '自治区'
                            elif '特别行政区' in area_str:
                                area = area_str.split('特别行政区')[0] + '特别行政区'
                            else:
                                area = area_str
                            break
                
                # 提取行业信息
                industry_keywords = ['所属行业', '行业', '行业分类', '主营业务', '概念']
                for keyword in industry_keywords:
                    if keyword in info_dict:
                        industry_str = str(info_dict[keyword]).strip()
                        if industry_str and industry_str != 'nan':
                            industry = industry_str
                            break
            
            # 添加延迟避免请求过快
            if (idx + 1) % 50 == 0:
                time.sleep(0.5)  # 每50只股票暂停0.5秒
                
        except Exception as e:
            # 如果获取详细信息失败，使用默认值
            pass
        
        result_list.append({
            'ts_code': ts_code,
            'symbol': code,
            'name': name,
            'area': area,
            'industry': industry
        })
        
        # 显示进度
        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"  进度: {idx + 1}/{total} ({100*(idx+1)/total:.1f}%)")
    
    # 创建DataFrame并保存
    result_df = pd.DataFrame(result_list)
    result_df = result_df.sort_values('symbol')  # 按代码排序
    
    # 保存为CSV
    output_path = Path(output_path)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n成功保存 {len(result_df)} 只股票信息到: {output_path}")
    print(f"字段: {', '.join(result_df.columns.tolist())}")
    print(f"\n前5条数据预览:")
    print(result_df.head().to_string(index=False))
    
    return result_df

if __name__ == "__main__":
    # 可以指定输出路径
    import sys
    output_file = sys.argv[1] if len(sys.argv) > 1 else "./data/stocklist.csv"
    
    fetch_stocklist(output_file)