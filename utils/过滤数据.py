"""
过滤数据，只保留2025-2026年的数据
"""

import json
import os

def filter_data_by_year(data_file='彩票数据.json', output_file='过滤后数据.json', years=[2025, 2026]):
    """过滤指定年份的数据"""
    
    print("=" * 60)
    print("🔍 数据过滤工具 - 保留指定年份数据")
    print("=" * 60)
    
    # 读取数据
    if not os.path.exists(data_file):
        print(f"❌ 文件 {data_file} 不存在")
        return
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return
    
    # 展平数据
    all_records = []
    for item in raw_data:
        if isinstance(item, list):
            all_records.extend(item)
        elif isinstance(item, dict):
            all_records.append(item)
    
    print(f"\n📊 原始数据统计:")
    print(f"   总记录数: {len(all_records)}")
    
    # 按年份统计
    year_stats = {}
    for record in all_records:
        period = record.get('period', '')
        if len(period) >= 4:
            year = int(period[:4])
            year_stats[year] = year_stats.get(year, 0) + 1
    
    print(f"\n📅 各年份记录数:")
    for year in sorted(year_stats.keys()):
        print(f"   {year}年: {year_stats[year]} 条")
    
    # 过滤指定年份
    filtered_records = []
    for record in all_records:
        period = record.get('period', '')
        if len(period) >= 4:
            year = int(period[:4])
            if year in years:
                filtered_records.append(record)
    
    print(f"\n✅ 过滤后数据:")
    print(f"   保留年份: {years}")
    print(f"   保留记录数: {len(filtered_records)}")
    print(f"   移除记录数: {len(all_records) - len(filtered_records)}")
    
    # 按期数排序
    filtered_records.sort(key=lambda x: x['period'])
    
    # 保存过滤后的数据
    try:
        # 按照原有格式保存（嵌套列表）
        data_to_save = [filtered_records]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        print(f"\n💾 已保存到: {output_file}")
        print(f"   文件大小: {os.path.getsize(output_file) / 1024:.2f} KB")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return
    
    # 显示最新10期
    if filtered_records:
        print(f"\n📋 最新10期数据:")
        latest_10 = filtered_records[-10:]
        for record in latest_10:
            print(f"   {record['period']} | {record['draw_time']} | 特码: {record['special_number']} | 生肖: {record['special_zodiac']}")
    
    print("\n" + "=" * 60)
    print("✅ 数据过滤完成！")
    print("=" * 60)
    
    # 询问是否替换原文件
    confirm = input(f"\n是否用过滤后的数据替换原文件 {data_file}？(y/n): ")
    if confirm.lower() == 'y':
        try:
            import shutil
            # 备份原文件
            backup_file = data_file.replace('.json', '_backup.json')
            shutil.copy2(data_file, backup_file)
            print(f"✅ 已备份原文件到: {backup_file}")
            
            # 替换原文件
            shutil.copy2(output_file, data_file)
            print(f"✅ 已替换原文件: {data_file}")
        except Exception as e:
            print(f"❌ 替换文件失败: {e}")
    else:
        print("ℹ️  保留原文件不变")


if __name__ == "__main__":
    filter_data_by_year(
        data_file='彩票数据.json',
        output_file='过滤后数据.json',
        years=[2025, 2026]
    )
