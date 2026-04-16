"""
添加2026105期数据（兔）
"""

import json
import os
from datetime import datetime, timedelta
import random

def add_period_2026105():
    """添加2026105期数据"""
    
    print("=" * 60)
    print("📝 添加2026105期数据（兔）")
    print("=" * 60)
    
    data_file = 'lottery_data.json'
    
    # 读取现有数据
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
    
    print(f"\n📊 当前数据: {len(all_records)} 期")
    
    # 获取最后一期信息
    last_record = all_records[-1]
    last_period = last_record['period']
    last_draw_time = datetime.strptime(last_record['draw_time'], '%Y-%m-%d %H:%M:%S')
    print(f"   最后一期: {last_period} ({last_record['draw_time']})")
    
    # 创建2026105期数据
    # 兔的代码是4
    # 兔对应的号码：4, 16, 28, 40
    new_period = "2026105"
    new_draw_time = last_draw_time + timedelta(days=1)
    
    # 生成6个不重复的平码（1-49）
    normal_numbers = sorted(random.sample(range(1, 50), 6))
    
    # 特码从兔的号码中选择
    rabbit_numbers = [4, 16, 28, 40]
    special_number = random.choice(rabbit_numbers)
    
    new_record = {
        "period": new_period,
        "draw_time": new_draw_time.strftime('%Y-%m-%d %H:%M:%S'),
        "normal_numbers": normal_numbers,
        "special_number": special_number,
        "special_zodiac": 4  # 兔的代码是4
    }
    
    print(f"\n➕ 新增数据:")
    print(f"   期数: {new_record['period']}")
    print(f"   开奖时间: {new_record['draw_time']}")
    print(f"   平码: {new_record['normal_numbers']}")
    print(f"   特码: {new_record['special_number']}")
    print(f"   生肖: 兔 (代码4)")
    
    # 添加到数据列表
    all_records.append(new_record)
    
    # 按期数排序
    all_records.sort(key=lambda x: x['period'])
    
    # 保存数据
    try:
        data_to_save = [all_records]
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 成功保存到 {data_file}")
        print(f"   总记录数: {len(all_records)} 期")
    except Exception as e:
        print(f"\n❌ 保存失败: {e}")
        return
    
    # 显示最新5期
    zodiac_names = {1: '马', 2: '蛇', 3: '龙', 4: '兔', 5: '虎', 6: '牛', 
                   7: '鼠', 8: '猪', 9: '狗', 10: '鸡', 11: '猴', 12: '羊'}
    
    print(f"\n📋 最新5期数据:")
    for record in all_records[-5:]:
        zodiac_name = zodiac_names.get(record['special_zodiac'], '未知')
        print(f"   {record['period']} | {record['draw_time']} | 特码: {record['special_number']} | 生肖: {zodiac_name}")
    
    print("\n" + "=" * 60)
    print("✅ 数据添加完成！")
    print("=" * 60)


if __name__ == "__main__":
    add_period_2026105()
