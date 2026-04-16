"""
添加彩票期数数据（支持命令行参数）
功能：
1. 从命令行传入真实的开奖数据
2. 参数验证，避免重复添加
3. 自动备份原有数据
4. 支持自定义期号、日期、号码等
"""

import json
import os
import sys
import shutil
from datetime import datetime, timedelta
import argparse


def backup_data(data_file):
    """备份数据文件"""
    if not os.path.exists(data_file):
        return False
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f"{data_file}.backup_{timestamp}"
    
    try:
        shutil.copy2(data_file, backup_file)
        print(f"✅ 数据已备份到: {backup_file}")
        return True
    except Exception as e:
        print(f"⚠️ 备份失败: {e}")
        return False


def validate_zodiac_code(zodiac_code):
    """验证生肖代码"""
    valid_codes = list(range(1, 13))
    if zodiac_code not in valid_codes:
        print(f"❌ 无效的生肖代码: {zodiac_code} (有效范围: 1-12)")
        return False
    return True


def validate_number(number, field_name="号码"):
    """验证号码是否在1-49范围内"""
    if not (1 <= number <= 49):
        print(f"❌ 无效的{field_name}: {number} (有效范围: 1-49)")
        return False
    return True


def validate_period_format(period):
    """验证期号格式"""
    if len(period) != 7 or not period.isdigit():
        print(f"❌ 无效的期号格式: {period} (应为7位数字，如: 2026107)")
        return False
    return True


def check_duplicate(all_records, new_period):
    """检查是否已存在该期数"""
    for record in all_records:
        if record['period'] == new_period:
            print(f"⚠️ 警告: 期号 {new_period} 已存在！")
            print(f"   现有数据: 特码={record['special_number']}, 生肖代码={record['special_zodiac']}")
            return True
    return False


def add_period(args):
    """添加新一期数据"""
    
    print("=" * 60)
    print("📝 添加彩票期数数据")
    print("=" * 60)
    
    data_file = args.file
    
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
    
    # 获取最后一期信息（用于计算下一期日期）
    if all_records:
        last_record = all_records[-1]
        last_period = last_record['period']
        last_draw_time = datetime.strptime(last_record['draw_time'], '%Y-%m-%d %H:%M:%S')
        print(f"   最后一期: {last_period} ({last_record['draw_time']})")
    else:
        last_draw_time = datetime.now()
        print("   ⚠️ 无历史数据，使用当前时间")
    
    # 验证输入参数
    new_period = args.period
    if not validate_period_format(new_period):
        return
    
    # 检查重复
    if check_duplicate(all_records, new_period):
        confirm = input("   是否覆盖已有数据？(y/n): ")
        if confirm.lower() != 'y':
            print("❌ 取消操作")
            return
        # 删除旧记录
        all_records = [r for r in all_records if r['period'] != new_period]
        print("   ✅ 已删除旧记录")
    
    # 验证生肖代码
    special_zodiac = args.zodiac
    if not validate_zodiac_code(special_zodiac):
        return
    
    # 验证特码
    special_number = args.special
    if not validate_number(special_number, "特码"):
        return
    
    # 验证平码
    normal_numbers = args.normal
    if len(normal_numbers) != 6:
        print(f"❌ 平码数量错误: {len(normal_numbers)} (应为6个)")
        return
    
    for num in normal_numbers:
        if not validate_number(num, f"平码({num})"):
            return
    
    # 检查平码是否有重复
    if len(set(normal_numbers)) != 6:
        print(f"❌ 平码不能有重复: {normal_numbers}")
        return
    
    # 检查特码是否与平码重复
    if special_number in normal_numbers:
        print(f"⚠️ 警告: 特码 {special_number} 与平码重复")
        confirm = input("   是否继续？(y/n): ")
        if confirm.lower() != 'y':
            print("❌ 取消操作")
            return
    
    # 处理开奖时间
    if args.time:
        try:
            new_draw_time = datetime.strptime(args.time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"❌ 无效的时间格式: {args.time} (应为: YYYY-MM-DD HH:MM:SS)")
            return
    else:
        # 默认使用上一期+1天
        new_draw_time = last_draw_time + timedelta(days=1)
    
    # 创建新记录
    new_record = {
        "period": new_period,
        "draw_time": new_draw_time.strftime('%Y-%m-%d %H:%M:%S'),
        "normal_numbers": sorted(normal_numbers),
        "special_number": special_number,
        "special_zodiac": special_zodiac
    }
    
    # 显示新增数据
    zodiac_names = {1: '马', 2: '蛇', 3: '龙', 4: '兔', 5: '虎', 6: '牛', 
                   7: '鼠', 8: '猪', 9: '狗', 10: '鸡', 11: '猴', 12: '羊'}
    zodiac_name = zodiac_names.get(special_zodiac, '未知')
    
    print(f"\n➕ 新增数据:")
    print(f"   期数: {new_record['period']}")
    print(f"   开奖时间: {new_record['draw_time']}")
    print(f"   平码: {new_record['normal_numbers']}")
    print(f"   特码: {new_record['special_number']}")
    print(f"   生肖: {zodiac_name} (代码{special_zodiac})")
    
    # 确认添加
    if not args.yes:
        confirm = input("\n确认添加以上数据？(y/n): ")
        if confirm.lower() != 'y':
            print("❌ 取消操作")
            return
    
    # 备份数据
    print(f"\n💾 正在备份数据...")
    backup_data(data_file)
    
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
    print(f"\n📋 最新5期数据:")
    for record in all_records[-5:]:
        z_name = zodiac_names.get(record['special_zodiac'], '未知')
        print(f"   {record['period']} | {record['draw_time']} | 特码: {record['special_number']} | 生肖: {z_name}")
    
    print("\n" + "=" * 60)
    print("✅ 数据添加完成！")
    print("=" * 60)


def main():
    """主函数 - 解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='添加彩票期数数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 添加2026107期数据
  python add_lottery_period.py -p 2026107 -s 15 -z 3 -n 5 12 23 34 41 48
  
  # 指定开奖时间
  python add_lottery_period.py -p 2026107 -s 15 -z 3 -n 5 12 23 34 41 48 -t "2026-04-17 21:32:32"
  
  # 跳过确认（自动模式）
  python add_lottery_period.py -p 2026107 -s 15 -z 3 -n 5 12 23 34 41 48 -y
  
  # 指定数据文件
  python add_lottery_period.py -f 彩票数据.json -p 2026107 -s 15 -z 3 -n 5 12 23 34 41 48
        """
    )
    
    parser.add_argument('-p', '--period', type=str, required=True,
                       help='期号（7位数字，如: 2026107）')
    parser.add_argument('-s', '--special', type=int, required=True,
                       help='特码（1-49）')
    parser.add_argument('-z', '--zodiac', type=int, required=True,
                       help='特码生肖代码（1-12）: 1=马,2=蛇,3=龙,4=兔,5=虎,6=牛,7=鼠,8=猪,9=狗,10=鸡,11=猴,12=羊')
    parser.add_argument('-n', '--normal', type=int, nargs=6, required=True,
                       help='6个平码（1-49，用空格分隔）')
    parser.add_argument('-t', '--time', type=str, default=None,
                       help='开奖时间（格式: YYYY-MM-DD HH:MM:SS），默认为上一期+1天')
    parser.add_argument('-f', '--file', type=str, default='彩票数据.json',
                       help='数据文件路径（默认: 彩票数据.json）')
    parser.add_argument('-y', '--yes', action='store_true',
                       help='跳过确认，直接添加')
    
    args = parser.parse_args()
    add_period(args)


if __name__ == "__main__":
    main()
