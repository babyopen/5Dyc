"""
数据更新和完整性检查工具
功能：
1. 从API获取最新历史数据
2. 与本地数据对比，找出缺失期数
3. 自动补充缺失数据
4. 保存更新后的数据
"""

import pandas as pd
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Set

# 导入现有的类
from predict_next import LotteryDataFetcher, ZodiacRules


class DataManager:
    """数据管理工具类"""
    
    def __init__(self, data_file='lottery_data.json'):
        self.data_file = data_file
        self.fetcher = LotteryDataFetcher()
        self.local_data = []
        self.load_local_data()
    
    def load_local_data(self):
        """加载本地数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    # 处理嵌套列表结构
                    for item in raw_data:
                        if isinstance(item, list):
                            self.local_data.extend(item)
                        elif isinstance(item, dict):
                            self.local_data.append(item)
                print(f"✅ 已加载本地数据: {len(self.local_data)} 条记录")
            except Exception as e:
                print(f"⚠️ 加载本地数据失败: {e}")
                self.local_data = []
        else:
            print("📝 本地数据文件不存在，将创建新文件")
            self.local_data = []
    
    def save_data(self, data_list: List[Dict]):
        """保存数据到JSON文件"""
        try:
            # 按照原有格式保存（嵌套列表）
            data_to_save = [data_list]
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            print(f"✅ 成功保存 {len(data_list)} 条记录到 {self.data_file}")
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
    
    def get_local_periods(self) -> Set[str]:
        """获取本地所有期数"""
        return {record['period'] for record in self.local_data if 'period' in record}
    
    def fetch_and_merge_data(self, start_year: int = 2020, end_year: int = None) -> List[Dict]:
        """从API获取数据并与本地数据合并"""
        if end_year is None:
            end_year = datetime.now().year
        
        print(f"\n📡 正在从API获取 {start_year}-{end_year} 年数据...")
        api_df = self.fetcher.fetch_history(start_year=start_year, end_year=end_year)
        
        if api_df.empty:
            print("❌ 未能从API获取数据")
            return self.local_data
        
        # 转换API数据为字典列表
        api_records = []
        for _, row in api_df.iterrows():
            record = {
                'period': row['period'],
                'draw_time': str(row['draw_time']),
                'normal_numbers': row['normal_numbers'].tolist() if hasattr(row['normal_numbers'], 'tolist') else row['normal_numbers'],
                'special_number': int(row['special_number']),
                'special_zodiac': int(row['special_zodiac'])
            }
            api_records.append(record)
        
        # 获取本地期数集合
        local_periods = self.get_local_periods()
        api_periods = {rec['period'] for rec in api_records}
        
        # 找出新增的记录
        new_records = [rec for rec in api_records if rec['period'] not in local_periods]
        
        print(f"\n📊 数据统计:")
        print(f"   本地记录数: {len(self.local_data)}")
        print(f"   API获取记录数: {len(api_records)}")
        print(f"   新增记录数: {len(new_records)}")
        
        # 合并数据
        merged_data = self.local_data.copy()
        merged_data.extend(new_records)
        
        # 去重并按期数排序
        seen_periods = set()
        unique_data = []
        for record in merged_data:
            if record['period'] not in seen_periods:
                seen_periods.add(record['period'])
                unique_data.append(record)
        
        # 按期数排序
        unique_data.sort(key=lambda x: x['period'])
        
        return unique_data
    
    def check_data_integrity(self, data_list: List[Dict] = None):
        """检查数据完整性"""
        if data_list is None:
            data_list = self.local_data
        
        if not data_list:
            print("❌ 没有数据可检查")
            return
        
        df = pd.DataFrame(data_list)
        
        print("\n" + "="*60)
        print("🔍 数据完整性检查报告")
        print("="*60)
        
        # 基本信息
        print(f"\n📊 基本统计:")
        print(f"   总记录数: {len(df)}")
        
        if 'draw_time' in df.columns:
            df['draw_time_dt'] = pd.to_datetime(df['draw_time'])
            print(f"   时间范围: {df['draw_time_dt'].min()} 到 {df['draw_time_dt'].max()}")
        
        if 'period' in df.columns:
            periods = sorted([int(p) for p in df['period'].unique()])
            print(f"   期数范围: {min(periods)} 到 {max(periods)}")
            
            # 检查期数连续性
            expected_periods = list(range(min(periods), max(periods) + 1))
            missing_periods = set(expected_periods) - set(periods)
            
            if missing_periods:
                print(f"\n⚠️ 发现 {len(missing_periods)} 个缺失期数:")
                missing_list = sorted(list(missing_periods))
                for period in missing_list[:30]:  # 只显示前30个
                    print(f"      缺失: {period}")
                if len(missing_list) > 30:
                    print(f"      ... 还有 {len(missing_list) - 30} 个")
            else:
                print(f"\n✅ 期数连续，无缺失")
        
        # 检查缺失值
        print(f"\n🔍 字段完整性:")
        required_fields = ['period', 'draw_time', 'normal_numbers', 'special_number', 'special_zodiac']
        for field in required_fields:
            if field in df.columns:
                missing_count = df[field].isnull().sum()
                if missing_count > 0:
                    print(f"   ⚠️ {field}: {missing_count} 个缺失值")
                else:
                    print(f"   ✅ {field}: 完整")
            else:
                print(f"   ❌ {field}: 字段不存在")
        
        # 检查重复
        if 'period' in df.columns:
            duplicates = df[df.duplicated(subset=['period'], keep=False)]
            if not duplicates.empty:
                print(f"\n⚠️ 发现 {len(duplicates)} 条重复记录")
            else:
                print(f"\n✅ 无重复记录")
        
        # 显示最新10期
        if len(df) > 0:
            print(f"\n📋 最新10期数据:")
            if 'draw_time' in df.columns:
                df_sorted = df.sort_values('draw_time', ascending=False)
                latest = df_sorted.head(10)[['period', 'draw_time', 'special_number', 'special_zodiac']]
                print(latest.to_string(index=False))
        
        print("\n" + "="*60)
    
    def show_latest(self, n: int = 10):
        """显示最新N期数据"""
        if not self.local_data:
            print("❌ 无数据")
            return
        
        df = pd.DataFrame(self.local_data)
        if 'draw_time' in df.columns:
            df['draw_time_dt'] = pd.to_datetime(df['draw_time'])
            df_sorted = df.sort_values('draw_time_dt', ascending=False)
            print(f"\n📋 最新 {n} 期数据:")
            print(df_sorted.head(n)[['period', 'draw_time', 'special_number', 'special_zodiac']].to_string(index=False))


def main():
    """主函数"""
    print("=" * 60)
    print("🔄 六合彩数据更新工具")
    print("=" * 60)
    
    manager = DataManager()
    
    # 1. 检查当前数据状态
    print("\n【步骤1】检查当前数据状态")
    manager.check_data_integrity()
    
    # 2. 从API获取并合并数据（仅2025-2026年）
    print("\n【步骤2】从API获取2025-2026年数据")
    updated_data = manager.fetch_and_merge_data(start_year=2025, end_year=2026)
    
    # 3. 再次检查数据完整性
    print("\n【步骤3】更新后数据完整性检查")
    manager.check_data_integrity(updated_data)
    
    # 4. 保存更新后的数据
    print("\n【步骤4】保存更新后的数据")
    if len(updated_data) > len(manager.local_data):
        confirm = input(f"发现 {len(updated_data) - len(manager.local_data)} 条新记录，是否保存？(y/n): ")
        if confirm.lower() == 'y':
            manager.save_data(updated_data)
            print("✅ 数据已保存")
        else:
            print("❌ 取消保存")
    else:
        print("ℹ️  没有新数据需要保存")
    
    # 5. 显示最新数据
    print("\n【步骤5】显示最新数据")
    manager.show_latest(10)
    
    print("\n" + "=" * 60)
    print("✅ 数据更新完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
