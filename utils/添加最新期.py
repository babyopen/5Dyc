"""添加最新一期数据"""

import pandas as pd
import json
import os
from datetime import datetime

class DataManager:
    def __init__(self, data_file='彩票数据.json'):
        self.data_file = data_file
        self.load_data()
    
    def load_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.data = [pd.DataFrame.from_dict(item) for item in data]
            except:
                self.data = []
        else:
            self.data = []
    
    def save_data(self):
        data_to_save = [df.to_dict('records') for df in self.data]
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    
    def add_period(self, period, draw_time, normal_numbers, special_number, special_zodiac):
        new_record = {
            'period': period,
            'draw_time': draw_time,
            'normal_numbers': normal_numbers,
            'special_number': special_number,
            'special_zodiac': special_zodiac
        }
        
        # 检查是否已存在
        for df in self.data:
            if period in df['period'].values:
                print(f"❌ 第 {period} 期已存在")
                return False
        
        # 添加到新的DataFrame
        new_df = pd.DataFrame([new_record])
        self.data.append(new_df)
        self.save_data()
        print(f"✅ 成功添加第 {period} 期数据")
        return True
    
    def show_latest(self, n=10):
        if not self.data:
            print("❌ 无数据")
            return
        
        all_data = pd.concat(self.data, ignore_index=True)
        all_data['draw_time'] = pd.to_datetime(all_data['draw_time'])
        all_data = all_data.sort_values('draw_time', ascending=False)
        
        print(f"📋 最新 {n} 期数据:")
        print(all_data.head(n)[['period', 'draw_time', 'special_number', 'special_zodiac']])


def main():
    print("=" * 60)
    print("📝 添加2026年最新5期数据")
    print("=" * 60)
    
    manager = DataManager()
    
    # 2026年最新5期数据
    periods_data = [
        {
            "period": "2026104",
            "draw_time": "2026-04-15 21:32:32",
            "normal_numbers": [33, 12, 24, 36, 48, 18],  # 假设平码
            "special_number": 1,  # 马对应的号码
            "special_zodiac": 1  # 马的代码
        },
        {
            "period": "2026103",
            "draw_time": "2026-04-14 21:32:32",
            "normal_numbers": [33, 12, 24, 36, 48, 18],  # 假设平码
            "special_number": 6,  # 牛对应的号码
            "special_zodiac": 6  # 牛的代码
        },
        {
            "period": "2026102",
            "draw_time": "2026-04-13 21:32:32",
            "normal_numbers": [33, 12, 24, 36, 48, 18],  # 假设平码
            "special_number": 8,  # 猪对应的号码
            "special_zodiac": 8  # 猪的代码
        },
        {
            "period": "2026101",
            "draw_time": "2026-04-12 21:32:32",
            "normal_numbers": [33, 12, 24, 36, 48, 18],  # 假设平码
            "special_number": 3,  # 龙对应的号码
            "special_zodiac": 3  # 龙的代码
        },
        {
            "period": "2026100",
            "draw_time": "2026-04-11 21:32:32",
            "normal_numbers": [33, 12, 24, 36, 48, 18],  # 假设平码
            "special_number": 9,  # 狗对应的号码
            "special_zodiac": 9  # 狗的代码
        }
    ]
    
    # 批量添加数据
    for data in periods_data:
        manager.add_period(
            data["period"],
            data["draw_time"],
            data["normal_numbers"],
            data["special_number"],
            data["special_zodiac"]
        )
    
    # 显示最新数据
    print("\n" + "-" * 60)
    manager.show_latest(10)


if __name__ == "__main__":
    main()
