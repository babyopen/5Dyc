"""查询特定期数的开奖信息"""

import pandas as pd
import requests
import time
from datetime import datetime

class LotteryDataFetcher:
    API_URL_TEMPLATE = "https://history.macaumarksix.com/history/macaujc2/y/{year}"
    ZODIAC_NAME_MAP = {
        "马": 1, "蛇": 2, "龙": 3, "兔": 4, "虎": 5, "牛": 6,
        "鼠": 7, "猪": 8, "狗": 9, "鸡": 10, "猴": 11, "羊": 12,
        "馬": 1, "龍": 3, "豬": 8, "雞": 10,
    }

    def __init__(self, timeout: int = 30):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'ZodiacPrediction/2.0', 'Accept': 'application/json'})
        self.timeout = timeout

    def fetch_history(self, start_year: int = 2020, end_year: int = None) -> pd.DataFrame:
        if end_year is None:
            end_year = datetime.now().year
        all_records = []
        for year in range(start_year, end_year + 1):
            print(f"📡 获取 {year} 年数据...")
            data = self._fetch_year(year)
            if data:
                all_records.extend(data)
            time.sleep(0.5)
        if not all_records:
            return pd.DataFrame()
        df = pd.DataFrame(all_records)
        df['draw_time'] = pd.to_datetime(df['draw_time'])
        df = df.sort_values('draw_time').reset_index(drop=True)
        return df

    def _fetch_year(self, year: int) -> list:
        url = self.API_URL_TEMPLATE.format(year=year)
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if not data.get('result') or data.get('code') != 200:
                return []
            records = data.get('data', [])
            parsed = []
            for item in records:
                expect = item.get('expect', '')
                open_time = item.get('openTime', '')
                open_code = item.get('openCode', '')
                zodiac_str = item.get('zodiac', '')
                if not open_code:
                    continue
                nums = [int(x.strip()) for x in open_code.split(',') if x.strip()]
                if len(nums) < 7:
                    continue
                normal_nums = nums[:6]
                special_num = nums[6]
                zodiac_names = [z.strip() for z in zodiac_str.split(',')] if zodiac_str else []
                special_zodiac = self.ZODIAC_NAME_MAP.get(zodiac_names[6], 0) if len(zodiac_names) >= 7 else 0
                parsed.append({
                    'period': expect,
                    'draw_time': open_time,
                    'normal_numbers': normal_nums,
                    'special_number': special_num,
                    'zodiac_names': zodiac_names if len(zodiac_names) >= 7 else [],
                })
            return parsed
        except Exception as e:
            print(f"   ⚠️ {year} 年异常: {e}")
            return []


def query_period(period_num: str):
    """查询特定期数"""
    print(f"🔍 查询第 {period_num} 期开奖信息...")
    
    # 先尝试从本地数据文件读取
    import json
    import os
    local_data = []
    if os.path.exists('lottery_data.json'):
        try:
            with open('lottery_data.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    for record in item:
                        if record['period'] == period_num:
                            print("\n" + "="*60)
                            print(f"🎲 第 {period_num} 期开奖结果")
                            print("="*60)
                            print(f"📅 开奖时间: {record['draw_time']}")
                            print(f"🎯 平码: {', '.join(map(str, record['normal_numbers']))}")
                            print(f"🏆 特码: {record['special_number']}")
                            print(f"🐎 特码生肖: {ZodiacRules.CODE_TO_NAME[record['special_zodiac']]}")
                            print("="*60)
                            return
        except:
            pass
    
    # 否则从API获取
    fetcher = LotteryDataFetcher()
    df = fetcher.fetch_history(start_year=2025, end_year=2026)
    
    if df.empty:
        print("❌ 未获取到数据")
        return
    
    # 查找特定期数
    result = df[df['period'] == period_num]
    
    if result.empty:
        print(f"❌ 未找到第 {period_num} 期数据")
        print("\n可用的最新期数:")
        print(df.tail(10)[['period', 'draw_time', 'special_number']])
        return
    
    # 显示结果
    row = result.iloc[0]
    print("\n" + "="*60)
    print(f"🎲 第 {period_num} 期开奖结果")
    print("="*60)
    print(f"📅 开奖时间: {row['draw_time']}")
    print(f"🎯 平码: {', '.join(map(str, row['normal_numbers']))}")
    print(f"🏆 特码: {row['special_number']}")
    if len(row['zodiac_names']) >= 7:
        print(f"🐎 特码生肖: {row['zodiac_names'][6]}")
    print("="*60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query_period(sys.argv[1])
    else:
        query_period("2026099")
