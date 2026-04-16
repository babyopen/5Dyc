"""生肖预测模型核心代码"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder


class ZodiacRules:
    """六合彩生肖、号码、五行规则（马年基准轮转）"""

    ZODIAC = ["马", "蛇", "龙", "兔", "虎", "牛", "鼠", "猪", "狗", "鸡", "猴", "羊"]
    ZODIAC_TO_CODE = {name: i + 1 for i, name in enumerate(ZODIAC)}
    CODE_TO_NAME = {v: k for k, v in ZODIAC_TO_CODE.items()}

    NUMBER_WUXING = {
        1: "水", 2: "火", 3: "火", 4: "金", 5: "金", 6: "土", 7: "土", 8: "木", 9: "木", 10: "火",
        11: "火", 12: "金", 13: "水", 14: "火", 15: "水", 16: "木", 17: "木", 18: "火", 19: "火", 20: "土",
        21: "土", 22: "水", 23: "水", 24: "木", 25: "木", 26: "金", 27: "金", 28: "土", 29: "土", 30: "水",
        31: "水", 32: "火", 33: "火", 34: "金", 35: "金", 36: "土", 37: "土", 38: "水", 39: "水", 40: "火",
        41: "火", 42: "金", 43: "金", 44: "水", 45: "水", 46: "木", 47: "木", 48: "火", 49: "火"
    }

    BASE_ALLOCATION_HORSE = {
        "马": [1, 13, 25, 37, 49],
        "蛇": [2, 14, 26, 38],
        "龙": [3, 15, 27, 39],
        "兔": [4, 16, 28, 40],
        "虎": [5, 17, 29, 41],
        "牛": [6, 18, 30, 42],
        "鼠": [7, 19, 31, 43],
        "猪": [8, 20, 32, 44],
        "狗": [9, 21, 33, 45],
        "鸡": [10, 22, 34, 46],
        "猴": [11, 23, 35, 47],
        "羊": [12, 24, 36, 48],
    }

    @classmethod
    def get_year_zodiac(cls, year: int) -> str:
        base_year = 2026
        base_zodiac = "马"
        base_idx = cls.ZODIAC.index(base_zodiac)
        idx = (base_idx - (year - base_year)) % 12
        return cls.ZODIAC[idx]

    @classmethod
    def get_zodiac_number_map(cls, year: int) -> Dict[int, int]:
        current_zodiac = cls.get_year_zodiac(year)
        offset = (cls.ZODIAC.index(current_zodiac) - cls.ZODIAC.index("马")) % 12
        mapping = {}
        for i, base_z in enumerate(cls.ZODIAC):
            target_z = cls.ZODIAC[(i + offset) % 12]
            target_code = cls.ZODIAC_TO_CODE[target_z]
            for num in cls.BASE_ALLOCATION_HORSE[base_z]:
                mapping[num] = target_code
        return mapping

    @classmethod
    def get_zodiac_numbers(cls, zodiac_code: int, year: int) -> List[int]:
        current_zodiac = cls.get_year_zodiac(year)
        offset = (cls.ZODIAC.index(current_zodiac) - cls.ZODIAC.index("马")) % 12
        for i, base_z in enumerate(cls.ZODIAC):
            target_z = cls.ZODIAC[(i + offset) % 12]
            if cls.ZODIAC_TO_CODE[target_z] == zodiac_code:
                return cls.BASE_ALLOCATION_HORSE[base_z].copy()
        return []

    @classmethod
    def number_to_zodiac_code(cls, number: int, year: int) -> int:
        return cls.get_zodiac_number_map(year).get(number, 0)

    @classmethod
    def number_to_wuxing(cls, number: int) -> str:
        return cls.NUMBER_WUXING.get(number, "未知")

    @classmethod
    def zodiac_code_to_typical_wuxing(cls, zodiac_code: int, year: int) -> str:
        nums = cls.get_zodiac_numbers(zodiac_code, year)
        return cls.number_to_wuxing(nums[0]) if nums else "未知"


class FeatureEngineer:
    """特征工程"""

    def __init__(self):
        self.history_df = None
        self.decay_factor = 0.9

    def build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        df = df.sort_values('draw_time').reset_index(drop=True)
        self.history_df = df.copy()
        features_list, labels = [], []
        for idx in range(10, len(df)):
            feat = self._build_single_feature(idx)
            features_list.append(feat)
            labels.append(df.loc[idx, 'special_zodiac'])
        X = pd.DataFrame(features_list)
        y = np.array(labels)
        return X, y

    def _build_single_feature(self, current_idx: int) -> Dict:
        feature = {}
        df = self.history_df
        hist = df.iloc[:current_idx].copy()
        current = df.iloc[current_idx]
        last = hist.iloc[-1] if len(hist) > 0 else None
        zodiac_seq = hist['special_zodiac'].tolist()
        current_year = current['draw_time'].year

        for z in range(1, 13):
            missing = self._calc_missing(zodiac_seq, z)
            feature[f'missing_{z}'] = missing
            max_miss = self._calc_max_missing(zodiac_seq, z)
            feature[f'missing_ratio_{z}'] = missing / max_miss if max_miss > 0 else 0
            feature[f'wcount_10_{z}'] = self._weighted_count_recent(zodiac_seq, z, 10)
            feature[f'wcount_20_{z}'] = self._weighted_count_recent(zodiac_seq, z, 20)
            feature[f'wcount_50_{z}'] = self._weighted_count_recent(zodiac_seq, z, 50)
            feature[f'count_10_{z}'] = self._count_recent(zodiac_seq, z, 10)
            feature[f'count_20_{z}'] = self._count_recent(zodiac_seq, z, 20)
            feature[f'streak_{z}'] = self._calc_streak(zodiac_seq, z)
            feature[f'streak_break_{z}'] = self._is_streak_break(zodiac_seq, z)

        weighted_counts = {z: self._weighted_count_recent(zodiac_seq, z, 20) for z in range(1, 13)}
        sorted_rank = sorted(weighted_counts.items(), key=lambda x: x[1], reverse=True)
        rank_dict = {z: i+1 for i, (z, _) in enumerate(sorted_rank)}
        for z in range(1, 13):
            feature[f'hot_rank_{z}'] = rank_dict.get(z, 12)

        if last is not None:
            last_zodiac = last['special_zodiac']
            last_year = last['draw_time'].year
            for z in range(1, 13):
                feature[f'pos_gap_{z}'] = abs(z - last_zodiac)
                wx_last = ZodiacRules.zodiac_code_to_typical_wuxing(last_zodiac, last_year)
                wx_curr = ZodiacRules.zodiac_code_to_typical_wuxing(z, current_year)
                feature[f'wuxing_rel_{z}'] = self._wuxing_relation(wx_last, wx_curr)
                feature[f'odd_even_same_{z}'] = 1 if (z % 2) == (last_zodiac % 2) else 0
                last_size = 1 if last_zodiac <= 6 else 2
                curr_size = 1 if z <= 6 else 2
                feature[f'size_same_{z}'] = 1 if last_size == curr_size else 0
                def zone(zc): return 1 if zc <= 4 else (2 if zc <= 8 else 3)
                feature[f'zone_same_{z}'] = 1 if zone(z) == zone(last_zodiac) else 0
                last_head = 0 if last_zodiac <= 9 else 1
                curr_head =