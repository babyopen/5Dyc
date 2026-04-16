"""快速预测下一期"""

import pandas as pd
import numpy as np
import requests
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import Counter

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')


# ==================== 第一部分：生肖规则引擎 ====================
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


# ==================== 第二部分：数据获取 ====================
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
        df['seq_period'] = range(1, len(df) + 1)
        print(f"✅ 获取 {len(df)} 期数据")
        return df

    def _fetch_year(self, year: int) -> List[Dict]:
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
                if special_zodiac == 0 and open_time:
                    try:
                        dt = datetime.strptime(open_time, "%Y-%m-%d %H:%M:%S")
                        special_zodiac = ZodiacRules.number_to_zodiac_code(special_num, dt.year)
                    except:
                        pass
                if special_zodiac == 0:
                    continue
                normal_zodiacs = []
                for i, num in enumerate(normal_nums):
                    zc = self.ZODIAC_NAME_MAP.get(zodiac_names[i], 0) if i < len(zodiac_names) else 0
                    if zc == 0 and open_time:
                        try:
                            dt = datetime.strptime(open_time, "%Y-%m-%d %H:%M:%S")
                            zc = ZodiacRules.number_to_zodiac_code(num, dt.year)
                        except:
                            zc = 0
                    normal_zodiacs.append(zc)
                parsed.append({
                    'period': expect,
                    'draw_time': open_time,
                    'normal_numbers': normal_nums,
                    'special_number': special_num,
                    'normal_zodiacs': normal_zodiacs,
                    'special_zodiac': special_zodiac,
                })
            return parsed
        except Exception as e:
            print(f"   ⚠️ {year} 年异常: {e}")
            return []


# ==================== 第三部分：特征工程 ====================
class FeatureEngineer:
    """增强特征工程：包含平码信息、时间衰减、周期特征等"""

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
                curr_head = 0 if z <= 9 else 1
                feature[f'head_same_{z}'] = 1 if last_head == curr_head else 0
                last_tail = last_zodiac % 10 if last_zodiac % 10 != 0 else 10
                curr_tail = z % 10 if z % 10 != 0 else 10
                feature[f'tail_same_{z}'] = 1 if last_tail == curr_tail else 0

            if 'normal_zodiacs' in last:
                normal_zodiacs = last['normal_zodiacs']
                counter = Counter(normal_zodiacs)
                feature['normal_has_special'] = 1 if last_zodiac in normal_zodiacs else 0
                feature['normal_special_count'] = counter.get(last_zodiac, 0)
                feature['normal_unique_zodiac'] = len(counter)
                total = len(normal_zodiacs)
                probs = [c/total for c in counter.values()]
                feature['normal_entropy'] = -sum(p * np.log(p) for p in probs if p > 0)
                normal_wuxings = [ZodiacRules.zodiac_code_to_typical_wuxing(z, last_year) for z in normal_zodiacs]
                wx_counter = Counter(normal_wuxings)
                for wx in ["金","木","水","火","土"]:
                    feature[f'normal_wx_{wx}_ratio'] = wx_counter.get(wx, 0) / 6.0

        feature['weekday'] = current['draw_time'].weekday()
        feature['month'] = current['draw_time'].month
        feature['lunar_month_sim'] = (current['draw_time'].month + 6) % 12

        for z in range(1, 13):
            intervals = self._get_intervals(zodiac_seq, z)
            feature[f'interval_mean_{z}'] = np.mean(intervals) if intervals else 0
            feature[f'interval_std_{z}'] = np.std(intervals) if len(intervals) > 1 else 0
            if len(zodiac_seq) >= 2:
                prev_missing = self._calc_missing(zodiac_seq[:-1], z)
                feature[f'missing_change_{z}'] = feature[f'missing_{z}'] - prev_missing
            else:
                feature[f'missing_change_{z}'] = 0

        return feature

    def _calc_missing(self, seq, target):
        for i, v in enumerate(reversed(seq)):
            if v == target:
                return i
        return len(seq)

    def _calc_max_missing(self, seq, target):
        max_gap = cur = 0
        for v in seq:
            if v == target:
                max_gap = max(max_gap, cur)
                cur = 0
            else:
                cur += 1
        return max(max_gap, cur)

    def _count_recent(self, seq, target, n):
        return seq[-n:].count(target) if len(seq) >= n else seq.count(target)

    def _weighted_count_recent(self, seq, target, n):
        recent = seq[-n:] if len(seq) >= n else seq
        weights = [self.decay_factor ** i for i in range(len(recent))][::-1]
        return sum(w for v, w in zip(recent, weights) if v == target)

    def _calc_streak(self, seq, target):
        if not seq or seq[-1] != target:
            return 0
        streak = 0
        for v in reversed(seq):
            if v == target:
                streak += 1
            else:
                break
        return streak

    def _is_streak_break(self, seq, target):
        if len(seq) < 2:
            return 0
        return 1 if (seq[-2] != target and seq[-1] == target) else 0

    def _get_intervals(self, seq, target):
        pos = [i for i, v in enumerate(seq) if v == target]
        return [pos[i] - pos[i-1] for i in range(1, len(pos))]

    def _wuxing_relation(self, wx1, wx2):
        if wx1 == "未知" or wx2 == "未知":
            return 1
        if wx1 == wx2:
            return 1
        sheng = {("金","水"), ("水","木"), ("木","火"), ("火","土"), ("土","金")}
        ke = {("金","木"), ("木","土"), ("土","水"), ("水","火"), ("火","金")}
        if (wx1, wx2) in sheng: return 2
        if (wx1, wx2) in ke: return 0
        if (wx2, wx1) in sheng: return 0
        if (wx2, wx1) in ke: return 2
        return 1


# ==================== 第四部分：模型预测 ====================
class ZodiacPredictor:
    def __init__(self, n_models: int = 3):
        self.models = []
        self.calibrators = []
        self.n_models = n_models
        self.feature_engineer = FeatureEngineer()
        self.label_encoder = LabelEncoder()
        self.missing_max_dict = {}

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        X, y = self.feature_engineer.build_features(df)
        y_encoded = self.label_encoder.fit_transform(y)
        zodiac_seq = df['special_zodiac'].tolist()
        for z in range(1, 13):
            self.missing_max_dict[z] = self.feature_engineer._calc_max_missing(zodiac_seq, z)
        return X, y_encoded

    def train(self, X: pd.DataFrame, y: np.ndarray):
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        freq = counts / total
        
        # 热号权重大，冷号权重小
        max_freq = max(freq)
        scale_pos_weight = {cls: 0.5 + 1.5 * (f / max_freq) for cls, f in zip(unique, freq)}
        
        # 打印类别权重（需要反向映射）
        decoded_unique = self.label_encoder.inverse_transform(unique)
        weight_dict = {ZodiacRules.CODE_TO_NAME[int(dc)]: f'{scale_pos_weight[ec]:.2f}' for ec, dc in zip(unique, decoded_unique)}
        print(f"   类别权重: {weight_dict}")
        
        seeds = [42, 123, 888, 2023, 777][:self.n_models]
        for seed in seeds:
            print(f"   🔹 训练模型 (seed={seed})...")
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=12,
                max_depth=6,
                learning_rate=0.05,
                n_estimators=250,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                eval_metric='mlogloss',
                use_label_encoder=False,
            )
            sample_weights = np.array([scale_pos_weight[yi] for yi in y])
            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=3)
            for train_idx, val_idx in tscv.split(X):
                if len(train_idx) < 50:
                    continue
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                model.fit(X_tr, y_tr, sample_weight=sample_weights[train_idx],
                          eval_set=[(X_val, y_val)], verbose=False)
            model.fit(X, y, sample_weight=sample_weights, verbose=False)

            self.models.append(model)
            self.calibrators.append(None)

        print(f"✅ 集成 {self.n_models} 个模型训练完成（热号加权策略）")

    def predict_proba_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        all_proba = []
        for model in self.models:
            proba = model.predict_proba(X)
            all_proba.append(proba)
        return np.mean(all_proba, axis=0)

    def predict_next(self, all_history_df: pd.DataFrame) -> Dict[int, float]:
        temp_eng = FeatureEngineer()
        temp_eng.history_df = all_history_df.sort_values('draw_time').reset_index(drop=True)
        latest_idx = len(temp_eng.history_df) - 1
        latest_feat = temp_eng._build_single_feature(latest_idx)
        X_latest = pd.DataFrame([latest_feat]).fillna(0)

        if hasattr(self.models[0], 'feature_names_in_'):
            for col in self.models[0].feature_names_in_:
                if col not in X_latest.columns:
                    X_latest[col] = 0
            X_latest = X_latest[self.models[0].feature_names_in_]

        proba = self.predict_proba_ensemble(X_latest)[0]
        result = {}
        for i, p in enumerate(proba):
            zodiac_code = self.label_encoder.inverse_transform([i])[0]
            result[zodiac_code] = p

        zodiac_seq = all_history_df['special_zodiac'].tolist()
        for z in range(1, 13):
            current_missing = temp_eng._calc_missing(zodiac_seq, z)
            max_missing = self.missing_max_dict.get(z, 100)
            if max_missing > 0 and current_missing >= 0.9 * max_missing:
                boost = 0.02 * (current_missing / max_missing)
                result[z] = min(result[z] + boost, 0.3)
        
        # 热号微调：近期出现频率高的生肖概率适度增加
        recent_20 = zodiac_seq[-20:] if len(zodiac_seq) >= 20 else zodiac_seq
        hot_counts = Counter(recent_20)
        max_hot = max(hot_counts.values()) if hot_counts else 1
        for z in range(1, 13):
            hot_ratio = hot_counts.get(z, 0) / max_hot
            result[z] = result[z] * (1 + 0.03 * hot_ratio)  # 热号提升3%
        
        # 归一化
        total = sum(result.values())
        for z in result:
            result[z] /= total
        return result


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("🔮 生肖预测 - 下一期预测（XGBoost版）")
    print("=" * 60)

    fetcher = LotteryDataFetcher()
    df = fetcher.fetch_history(start_year=2020, end_year=datetime.now().year)
    if df.empty:
        print("❌ 无数据，退出")
        return

    print(f"\n📋 共 {len(df)} 期数据")
    print(f"   最新一期: {df.iloc[-1]['period']} ({df.iloc[-1]['draw_time'].strftime('%Y-%m-%d')})")
    print(f"   特码: {df.iloc[-1]['special_number']} ({ZodiacRules.CODE_TO_NAME.get(df.iloc[-1]['special_zodiac'], '未知')})")

    predictor = ZodiacPredictor(n_models=3)
    X, y = predictor.prepare_data(df)
    print(f"\n🔧 特征维度: {X.shape}")

    split = int(len(X) * 0.8)
    X_train = X.iloc[:split]
    y_train = y[:split]

    print("\n🤖 训练集成模型...")
    predictor.train(X_train, y_train)

    print("\n🔮 预测下一期特别号生肖概率（经遗漏修正）：")
    next_proba = predictor.predict_next(df)
    sorted_proba = sorted(next_proba.items(), key=lambda x: x[1], reverse=True)

    next_year = datetime.now().year
    last_date = df.iloc[-1]['draw_time']
    # 每天开奖，下期就是最新一期+1天
    next_date = last_date + timedelta(days=1)

    print(f"\n   预计开奖: {next_date.strftime('%Y-%m-%d')} ({ZodiacRules.get_year_zodiac(next_year)}年)")
    print(f"   {'代码':<4} {'生肖':<4} {'概率':<8} {'对应号码(五行)'}")
    print("   " + "-" * 60)
    for code, prob in sorted_proba:
        name = ZodiacRules.CODE_TO_NAME[code]
        numbers = ZodiacRules.get_zodiac_numbers(code, next_year)
        num_str = ', '.join(f"{n}({ZodiacRules.number_to_wuxing(n)})" for n in sorted(numbers))
        bar = "█" * int(prob * 40)
        print(f"   {code:<4} {name:<4} {prob:.4f}   {num_str}")
        print(f"           {bar}")

    print("\n✅ 预测完成！")


if __name__ == "__main__":
    main()
