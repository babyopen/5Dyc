"""
基于2026年数据的生肖预测
只使用2026年的数据进行训练和预测
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import Counter

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# 导入现有的类
from predict_next import ZodiacRules, FeatureEngineer, ZodiacPredictor


class LocalDataLoader:
    """从本地JSON文件加载数据"""
    
    @staticmethod
    def load_from_json(data_file='lottery_data.json', year=2026) -> pd.DataFrame:
        """从JSON文件加载指定年份的数据"""
        if not os.path.exists(data_file):
            print(f"❌ 文件 {data_file} 不存在")
            return pd.DataFrame()
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # 展平数据
            all_records = []
            for item in raw_data:
                if isinstance(item, list):
                    all_records.extend(item)
                elif isinstance(item, dict):
                    all_records.append(item)
            
            if not all_records:
                print("❌ 数据为空")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_records)
            df['draw_time'] = pd.to_datetime(df['draw_time'])
            
            # 过滤指定年份
            df['year'] = df['draw_time'].dt.year
            df_filtered = df[df['year'] == year].copy()
            
            if df_filtered.empty:
                print(f"❌ 没有找到 {year} 年的数据")
                return pd.DataFrame()
            
            df_filtered = df_filtered.sort_values('draw_time').reset_index(drop=True)
            
            # 添加序号
            df_filtered['seq_period'] = range(1, len(df_filtered) + 1)
            
            print(f"✅ 成功加载 {year} 年数据: {len(df_filtered)} 期")
            return df_filtered
            
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return pd.DataFrame()


def main():
    target_year = 2026
    
    print("=" * 60)
    print(f"🔮 生肖预测 - {target_year}年数据专用版")
    print("=" * 60)
    
    # 1. 加载本地数据
    print(f"\n【步骤1】加载{target_year}年数据...")
    df = LocalDataLoader.load_from_json('lottery_data.json', year=target_year)
    
    if df.empty:
        print("❌ 无数据，退出")
        return
    
    # 2. 显示数据统计
    print(f"\n📋 数据统计:")
    print(f"   总期数: {len(df)}")
    print(f"   时间范围: {df.iloc[0]['draw_time'].strftime('%Y-%m-%d')} 到 {df.iloc[-1]['draw_time'].strftime('%Y-%m-%d')}")
    print(f"   期数范围: {df.iloc[0]['period']} 到 {df.iloc[-1]['period']}")
    print(f"   最新一期: {df.iloc[-1]['period']} ({df.iloc[-1]['draw_time'].strftime('%Y-%m-%d')})")
    print(f"   特码: {df.iloc[-1]['special_number']} ({ZodiacRules.CODE_TO_NAME.get(df.iloc[-1]['special_zodiac'], '未知')})")
    
    # 显示前5期和最后5期
    print(f"\n📊 前5期数据:")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        print(f"   {row['period']} | {row['draw_time'].strftime('%Y-%m-%d')} | 特码: {row['special_number']} | 生肖: {ZodiacRules.CODE_TO_NAME.get(row['special_zodiac'], '未知')}")
    
    print(f"\n📊 最后5期数据:")
    for i in range(max(0, len(df)-5), len(df)):
        row = df.iloc[i]
        print(f"   {row['period']} | {row['draw_time'].strftime('%Y-%m-%d')} | 特码: {row['special_number']} | 生肖: {ZodiacRules.CODE_TO_NAME.get(row['special_zodiac'], '未知')}")
    
    # 3. 准备特征
    print(f"\n【步骤2】准备特征工程...")
    predictor = ZodiacPredictor(n_models=3)
    X, y = predictor.prepare_data(df)
    print(f"   特征维度: {X.shape}")
    print(f"   样本数量: {len(X)}")
    
    # 检查样本数量是否足够
    if len(X) < 50:
        print(f"\n⚠️ 警告: 样本数量较少 ({len(X)}个)，可能影响模型效果")
        print(f"   建议: 至少需要50个以上样本才能获得较好的预测效果")
    
    # 4. 划分训练集（使用前80%的数据）
    split = int(len(X) * 0.8)
    X_train = X.iloc[:split]
    y_train = y[:split]
    
    print(f"\n📊 训练集/测试集划分:")
    print(f"   训练集: {len(X_train)} 样本")
    print(f"   测试集: {len(X) - len(X_train)} 样本")
    
    # 5. 训练模型
    print(f"\n【步骤3】训练集成模型...")
    predictor.train(X_train, y_train)
    
    # 6. 预测下一期
    print(f"\n【步骤4】预测下一期特别号生肖概率...")
    next_proba = predictor.predict_next(df)
    sorted_proba = sorted(next_proba.items(), key=lambda x: x[1], reverse=True)
    
    # 7. 显示预测结果
    next_year = target_year
    last_date = df.iloc[-1]['draw_time']
    next_date = last_date + timedelta(days=1)
    
    print(f"\n{'='*60}")
    print(f"🎯 预测结果 - 下一期（预计 {next_date.strftime('%Y-%m-%d')}）")
    print(f"{'='*60}")
    print(f"   当前年份: {ZodiacRules.get_year_zodiac(next_year)}年")
    print(f"\n   {'排名':<4} {'代码':<4} {'生肖':<4} {'概率':<8} {'推荐指数':<10} {'对应号码(五行)'}")
    print(f"   {'-'*80}")
    
    for rank, (code, prob) in enumerate(sorted_proba, 1):
        name = ZodiacRules.CODE_TO_NAME[code]
        numbers = ZodiacRules.get_zodiac_numbers(code, next_year)
        num_str = ', '.join(f"{n}({ZodiacRules.number_to_wuxing(n)})" for n in sorted(numbers))
        
        # 计算推荐指数（星级）
        stars = "⭐" * max(1, int(prob * 20))
        
        # 高亮前3名
        if rank <= 3:
            print(f"   👑{rank:<3} {code:<4} {name:<4} {prob:.4f}   {stars:<10} {num_str}")
        else:
            print(f"   {rank:<4} {code:<4} {name:<4} {prob:.4f}   {stars:<10} {num_str}")
    
    # 8. 显示Top 5推荐
    print(f"\n{'='*60}")
    print(f"🏆 Top 5 推荐生肖")
    print(f"{'='*60}")
    for rank, (code, prob) in enumerate(sorted_proba[:5], 1):
        name = ZodiacRules.CODE_TO_NAME[code]
        numbers = ZodiacRules.get_zodiac_numbers(code, next_year)
        num_str = ', '.join(str(n) for n in sorted(numbers))
        print(f"   {rank}. {name} (代码{code}) - 概率: {prob:.2%}")
        print(f"      推荐号码: {num_str}")
    
    # 9. 生肖出现频率统计
    print(f"\n{'='*60}")
    print(f"📈 {target_year}年生肖出现频率统计")
    print(f"{'='*60}")
    zodiac_counts = Counter(df['special_zodiac'])
    total = len(df)
    for code in range(1, 13):
        count = zodiac_counts.get(code, 0)
        freq = count / total * 100
        bar = "█" * int(freq / 2)
        name = ZodiacRules.CODE_TO_NAME[code]
        print(f"   {name:<4} ({code:2d}): {count:3d}次 ({freq:5.1f}%) {bar}")
    
    print(f"\n{'='*60}")
    print(f"💡 预测说明:")
    print(f"   • 仅基于{target_year}年共 {len(df)} 期历史数据训练")
    print(f"   • 使用XGBoost集成学习模型（3个子模型）")
    print(f"   • 考虑遗漏值、热号、五行相生相克等因素")
    print(f"   • ⚠️ 注意: 仅使用单年数据，样本量有限，预测仅供参考")
    print(f"   • 概率仅供参考，请理性投注")
    print(f"{'='*60}")
    
    print("\n✅ 预测完成！")


if __name__ == "__main__":
    main()
