"""
基于2026年数据的生肖预测
只使用2026年的数据进行训练和预测
输出格式：清晰的表格形式
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
from 预测下一期 import ZodiacRules, FeatureEngineer, ZodiacPredictor


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
    print(f" 生肖预测 - {target_year}年数据专用版")
    print("=" * 60)
    
    # 1. 加载本地数据
    print(f"\n【步骤1】加载{target_year}年数据...")
    df = LocalDataLoader.load_from_json('data/彩票数据.json', year=target_year)
    
    if df.empty:
        print("❌ 无数据，退出")
        return
    
    # 2. 准备特征
    print(f"\n【步骤2】准备特征工程...")
    predictor = ZodiacPredictor(n_models=3)
    X, y = predictor.prepare_data(df)
    print(f"   特征维度: {X.shape}")
    print(f"   样本数量: {len(X)}")
    
    # 检查样本数量是否足够
    if len(X) < 50:
        print(f"\n⚠️ 警告: 样本数量较少 ({len(X)}个)，可能影响模型效果")
        print(f"   建议: 至少需要50个以上样本才能获得较好的预测效果")
    
    # 3. 划分训练集（使用前80%的数据）
    split = int(len(X) * 0.8)
    X_train = X.iloc[:split]
    y_train = y[:split]
    
    print(f"\n📊 训练集/测试集划分:")
    print(f"   训练集: {len(X_train)} 样本")
    print(f"   测试集: {len(X) - len(X_train)} 样本")
    
    # 4. 训练模型
    print(f"\n【步骤3】训练集成模型...")
    predictor.train(X_train, y_train)
    
    # 5. 预测下一期
    print(f"\n【步骤4】预测下一期特别号生肖概率...")
    next_proba = predictor.predict_next(df)
    sorted_proba = sorted(next_proba.items(), key=lambda x: x[1], reverse=True)
    
    # 6. 显示预测结果 - 新格式
    next_period = int(df.iloc[-1]['period']) + 1
    last_date = df.iloc[-1]['draw_time']
    next_date = last_date + timedelta(days=1)
    next_year = target_year
    
    print(f"\n 下一期（{next_period}期）推荐结果")
    print()
    print("📊 数据更新")
    print()
    print(f"   • 最新期数: {df.iloc[-1]['period']}期 ({df.iloc[-1]['draw_time'].strftime('%Y-%m-%d')})")
    print(f"   • 特码: {df.iloc[-1]['special_number']}")
    print(f"   • 生肖: {ZodiacRules.CODE_TO_NAME.get(df.iloc[-1]['special_zodiac'], '未知')} (代码{df.iloc[-1]['special_zodiac']}) ✅")
    print(f"   • 总数据量: {len(df)}期")
    
    # 7. Top 5 推荐生肖 - 表格格式
    print(f"\n🏆 Top 5 推荐生肖 ({next_period}期 - {next_date.strftime('%Y-%m-%d')})")
    print()
    print(f"   {'排名':<8} {'生肖':<8} {'概率':<12} {'推荐号码'}")
    print(f"   {'─'*70}")
    
    for rank, (code, prob) in enumerate(sorted_proba[:5], 1):
        name = ZodiacRules.CODE_TO_NAME[code]
        numbers = ZodiacRules.get_zodiac_numbers(code, next_year)
        num_str = ', '.join(str(n) for n in sorted(numbers))
        
        # 排名图标
        if rank == 1:
            rank_icon = "👑 1"
        elif rank == 2:
            rank_icon = "👑 2"
        elif rank == 3:
            rank_icon = "👑 3"
        else:
            rank_icon = f"  {rank}"
        
        print(f"   {rank_icon:<8} {name:<8} {prob:.2%}    {num_str}")
    
    # 8. 关键分析
    print(f"\n💡 关键分析")
    print()
    zodiac_counts = Counter(df['special_zodiac'])
    total = len(df)
    
    for rank, (code, prob) in enumerate(sorted_proba[:3], 1):
        name = ZodiacRules.CODE_TO_NAME[code]
        count = zodiac_counts.get(code, 0)
        freq = count / total * 100
        
        # 分析描述
        if freq < 5:
            level = "较低水平"
        elif freq < 10:
            level = "中等水平"
        else:
            level = "较高水平"
        
        title = "位居榜首" if rank == 1 else "排名靠前"
        print(f"   {rank}. {name} (代码{code}) 以 {prob:.2%} 的概率{title}！")
        print(f"      ○ {name}在{target_year}年出现{count}次 ({freq:.1f}%), 处于{level}")
        if rank == 1:
            print(f"      ○ 模型预测其即将爆发")
        print()
    
    # 9. 核心推荐 - 策略建议
    print(f"\n💡 核心推荐")
    print(f"   重点关注：")
    
    # 首选组合（前2名）
    top2_sum = sum(prob for _, prob in sorted_proba[:2])
    top2_names = [ZodiacRules.CODE_TO_NAME[code] for code, _ in sorted_proba[:2]]
    print(f"      首选组合： {' + '.join(top2_names)}（合计概率 {top2_sum:.2%}）")
    
    # 重点号码（前2名的所有号码）
    focus_numbers = []
    for code, _ in sorted_proba[:2]:
        numbers = ZodiacRules.get_zodiac_numbers(code, next_year)
        focus_numbers.extend(sorted(numbers))
    print(f"      重点号码： {', '.join(map(str, focus_numbers))}")
    
    print(f"\n   策略建议：")
    
    # 高概率区（前2名）
    high_prob = sorted_proba[:2]
    high_prob_str = '、'.join([f"{ZodiacRules.CODE_TO_NAME[code]}（{prob:.2%}）" for code, prob in high_prob])
    print(f"      高概率区： {high_prob_str}")
    
    # 中概率区（第3-4名）
    mid_prob = sorted_proba[2:4]
    mid_prob_str = '、'.join([f"{ZodiacRules.CODE_TO_NAME[code]}（{prob:.2%}）" for code, prob in mid_prob])
    print(f"      中概率区： {mid_prob_str}")
    
    # 防守区（第5-6名）
    low_prob = sorted_proba[4:6]
    low_prob_str = '、'.join([f"{ZodiacRules.CODE_TO_NAME[code]}（{prob:.2%}）" for code, prob in low_prob])
    print(f"      防守区： {low_prob_str}")
    
    # 10. 预测说明
    print(f"\n{'─'*60}")
    print(f"✅ 预测完成！")
    print(f"{'─'*60}")
    print(f"\n 预测说明:")
    print(f"   • 仅基于{target_year}年共 {len(df)} 期历史数据训练")
    print(f"   • 使用XGBoost集成学习模型（3个子模型）")
    print(f"   • 考虑遗漏值、热号、五行相生相克等因素")
    print(f"   • ⚠️ 注意: 仅使用单年数据，样本量有限，预测仅供参考")
    print(f"   • 概率仅供参考，请理性投注")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
