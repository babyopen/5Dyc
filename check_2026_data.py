import json

# 加载数据
with open('lottery_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 展平数据
all_data = [item for sublist in data for item in sublist]

# 筛选2026年数据
year_2026 = [d for d in all_data if d['period'].startswith('2026')]

# 统计数量
print(f'2026年数据数量: {len(year_2026)}')

if year_2026:
    # 按期号排序
    sorted_periods = sorted([d['period'] for d in year_2026])
    print(f'最新期号: {sorted_periods[-1]}')
    print(f'最早期号: {sorted_periods[0]}')

    # 检查是否有缺失期号
    expected_periods = [f'2026{i:03d}' for i in range(1, 105)]
    actual_periods = sorted([d['period'] for d in year_2026])
    
    missing = [p for p in expected_periods if p not in actual_periods]
    if missing:
        print(f'缺失期号: {missing}')
    else:
        print('✅ 2026年数据完整，无缺失期号')
