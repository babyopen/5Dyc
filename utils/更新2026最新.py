import json

# 加载数据
with open('彩票数据.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 展平数据
all_data = [item for sublist in data for item in sublist]

# 需要更新的期号和对应的生肖代码
updates = {
    "2026104": 1,  # 马
    "2026103": 6,  # 牛
    "2026102": 8,  # 猪
    "2026101": 3,  # 龙
    "2026100": 9   # 狗
}

# 更新数据
updated = 0
for item in all_data:
    period = item['period']
    if period in updates:
        item['special_zodiac'] = updates[period]
        updated += 1

# 重新组织数据为原来的结构
data = [all_data]

# 保存数据
with open('彩票数据.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f'✅ 成功更新 {updated} 期数据')

# 验证更新结果
print('\n更新后的最新5期数据:')
for period in sorted(updates.keys(), reverse=True):
    for item in all_data:
        if item['period'] == period:
            print(f"期号: {period}, 生肖代码: {item['special_zodiac']}")
            break
