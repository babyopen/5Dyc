import json

# 加载数据
with open('彩票数据.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 展平数据
all_data = [item for sublist in data for item in sublist]

# 更新2026104期的特码
updated = False
for item in all_data:
    if item['period'] == '2026104':
        item['special_number'] = 1  # 更新特码为1
        item['special_zodiac'] = 1  # 马的生肖代码
        updated = True
        break

if updated:
    # 重新组织数据为原来的结构
    data = [all_data]
    
    # 保存数据
    with open('彩票数据.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print('✅ 成功更新2026104期数据')
    print('   期号: 2026104')
    print('   特码: 1')
    print('   生肖: 马')
else:
    print('❌ 未找到2026104期数据')
