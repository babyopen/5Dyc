from xboyi import LotteryDataFetcher, ZodiacRules

# 获取历史数据
fetcher = LotteryDataFetcher()
df = fetcher.fetch_history()

# 显示最近10期
recent = df.tail(10)
print('最近10期开奖数据：')
print('期号         日期          特别号  特别生肖')
print('-' * 40)
for _, row in recent.iterrows():
    print(f"{row['period']}  {row['draw_time']}  {row['special_number']}    {ZodiacRules.CODE_TO_NAME[row['special_zodiac']]}")
