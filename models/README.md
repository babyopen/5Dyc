# 模型文件说明

## 概述

`models/` 目录用于存储训练好的机器学习模型文件，以便快速加载和预测，避免每次都要重新训练。

## 当前状态

⚠️ **目前此目录为空**

原因：
- 当前系统设计为每次运行时实时训练模型
- 这样可以确保使用最新的数据
- 模型文件大小较大，未默认包含在仓库中

## 如何保存模型

### 方法一：手动保存

在运行预测脚本后，可以手动保存模型：

```python
import pickle
from scripts.predict_next import ZodiacPredictor

# 训练模型
predictor = ZodiacPredictor(n_models=3)
X, y = predictor.prepare_data(df)
predictor.train(X_train, y_train)

# 保存模型
with open('models/zodiac_predictor_v1.pkl', 'wb') as f:
    pickle.dump({
        'models': predictor.models,
        'calibrators': predictor.calibrators,
        'label_encoder': predictor.label_encoder,
        'missing_max_dict': predictor.missing_max_dict,
        'feature_engineer': predictor.feature_engineer
    }, f)

print("✅ 模型已保存到 models/zodiac_predictor_v1.pkl")
```

### 方法二：使用模型管理脚本

创建 `utils/save_model.py`：

```bash
cd utils
python3 save_model.py --output models/model_20260417.pkl
```

## 如何加载模型

```python
import pickle

# 加载模型
with open('models/zodiac_predictor_v1.pkl', 'rb') as f:
    model_data = pickle.load(f)

# 使用模型进行预测
predictor = ZodiacPredictor()
predictor.models = model_data['models']
predictor.label_encoder = model_data['label_encoder']
predictor.missing_max_dict = model_data['missing_max_dict']
predictor.feature_engineer = model_data['feature_engineer']

# 预测下一期
next_proba = predictor.predict_next(df)
```

## 模型文件命名规范

建议的命名格式：

```
{描述}_{日期}_{版本}.pkl

示例：
- zodiac_full_20260417_v1.pkl      # 完整数据训练的模型
- zodiac_2026only_20260417_v1.pkl  # 仅2026年数据
- zodiac_best_20260417.pkl         # 表现最好的模型
```

## 模型文件大小

典型大小：
- 单个 XGBoost 模型：~5-10 MB
- 3个模型的集成：~15-30 MB
- 包含所有元数据：~20-40 MB

## .gitignore 配置

由于模型文件较大且可以从数据重新训练，建议在 `.gitignore` 中添加：

```gitignore
# 模型文件
models/*.pkl
models/*.joblib
models/*.model
```

如果需要上传特定模型到 Git，可以使用：
```bash
git add -f models/specific_model.pkl
```

## 模型版本管理

### 版本记录表

| 版本 | 日期 | 训练数据 | 准确率 | 备注 |
|------|------|----------|--------|------|
| v1.0 | 2026-04-17 | 2020-2026 | ~18% | 初始版本 |
| ... | ... | ... | ... | ... |

### 保留策略

建议保留：
1. **最新版本** - 始终保留
2. **最佳版本** - 历史表现最好的
3. **基准版本** - 用于对比的基线模型

定期清理旧模型以节省空间。

## 模型性能监控

建议记录每次模型的：
- 训练时间
- Top-1/3/5 准确率
- 对数损失 (Log Loss)
- 特征重要性排名

可以创建 `models/performance_log.csv` 来追踪。

## 注意事项

⚠️ **重要提醒：**

1. **兼容性问题**
   - 模型与训练时的库版本相关
   - 升级 XGBoost 等库后可能需要重新训练
   - 建议记录训练时的环境信息

2. **数据一致性**
   - 加载模型时必须使用相同的特征工程
   - LabelEncoder 必须一致
   - 最好保存完整的 predictor 对象

3. **安全性**
   - 不要加载不可信的 pickle 文件
   - pickle 可能执行恶意代码
   - 考虑使用 joblib 或 ONNX 格式

4. **存储空间**
   - 模型文件会占用较多空间
   - 定期清理不需要的模型
   - 考虑压缩存储

## 替代方案

### 方案一：不保存模型（当前方式）
**优点：**
- 始终使用最新数据
- 无需管理模型文件
- 避免版本兼容问题

**缺点：**
- 每次预测需要重新训练
- 耗时较长（30-60秒）

### 方案二：保存模型
**优点：**
- 预测速度快（<1秒）
- 可以离线使用
- 便于模型对比

**缺点：**
- 需要管理模型文件
- 可能过时
- 占用存储空间

### 方案三：混合方式（推荐）
- 每天训练一次新模型并保存
- 当天多次预测使用缓存的模型
- 定期评估模型性能，必要时重新训练

## 未来改进

计划添加的功能：
1. ✅ 自动模型保存/加载
2. ✅ 模型性能追踪
3. ✅ 模型自动选择（选择表现最好的）
4. ✅ 模型压缩和优化
5. ✅ 在线模型更新

---

*最后更新：2026-04-17*
