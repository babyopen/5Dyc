"""
模型管理工具
功能：
1. 保存训练好的模型
2. 加载已保存的模型
3. 模型性能评估
4. 模型版本管理
"""

import pickle
import os
import json
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import numpy as np

# 导入预测器
import sys
sys.path.append('..')
from scripts.predict_next import ZodiacPredictor, LotteryDataFetcher


class ModelManager:
    """模型管理器"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.performance_log = os.path.join(models_dir, 'performance_log.json')
    
    def save_model(self, predictor: ZodiacPredictor, 
                   train_data_info: Dict, 
                   model_name: str = None) -> str:
        """
        保存模型
        
        Args:
            predictor: 训练好的预测器
            train_data_info: 训练数据信息
            model_name: 模型名称（可选）
        
        Returns:
            保存的文件路径
        """
        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"zodiac_model_{timestamp}.pkl"
        
        # 确保文件名以 .pkl 结尾
        if not model_name.endswith('.pkl'):
            model_name += '.pkl'
        
        filepath = os.path.join(self.models_dir, model_name)
        
        # 准备保存的数据
        model_data = {
            'models': predictor.models,
            'calibrators': predictor.calibrators,
            'label_encoder': predictor.label_encoder,
            'missing_max_dict': predictor.missing_max_dict,
            'feature_engineer': predictor.feature_engineer,
            'n_models': predictor.n_models,
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'train_data_info': train_data_info,
                'version': '1.0'
            }
        }
        
        # 保存模型
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"✅ 模型已保存: {filepath}")
            print(f"   文件大小: {file_size:.2f} MB")
            
            # 记录到性能日志
            self._log_performance(model_name, train_data_info, file_size)
            
            return filepath
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return None
    
    def load_model(self, model_path: str) -> Optional[ZodiacPredictor]:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
        
        Returns:
            加载的预测器，失败返回 None
        """
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 重建预测器
            predictor = ZodiacPredictor(n_models=model_data['n_models'])
            predictor.models = model_data['models']
            predictor.calibrators = model_data['calibrators']
            predictor.label_encoder = model_data['label_encoder']
            predictor.missing_max_dict = model_data['missing_max_dict']
            predictor.feature_engineer = model_data['feature_engineer']
            
            metadata = model_data.get('metadata', {})
            saved_at = metadata.get('saved_at', '未知')
            
            print(f"✅ 模型已加载: {model_path}")
            print(f"   保存时间: {saved_at}")
            print(f"   模型数量: {len(predictor.models)}")
            
            return predictor
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def list_models(self) -> list:
        """列出所有可用的模型"""
        if not os.path.exists(self.models_dir):
            print("📂 models 目录不存在")
            return []
        
        models = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        
        if not models:
            print("📂 models 目录中没有模型文件")
            return []
        
        print(f"\n📦 可用的模型 ({len(models)} 个):")
        print("-" * 80)
        
        model_info = []
        for model_file in sorted(models):
            filepath = os.path.join(self.models_dir, model_file)
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            print(f"  • {model_file:<40} {file_size:6.2f} MB  {mod_time.strftime('%Y-%m-%d %H:%M')}")
            model_info.append({
                'name': model_file,
                'size_mb': file_size,
                'modified': mod_time.isoformat()
            })
        
        print("-" * 80)
        return model_info
    
    def delete_model(self, model_name: str) -> bool:
        """删除指定的模型"""
        filepath = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(filepath):
            print(f"❌ 模型不存在: {model_name}")
            return False
        
        try:
            os.remove(filepath)
            print(f"✅ 已删除: {model_name}")
            return True
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False
    
    def _log_performance(self, model_name: str, 
                        train_info: Dict, 
                        file_size: float):
        """记录模型性能日志"""
        log_entry = {
            'model_name': model_name,
            'saved_at': datetime.now().isoformat(),
            'file_size_mb': round(file_size, 2),
            'train_info': train_info
        }
        
        # 读取现有日志
        logs = []
        if os.path.exists(self.performance_log):
            try:
                with open(self.performance_log, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # 添加新记录
        logs.append(log_entry)
        
        # 保存日志
        try:
            with open(self.performance_log, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 性能日志保存失败: {e}")


def train_and_save_model(use_full_data=True, model_name=None):
    """训练并保存模型"""
    
    print("=" * 80)
    print("🚀 开始训练并保存模型")
    print("=" * 80)
    
    # 获取数据
    fetcher = LotteryDataFetcher()
    
    if use_full_data:
        print("\n📡 获取完整历史数据...")
        df = fetcher.fetch_history(start_year=2020, end_year=2026)
        data_info = {'years': '2020-2026', 'type': 'full'}
    else:
        print("\n📡 获取2026年数据...")
        # 这里需要从本地加载2026年数据
        import json
        with open('../data/彩票数据.json', 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        all_records = []
        for item in raw_data:
            if isinstance(item, list):
                all_records.extend(item)
        
        df_2026 = [r for r in all_records if r['period'].startswith('2026')]
        df = pd.DataFrame(df_2026)
        df['draw_time'] = pd.to_datetime(df['draw_time'])
        df = df.sort_values('draw_time').reset_index(drop=True)
        data_info = {'years': '2026', 'type': '2026_only'}
    
    if df.empty:
        print("❌ 无数据")
        return
    
    print(f"\n📊 数据量: {len(df)} 期")
    
    # 训练模型
    predictor = ZodiacPredictor(n_models=3)
    X, y = predictor.prepare_data(df)
    
    split = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split], y[:split]
    
    print(f"\n🤖 训练模型...")
    predictor.train(X_train, y_train)
    
    # 添加训练信息
    data_info.update({
        'total_periods': len(df),
        'train_samples': len(X_train),
        'feature_dim': X.shape[1],
        'trained_at': datetime.now().isoformat()
    })
    
    # 保存模型
    manager = ModelManager()
    filepath = manager.save_model(predictor, data_info, model_name)
    
    if filepath:
        print(f"\n✅ 完成！模型已保存到: {filepath}")
    else:
        print(f"\n❌ 保存失败")


def main():
    """主函数 - 命令行界面"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模型管理工具')
    parser.add_argument('action', choices=['train', 'list', 'load', 'delete'],
                       help='操作类型')
    parser.add_argument('--model', type=str, help='模型文件名')
    parser.add_argument('--full-data', action='store_true', 
                       help='使用完整历史数据（默认仅2026年）')
    parser.add_argument('--name', type=str, help='自定义模型名称')
    
    args = parser.parse_args()
    
    manager = ModelManager()
    
    if args.action == 'train':
        train_and_save_model(
            use_full_data=args.full_data,
            model_name=args.name
        )
    
    elif args.action == 'list':
        manager.list_models()
    
    elif args.action == 'load':
        if not args.model:
            print("❌ 请指定模型文件: --model <filename>")
            return
        
        model_path = os.path.join('models', args.model)
        predictor = manager.load_model(model_path)
        
        if predictor:
            print("\n💡 提示: 可以使用 predictor.predict_next(df) 进行预测")
    
    elif args.action == 'delete':
        if not args.model:
            print("❌ 请指定要删除的模型: --model <filename>")
            return
        
        confirm = input(f"确认删除 {args.model}? (y/n): ")
        if confirm.lower() == 'y':
            manager.delete_model(args.model)


if __name__ == "__main__":
    main()

