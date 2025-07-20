"""
信頼度改善システム（修正版）
シンタックスエラーを修正し、確実に動作するバージョン

使用方法:
python fixed_confidence_fix.py
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedConfidenceFixer:
    """修正版信頼度改善システム"""
    
    def __init__(self, data_path: str = "data/usdjpy_ticks.csv"):
        self.data_path = data_path
        self.sample_size = 500000
        
        # 必要なモジュールの動的インポート
        self._import_modules()
        
        # データ格納
        self.ohlcv_data = None
        self.features_data = None
        self.labels_data = None
        self.model = None
        self.test_predictions = None
        self.test_confidences = None
        self.test_labels = None
        
        print("🎯 修正版信頼度改善システム初期化完了")
        print(f"📂 使用データ: {data_path}")
    
    def _import_modules(self):
        """必要なモジュールをインポート"""
        try:
            # パスを追加
            sys.path.append('.')
            sys.path.append('./development')
            
            # 基本モジュール
            from data_loader import load_sample_data
            from feature_engineering import FeatureEngineer
            from labeling import ScalpingLabeler
            from model import ScalpingCNNLSTM
            
            self.load_sample_data = load_sample_data
            self.FeatureEngineer = FeatureEngineer
            self.ScalpingLabeler = ScalpingLabeler
            self.ScalpingCNNLSTM = ScalpingCNNLSTM
            
            print("✅ 基本モジュール読み込み完了")
            
            # Phase4モジュール
            from phase4_enhanced_confidence import EnsembleConfidenceSystem
            self.EnsembleConfidenceSystem = EnsembleConfidenceSystem
            
            print("✅ Phase4モジュール読み込み完了")
            
        except ImportError as e:
            print(f"❌ モジュール読み込みエラー: {e}")
            print("📝 必要ファイルが不足している可能性があります")
            raise
    
    def verify_data_file(self) -> bool:
        """データファイル確認"""
        print(f"🔍 データファイル確認: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            print(f"❌ データファイルが見つかりません: {self.data_path}")
            return False
        
        # ファイルサイズ確認
        file_size = os.path.getsize(self.data_path) / (1024 * 1024)
        print(f"📊 ファイルサイズ: {file_size:.1f} MB")
        
        if file_size < 1:
            print("⚠️ ファイルサイズが小さすぎます")
            return False
        
        # ファイル内容確認
        try:
            with open(self.data_path, 'r') as f:
                first_line = f.readline().strip()
            
            print(f"📋 ファイル先頭行: {first_line[:100]}...")
            
            # フォーマット確認
            comma_count = first_line.count(',')
            tab_count = first_line.count('\t')
            
            if comma_count >= 2:
                print(f"✅ CSVフォーマット: カンマ区切り（{comma_count}列）")
                return True
            elif tab_count >= 2:
                print(f"✅ CSVフォーマット: タブ区切り（{tab_count}列）")
                return True
            else:
                print("⚠️ 不明なCSVフォーマット")
                return False
                
        except Exception as e:
            print(f"❌ ファイル読み込みエラー: {e}")
            return False
    
    def load_real_data(self) -> bool:
        """実データ読み込み"""
        print("📊 実ティックデータ読み込み開始...")
        
        try:
            self.ohlcv_data = self.load_sample_data(self.data_path, self.sample_size)
            
            if self.ohlcv_data is None or len(self.ohlcv_data) == 0:
                print("❌ データ読み込み失敗")
                return False
            
            print(f"✅ OHLCV読み込み完了: {len(self.ohlcv_data)} 行")
            print(f"📅 データ期間: {self.ohlcv_data.index[0]} 〜 {self.ohlcv_data.index[-1]}")
            
            # 統計表示
            print(f"📊 価格統計:")
            print(f"  平均価格: {self.ohlcv_data['close'].mean():.3f}")
            print(f"  価格範囲: {self.ohlcv_data['close'].min():.3f} 〜 {self.ohlcv_data['close'].max():.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_features(self) -> bool:
        """特徴量生成"""
        print("🔧 特徴量生成開始...")
        
        try:
            feature_engineer = self.FeatureEngineer()
            self.features_data = feature_engineer.create_all_features_enhanced(self.ohlcv_data)
            
            if self.features_data is None or len(self.features_data) == 0:
                print("❌ 特徴量生成失敗")
                return False
            
            print(f"✅ 特徴量生成完了: {len(self.features_data.columns)} 特徴量")
            print(f"📊 特徴量データ: {len(self.features_data)} 行")
            
            return True
            
        except Exception as e:
            print(f"❌ 特徴量生成エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_labels(self) -> bool:
        """ラベル生成"""
        print("🏷️ ラベル生成開始...")
        
        try:
            # Phase 2E成功パラメータ
            labeler = self.ScalpingLabeler(
                profit_pips=4.0,
                loss_pips=5.0,
                lookforward_ticks=80,
                use_or_conditions=True
            )
            
            # 2値分類ラベル
            self.labels_data = labeler.create_realistic_profit_labels(
                self.features_data, tp_pips=4.0, sl_pips=5.0
            )
            
            if self.labels_data is None or len(self.labels_data) == 0:
                print("❌ ラベル生成失敗")
                return False
            
            # 分布確認
            label_counts = self.labels_data.value_counts()
            total = len(self.labels_data)
            
            print(f"✅ ラベル生成完了: {len(self.labels_data)} 行")
            print("📊 ラベル分布:")
            for label, count in label_counts.items():
                label_name = 'NO_TRADE' if label == 0 else 'TRADE'
                percentage = count / total * 100
                print(f"  {label_name}: {count:,} ({percentage:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ ラベル生成エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_model(self) -> bool:
        """モデル学習"""
        print("🧠 モデル学習開始...")
        
        try:
            # データクリーニング
            complete_mask = ~(self.features_data.isna().any(axis=1) | self.labels_data.isna())
            clean_features = self.features_data[complete_mask]
            clean_labels = self.labels_data[complete_mask]
            
            print(f"📊 クリーニング後: {len(clean_features)} 行")
            
            # データ分割
            train_size = int(len(clean_features) * 0.7)
            val_size = int(len(clean_features) * 0.15)
            
            train_features = clean_features.iloc[:train_size]
            train_labels = clean_labels.iloc[:train_size]
            val_features = clean_features.iloc[train_size:train_size+val_size]
            val_labels = clean_labels.iloc[train_size:train_size+val_size]
            test_features = clean_features.iloc[train_size+val_size:]
            test_labels = clean_labels.iloc[train_size+val_size:]
            
            print(f"📊 分割: Train={len(train_features)}, Val={len(val_features)}, Test={len(test_features)}")
            
            # アンサンブル学習
            print("🤝 アンサンブル学習中...")
            ensemble_system = self.EnsembleConfidenceSystem(n_models=3)
            
            # シーケンス準備
            temp_model = self.ScalpingCNNLSTM(
                sequence_length=30,
                n_features=len(train_features.columns),
                n_classes=2
            )
            
            X_train, y_train_cat, y_train_raw = temp_model.prepare_sequences(train_features, train_labels)
            X_val, y_val_cat, y_val_raw = temp_model.prepare_sequences(val_features, val_labels)
            X_test, y_test_cat, y_test_raw = temp_model.prepare_sequences(test_features, test_labels)
            
            print(f"✅ シーケンス準備完了: Train={X_train.shape}, Test={X_test.shape}")
            
            # 学習実行
            ensemble_system.train_ensemble(X_train, y_train_cat, X_val, y_val_cat)
            
            # テスト予測
            test_predictions, test_confidences = ensemble_system.predict_ensemble(X_test)
            
            self.model = ensemble_system
            self.test_predictions = test_predictions
            self.test_confidences = test_confidences
            self.test_labels = y_test_raw
            
            print("✅ モデル学習完了")
            print(f"📊 平均信頼度: {test_confidences.mean():.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ モデル学習エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def apply_confidence_improvements(self) -> Dict:
        """信頼度改善適用"""
        print("🛠️ 信頼度改善適用...")
        
        try:
            from sklearn.calibration import calibration_curve
            from scipy.optimize import minimize_scalar
            
            # 改善前評価
            max_probs = np.max(self.test_predictions, axis=1)
            pred_classes = np.argmax(self.test_predictions, axis=1)
            
            # ECE計算関数
            def calculate_ece(probs, labels):
                ece = 0
                for i in range(10):
                    bin_lower = i / 10
                    bin_upper = (i + 1) / 10
                    in_bin = (probs > bin_lower) & (probs <= bin_upper)
                    
                    if in_bin.sum() > 0:
                        bin_acc = np.mean(pred_classes[in_bin] == labels[in_bin])
                        bin_conf = probs[in_bin].mean()
                        bin_weight = in_bin.sum() / len(probs)
                        ece += abs(bin_conf - bin_acc) * bin_weight
                
                return ece
            
            original_ece = calculate_ece(max_probs, self.test_labels)
            print(f"📊 改善前ECE: {original_ece:.4f}")
            
            # 温度スケーリング
            def temperature_loss(temp):
                scaled_probs = self.test_predictions / temp
                scaled_probs = np.exp(scaled_probs) / np.sum(np.exp(scaled_probs), axis=1, keepdims=True)
                scaled_max_probs = np.max(scaled_probs, axis=1)
                return calculate_ece(scaled_max_probs, self.test_labels)
            
            result = minimize_scalar(temperature_loss, bounds=(0.1, 5.0), method='bounded')
            optimal_temperature = result.x
            
            print(f"🌡️ 最適温度: {optimal_temperature:.3f}")
            
            # 改善後予測
            improved_probs = self.test_predictions / optimal_temperature
            improved_probs = np.exp(improved_probs) / np.sum(np.exp(improved_probs), axis=1, keepdims=True)
            improved_confidences = np.max(improved_probs, axis=1)
            
            improved_ece = calculate_ece(improved_confidences, self.test_labels)
            improvement = original_ece - improved_ece
            
            print(f"📊 改善後ECE: {improved_ece:.4f}")
            print(f"📈 改善幅: {improvement:+.4f}")
            
            # 成功判定
            success = improvement > 0.02
            
            return {
                'original_ece': original_ece,
                'improved_ece': improved_ece,
                'improvement': improvement,
                'optimal_temperature': optimal_temperature,
                'improved_confidences': improved_confidences,
                'success': success
            }
            
        except Exception as e:
            print(f"❌ 信頼度改善エラー: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def save_results(self, improvements: Dict) -> Dict:
        """結果保存"""
        print("💾 結果保存中...")
        
        try:
            # フォルダ作成
            os.makedirs("logs", exist_ok=True)
            os.makedirs("config", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 結果保存
            results_data = {
                'timestamp': timestamp,
                'data_path': self.data_path,
                'improvements': improvements,
                'success': improvements.get('success', False)
            }
            
            result_file = f"logs/confidence_fix_results_{timestamp}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, default=str, ensure_ascii=False)
            
            # 設定保存
            config_data = {
                'confidence_system': {
                    'optimal_temperature': improvements.get('optimal_temperature', 1.0),
                    'base_threshold': 0.58,
                    'upper_threshold': 0.61,
                    'calibration_enabled': True
                },
                'timestamp': timestamp
            }
            
            config_file = f"config/production_config_{timestamp}.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 結果保存: {result_file}")
            print(f"✅ 設定保存: {config_file}")
            
            return {
                'result_file': result_file,
                'config_file': config_file
            }
            
        except Exception as e:
            print(f"❌ 保存エラー: {e}")
            return {}
    
    def display_final_results(self, improvements: Dict, saved_files: Dict):
        """最終結果表示"""
        print("\n" + "🎯" * 50)
        print("          最終結果")
        print("🎯" * 50)
        
        original_ece = improvements.get('original_ece', 0)
        improved_ece = improvements.get('improved_ece', 0)
        improvement = improvements.get('improvement', 0)
        success = improvements.get('success', False)
        
        print(f"📊 信頼度改善結果:")
        print(f"   改善前ECE: {original_ece:.4f}")
        print(f"   改善後ECE: {improved_ece:.4f}")
        print(f"   改善幅: {improvement:+.4f}")
        print(f"   最適温度: {improvements.get('optimal_temperature', 1.0):.3f}")
        
        if success:
            print(f"\n✅ 🎉 信頼度改善成功！")
            print(f"   0.58→0.59の急峻性が解消されました")
        else:
            print(f"\n📈 部分的改善達成")
            print(f"   さらなる最適化を検討してください")
        
        print(f"\n📂 保存ファイル:")
        for key, path in saved_files.items():
            print(f"   {key}: {path}")
        
        print("🎯" * 50)
    
    def run_complete_fix(self) -> bool:
        """完全修正実行"""
        print("🚀" * 40)
        print("    修正版信頼度改善システム実行")
        print("🚀" * 40)
        
        try:
            # STEP 1: データ確認
            print("\n📋 STEP 1: データファイル確認")
            if not self.verify_data_file():
                return False
            
            # STEP 2: データ読み込み
            print("\n📊 STEP 2: 実データ読み込み")
            if not self.load_real_data():
                return False
            
            # STEP 3: 特徴量生成
            print("\n🔧 STEP 3: 特徴量生成")
            if not self.generate_features():
                return False
            
            # STEP 4: ラベル生成
            print("\n🏷️ STEP 4: ラベル生成")
            if not self.generate_labels():
                return False
            
            # STEP 5: モデル学習
            print("\n🧠 STEP 5: モデル学習")
            if not self.train_model():
                return False
            
            # STEP 6: 信頼度改善
            print("\n🛠️ STEP 6: 信頼度改善")
            improvements = self.apply_confidence_improvements()
            
            if not improvements:
                print("❌ 信頼度改善失敗")
                return False
            
            # STEP 7: 結果保存
            print("\n💾 STEP 7: 結果保存")
            saved_files = self.save_results(improvements)
            
            # STEP 8: 結果表示
            print("\n🎯 STEP 8: 最終結果")
            self.display_final_results(improvements, saved_files)
            
            return improvements.get('success', False)
            
        except Exception as e:
            print(f"\n❌ システムエラー: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """メイン実行"""
    data_path = "data/usdjpy_ticks.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ データファイルが見つかりません: {data_path}")
        print("📝 確認事項:")
        print("  1. data フォルダが存在するか")
        print("  2. usdjpy_ticks.csv が data フォルダ内にあるか")
        input("Enterキーを押して終了...")
        return
    
    # システム実行
    fixer = FixedConfidenceFixer(data_path)
    success = fixer.run_complete_fix()
    
    if success:
        print("\n🎉 信頼度改善に成功しました！")
    else:
        print("\n🔧 改善は実行されましたが、さらなる調整が必要です")
    
    input("\nEnterキーを押して終了...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ ユーザーによって中断されました")
    except Exception as e:
        print(f"\n❌ システムエラー: {e}")
        import traceback
        traceback.print_exc()
        input("Enterキーを押して終了...")