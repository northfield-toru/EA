"""
月次モデル更新スクリプト（完全自動化版）

使用方法:
python monthly_model_update.py
python monthly_model_update.py --new_data data/usdjpy_ticks_new.csv
python monthly_model_update.py --new_data data/usdjpy_ticks_new.csv --auto_deploy
"""

import os
import sys
import json
import shutil
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple

# Windows文字エンコーディング修正
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# ログ設定（絵文字なし、Windows対応）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monthly_update.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

# 安全な出力関数
def safe_print(message):
    """Windows環境でも安全に絵文字を表示"""
    if sys.platform == "win32":
        emoji_replacements = {
            '🚀': '[START]', '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARN]',
            '📊': '[DATA]', '📦': '[BACKUP]', '🧠': '[TRAIN]', '🔍': '[CHECK]',
            '🎯': '[TARGET]', '📈': '[PROGRESS]', '🔄': '[UPDATE]', '💾': '[SAVE]',
            '🎉': '[SUCCESS]', '🔧': '[FIX]', '📋': '[INFO]', '🏷️': '[LABEL]'
        }
        
        safe_message = message
        for emoji, replacement in emoji_replacements.items():
            safe_message = safe_message.replace(emoji, replacement)
        print(safe_message)
    else:
        print(message)

def safe_log_info(message):
    """ログとコンソール両方に安全に出力（修正版）"""
    # 絵文字を除去してログに記録
    clean_message = message
    for emoji in ['🚀', '✅', '❌', '⚠️', '📊', '📦', '🧠', '🔍', '🎯', '📈', '🔄', '💾', '🎉', '🔧', '📋', '🏷️']:
        clean_message = clean_message.replace(emoji, '').strip()
    
    # 無限再帰を避けるため、直接logger.infoを呼ぶ
    logger.info(clean_message)
    # コンソールには元のメッセージ（絵文字置換済み）を表示
    safe_print(message)

# 安全な出力関数
def safe_print(message):
    """Windows環境でも安全に絵文字を表示"""
    if sys.platform == "win32":
        emoji_replacements = {
            '🚀': '[START]', '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARN]',
            '📊': '[DATA]', '📦': '[BACKUP]', '🧠': '[TRAIN]', '🔍': '[CHECK]',
            '🎯': '[TARGET]', '📈': '[PROGRESS]', '🔄': '[UPDATE]', '💾': '[SAVE]',
            '🎉': '[SUCCESS]', '🔧': '[FIX]', '📋': '[INFO]', '🏷️': '[LABEL]'
        }
        
        safe_message = message
        for emoji, replacement in emoji_replacements.items():
            safe_message = safe_message.replace(emoji, replacement)
        print(safe_message)
    else:
        print(message)

def safe_log_info(message):
    """ログとコンソール両方に安全に出力"""
    clean_message = message
    for emoji in ['🚀', '✅', '❌', '⚠️', '📊', '📦', '🧠', '🔍', '🎯', '📈', '🔄', '💾', '🎉', '🔧', '📋', '🏷️']:
        clean_message = clean_message.replace(emoji, '').strip()
    
    logger.info(clean_message)
    safe_print(message)


class MonthlyModelUpdater:
    """月次モデル更新システム"""
    
    def __init__(self, 
                 new_data_path: str = None,
                 sample_size: int = 1000000,
                 n_models: int = 3,
                 epochs: int = 30,
                 backup_old_model: bool = True,
                 auto_deploy: bool = False):
        """
        Args:
            new_data_path: 新しいティックデータのパス
            sample_size: 学習に使用するサンプル数
            n_models: アンサンブルモデル数
            epochs: 学習エポック数
            backup_old_model: 旧モデルをバックアップするか
            auto_deploy: 自動で本番適用するか
        """
        self.new_data_path = new_data_path or "data/usdjpy_ticks_new.csv"
        self.sample_size = sample_size
        self.n_models = n_models
        self.epochs = epochs
        self.backup_old_model = backup_old_model
        self.auto_deploy = auto_deploy
        
        # パス設定
        self.current_model_path = "models/best_confidence_model.h5"
        self.current_config_path = "config/production_config.json"
        self.backup_dir = "models/backup"
        self.new_model_path = "models/best_confidence_model_new.h5"
        self.new_config_path = "config/production_config_new.json"
        
        # 更新情報
        self.update_date = datetime.now()
        self.previous_performance = None
        self.new_performance = None
        self.update_successful = False
        
        # 必要なディレクトリ作成
        os.makedirs("models", exist_ok=True)
        os.makedirs("config", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        safe_log_info(f"月次更新システム初期化完了: {self.update_date.strftime('%Y年%m月')}")
    
    def verify_environment(self) -> bool:
        """環境確認"""
        safe_log_info("🔍 環境確認開始...")
        
        # 必要ファイルの確認
        required_files = [
            "run_confidence_fix.py",
            "phase4_enhanced_confidence.py",
            "utils.py"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            logger.error(f"❌ 必要ファイルが不足: {missing_files}")
            return False
        
        # 新データファイル確認
        if not os.path.exists(self.new_data_path):
            logger.error(f"❌ 新データファイルが見つかりません: {self.new_data_path}")
            return False
        
        # ファイルサイズ確認
        file_size = os.path.getsize(self.new_data_path) / (1024 * 1024)
        if file_size < 50:  # 50MB未満
            logger.warning(f"⚠️ データファイルが小さいです: {file_size:.1f}MB")
        else:
            safe_log_info(f"✅ データファイルサイズ: {file_size:.1f}MB")
        
        # ディスク容量確認
        free_space = shutil.disk_usage(".").free / (1024 * 1024 * 1024)
        if free_space < 10:  # 10GB未満
            logger.warning(f"⚠️ ディスク容量不足: {free_space:.1f}GB")
        else:
            safe_log_info(f"✅ ディスク容量: {free_space:.1f}GB")
        
        safe_log_info("✅ 環境確認完了")
        return True
    
    def backup_current_model(self) -> bool:
        """現在のモデルをバックアップ"""
        if not self.backup_old_model:
            safe_log_info("📦 バックアップはスキップされました")
            return True
        
        safe_log_info("📦 現在のモデルをバックアップ中...")
        
        try:
            backup_suffix = self.update_date.strftime("%Y%m")
            
            # モデルファイルのバックアップ
            if os.path.exists(self.current_model_path):
                backup_model_path = f"{self.backup_dir}/best_confidence_model_{backup_suffix}.h5"
                shutil.copy2(self.current_model_path, backup_model_path)
                safe_log_info(f"✅ モデルバックアップ: {backup_model_path}")
            
            # 設定ファイルのバックアップ
            if os.path.exists(self.current_config_path):
                backup_config_path = f"{self.backup_dir}/production_config_{backup_suffix}.json"
                shutil.copy2(self.current_config_path, backup_config_path)
                safe_log_info(f"✅ 設定バックアップ: {backup_config_path}")
            
            # 性能記録の保存
            if os.path.exists("logs/model_performance.json"):
                backup_perf_path = f"{self.backup_dir}/model_performance_{backup_suffix}.json"
                shutil.copy2("logs/model_performance.json", backup_perf_path)
                safe_log_info(f"✅ 性能記録バックアップ: {backup_perf_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ バックアップエラー: {e}")
            return False
    
    def load_previous_performance(self) -> Optional[Dict]:
        """前回の性能データを読み込み"""
        try:
            if os.path.exists("logs/model_performance.json"):
                with open("logs/model_performance.json", 'r') as f:
                    self.previous_performance = json.load(f)
                    safe_log_info(f"📊 前回性能読み込み完了")
                    return self.previous_performance
        except Exception as e:
            logger.warning(f"⚠️ 前回性能データ読み込みエラー: {e}")
        
        return None
    
    def train_new_model(self) -> bool:
        """新しいモデルを学習"""
        safe_log_info(f"🧠 新モデル学習開始...")
        safe_log_info(f"   データ: {self.new_data_path}")
        safe_log_info(f"   サンプル: {self.sample_size:,}")
        safe_log_info(f"   モデル数: {self.n_models}")
        safe_log_info(f"   エポック: {self.epochs}")
        
        try:
            # run_confidence_fix.py を新データで実行
            import subprocess
            
            # 一時的に新データを標準の場所にコピー
            temp_data_path = "data/usdjpy_ticks.csv.bak"
            standard_data_path = "data/usdjpy_ticks.csv"
            
            # 元データのバックアップ
            if os.path.exists(standard_data_path):
                shutil.copy2(standard_data_path, temp_data_path)
            
            # 新データを標準の場所にコピー
            shutil.copy2(self.new_data_path, standard_data_path)
            
            safe_log_info("🔄 新データで学習システム実行中...")
            
            # 学習実行（capture_output=Falseで進捗表示）
            result = subprocess.run([
                sys.executable, "run_confidence_fix.py"
            ], text=True, capture_output=False)
            
            # 元データを復元
            if os.path.exists(temp_data_path):
                shutil.copy2(temp_data_path, standard_data_path)
                os.remove(temp_data_path)
            
            if result.returncode == 0:
                safe_log_info("✅ 新モデル学習完了")
                return True
            else:
                logger.error(f"❌ 学習エラー（終了コード: {result.returncode}）")
                return False
                
        except Exception as e:
            logger.error(f"❌ 学習実行エラー: {e}")
            return False
    
    def evaluate_new_model(self) -> bool:
        """新モデルの性能評価"""
        safe_log_info("📊 新モデル性能評価中...")
        
        try:
            # 最新の結果ファイルを探す
            log_files = list(Path("logs").glob("confidence_fix_results_*.json"))
            if not log_files:
                logger.error("❌ 結果ファイルが見つかりません")
                return False
            
            # 最新のファイルを取得
            latest_file = max(log_files, key=os.path.getctime)
            
            with open(latest_file, 'r') as f:
                results = json.load(f)
            
            # 性能指標を抽出
            improvements = results.get('improvements', {})
            
            self.new_performance = {
                'timestamp': self.update_date.isoformat(),
                'data_source': self.new_data_path,
                'ece_original': improvements.get('original_ece', 0),
                'ece_improved': improvements.get('improved_ece', 0),
                'ece_improvement': improvements.get('improvement', 0),
                'optimal_temperature': improvements.get('optimal_temperature', 1.0),
                'calibration_success': improvements.get('success', False),
                'model_file': latest_file.name
            }
            
            safe_log_info(f"📊 新モデル性能:")
            safe_log_info(f"   ECE改善前: {self.new_performance['ece_original']:.4f}")
            safe_log_info(f"   ECE改善後: {self.new_performance['ece_improved']:.4f}")
            safe_log_info(f"   改善幅: {self.new_performance['ece_improvement']:+.4f}")
            safe_log_info(f"   最適温度: {self.new_performance['optimal_temperature']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 性能評価エラー: {e}")
            return False
    
    def compare_performance(self) -> bool:
        """新旧モデルの性能比較"""
        safe_log_info("🔍 性能比較実行中...")
        
        if not self.new_performance:
            logger.error("❌ 新モデルの性能データがありません")
            return False
        
        # 新モデルの基本判定
        new_ece = self.new_performance['ece_improved']
        new_improvement = self.new_performance['ece_improvement']
        new_calibration_success = self.new_performance['calibration_success']
        
        safe_log_info("📊 性能比較結果:")
        
        # 基本品質チェック
        quality_checks = []
        
        if new_ece < 0.05:
            safe_log_info("   ✅ ECE目標達成 (<0.05)")
            quality_checks.append(True)
        else:
            logger.warning(f"   ⚠️ ECE目標未達成: {new_ece:.4f}")
            quality_checks.append(False)
        
        if new_improvement > 0.02:
            safe_log_info(f"   ✅ 大幅改善達成: {new_improvement:+.4f}")
            quality_checks.append(True)
        elif new_improvement > 0.01:
            safe_log_info(f"   ✅ 改善達成: {new_improvement:+.4f}")
            quality_checks.append(True)
        else:
            logger.warning(f"   ⚠️ 改善幅が小さい: {new_improvement:+.4f}")
            quality_checks.append(False)
        
        if new_calibration_success:
            safe_log_info("   ✅ キャリブレーション成功")
            quality_checks.append(True)
        else:
            logger.warning("   ⚠️ キャリブレーション課題あり")
            quality_checks.append(False)
        
        # 前回モデルとの比較
        if self.previous_performance:
            prev_ece = self.previous_performance.get('ece_improved', 999)
            prev_improvement = self.previous_performance.get('ece_improvement', 0)
            
            safe_log_info("📈 前回モデルとの比較:")
            
            if new_ece < prev_ece:
                safe_log_info(f"   ✅ ECE改善: {prev_ece:.4f} → {new_ece:.4f}")
                quality_checks.append(True)
            else:
                logger.warning(f"   ⚠️ ECE悪化: {prev_ece:.4f} → {new_ece:.4f}")
                quality_checks.append(False)
            
            if new_improvement > prev_improvement:
                safe_log_info(f"   ✅ 改善幅向上: {prev_improvement:+.4f} → {new_improvement:+.4f}")
                quality_checks.append(True)
            else:
                safe_log_info(f"   📊 改善幅比較: {prev_improvement:+.4f} → {new_improvement:+.4f}")
        
        # 総合判定
        success_ratio = sum(quality_checks) / len(quality_checks)
        
        if success_ratio >= 0.8:
            safe_log_info(f"🎉 新モデル採用推奨: 品質チェック {sum(quality_checks)}/{len(quality_checks)} 通過")
            return True
        elif success_ratio >= 0.6:
            logger.warning(f"⚠️ 新モデル要検討: 品質チェック {sum(quality_checks)}/{len(quality_checks)} 通過")
            return False  # 手動判断を要求
        else:
            logger.error(f"❌ 新モデル採用非推奨: 品質チェック {sum(quality_checks)}/{len(quality_checks)} 通過")
            return False
    
    def deploy_new_model(self) -> bool:
        """新モデルを本番環境に適用"""
        safe_log_info("🚀 新モデル本番適用開始...")
        
        try:
            # 最新の設定ファイルを探す
            config_files = list(Path("config").glob("production_config_*.json"))
            if not config_files:
                logger.error("❌ 新しい設定ファイルが見つかりません")
                return False
            
            latest_config = max(config_files, key=os.path.getctime)
            
            # 本番設定ファイルに上書き
            shutil.copy2(latest_config, self.current_config_path)
            safe_log_info(f"✅ 設定ファイル更新: {self.current_config_path}")
            
            # MT5用設定ファイル生成
            self._generate_mt5_config()
            
            # 性能記録の更新
            self._save_performance_record()
            
            safe_log_info("✅ 新モデル本番適用完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ 本番適用エラー: {e}")
            return False
    
    def _generate_mt5_config(self):
        """MT5用設定ファイル生成"""
        try:
            # 本番設定を読み込み
            with open(self.current_config_path, 'r') as f:
                config = json.load(f)
            
            # MT5用設定を抽出・変換
            mt5_config = {
                'model_info': {
                    'update_date': self.update_date.isoformat(),
                    'model_version': self.update_date.strftime("v%Y%m"),
                    'data_source': self.new_data_path
                },
                'confidence_system': {
                    'optimal_temperature': self.new_performance['optimal_temperature'],
                    'base_threshold': 0.58,
                    'upper_threshold': 0.61,
                    'smooth_filtering': True,
                    'calibration_enabled': True
                },
                'trading_parameters': {
                    'min_confidence_for_trade': 0.50,
                    'max_confidence_for_scaling': 0.90,
                    'lot_size_base': 0.01,
                    'lot_size_max': 0.10,
                    'tp_pips': 4.0,
                    'sl_pips': 5.0
                },
                'risk_management': {
                    'max_daily_trades': 50,
                    'max_concurrent_trades': 3,
                    'daily_loss_limit_pips': 50,
                    'weekly_loss_limit_pips': 150
                },
                'monitoring': {
                    'enable_performance_tracking': True,
                    'log_all_signals': True,
                    'alert_on_unusual_activity': True
                }
            }
            
            # MT5設定ファイル保存
            mt5_config_path = "config/mt5_config.json"
            with open(mt5_config_path, 'w') as f:
                json.dump(mt5_config, f, indent=2, ensure_ascii=False)
            
            safe_log_info(f"✅ MT5設定ファイル生成: {mt5_config_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ MT5設定生成エラー: {e}")
    
    def _save_performance_record(self):
        """性能記録保存"""
        try:
            performance_record = {
                'current_model': self.new_performance,
                'update_history': []
            }
            
            # 既存の履歴があれば読み込み
            perf_file = "logs/model_performance.json"
            if os.path.exists(perf_file):
                with open(perf_file, 'r') as f:
                    existing_record = json.load(f)
                    
                # 前回のモデルを履歴に追加
                if 'current_model' in existing_record:
                    performance_record['update_history'] = existing_record.get('update_history', [])
                    performance_record['update_history'].append(existing_record['current_model'])
            
            # 履歴は最新10件まで保持
            performance_record['update_history'] = performance_record['update_history'][-10:]
            
            # 保存
            with open(perf_file, 'w') as f:
                json.dump(performance_record, f, indent=2, ensure_ascii=False)
            
            safe_log_info("✅ 性能記録更新完了")
            
        except Exception as e:
            logger.warning(f"⚠️ 性能記録保存エラー: {e}")
    
    def rollback_to_previous_model(self) -> bool:
        """前回モデルにロールバック"""
        safe_log_info("🔄 前回モデルにロールバック中...")
        
        try:
            backup_suffix = (self.update_date - timedelta(days=30)).strftime("%Y%m")
            
            # バックアップファイルの確認
            backup_model = f"{self.backup_dir}/best_confidence_model_{backup_suffix}.h5"
            backup_config = f"{self.backup_dir}/production_config_{backup_suffix}.json"
            
            if not os.path.exists(backup_model):
                logger.error(f"❌ バックアップモデルが見つかりません: {backup_model}")
                return False
            
            if not os.path.exists(backup_config):
                logger.error(f"❌ バックアップ設定が見つかりません: {backup_config}")
                return False
            
            # ロールバック実行
            shutil.copy2(backup_model, self.current_model_path)
            shutil.copy2(backup_config, self.current_config_path)
            
            safe_log_info("✅ ロールバック完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ ロールバックエラー: {e}")
            return False
    
    def generate_update_report(self) -> str:
        """更新レポート生成"""
        report_lines = []
        
        # ヘッダー
        report_lines.append("=" * 60)
        report_lines.append(f"   月次モデル更新レポート")
        report_lines.append(f"   更新日時: {self.update_date.strftime('%Y年%m月%d日 %H:%M')}")
        report_lines.append("=" * 60)
        
        # データ情報
        report_lines.append(f"\n📊 データ情報:")
        report_lines.append(f"   新データ: {self.new_data_path}")
        if os.path.exists(self.new_data_path):
            file_size = os.path.getsize(self.new_data_path) / (1024 * 1024)
            report_lines.append(f"   データサイズ: {file_size:.1f}MB")
        
        # 学習設定
        report_lines.append(f"\n🧠 学習設定:")
        report_lines.append(f"   サンプルサイズ: {self.sample_size:,}")
        report_lines.append(f"   アンサンブルモデル数: {self.n_models}")
        report_lines.append(f"   学習エポック数: {self.epochs}")
        
        # 性能比較
        if self.new_performance:
            report_lines.append(f"\n📈 新モデル性能:")
            report_lines.append(f"   ECE改善前: {self.new_performance['ece_original']:.4f}")
            report_lines.append(f"   ECE改善後: {self.new_performance['ece_improved']:.4f}")
            report_lines.append(f"   改善幅: {self.new_performance['ece_improvement']:+.4f}")
            report_lines.append(f"   最適温度: {self.new_performance['optimal_temperature']:.3f}")
            report_lines.append(f"   キャリブレーション: {'成功' if self.new_performance['calibration_success'] else '要改善'}")
        
        if self.previous_performance:
            report_lines.append(f"\n📊 前回モデルとの比較:")
            prev_ece = self.previous_performance.get('ece_improved', 0)
            new_ece = self.new_performance['ece_improved']
            
            if new_ece < prev_ece:
                report_lines.append(f"   ECE変化: {prev_ece:.4f} → {new_ece:.4f} (改善)")
            else:
                report_lines.append(f"   ECE変化: {prev_ece:.4f} → {new_ece:.4f} (悪化)")
        
        # 更新結果
        report_lines.append(f"\n🎯 更新結果:")
        if self.update_successful:
            report_lines.append("   ✅ モデル更新成功")
            report_lines.append("   ✅ 本番環境適用完了")
        else:
            report_lines.append("   ❌ モデル更新失敗")
            report_lines.append("   🔄 前回モデル継続使用")
        
        # 次回推奨事項
        report_lines.append(f"\n💡 次回推奨事項:")
        if self.new_performance and self.new_performance['ece_improved'] > 0.05:
            report_lines.append("   - より多くのデータでの学習")
            report_lines.append("   - キャリブレーション手法の見直し")
        if self.new_performance and self.new_performance['ece_improvement'] < 0.02:
            report_lines.append("   - 特徴量エンジニアリングの強化")
            report_lines.append("   - モデルアーキテクチャの調整")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def run_monthly_update(self) -> bool:
        """月次更新の実行"""
        safe_log_info("🚀" * 30)
        safe_log_info(f"    月次モデル更新開始: {self.update_date.strftime('%Y年%m月')}")
        safe_log_info("🚀" * 30)
        
        try:
            # Phase 1: 環境確認
            if not self.verify_environment():
                return False
            
            # Phase 2: バックアップ
            if not self.backup_current_model():
                return False
            
            # Phase 3: 前回性能読み込み
            self.load_previous_performance()
            
            # Phase 4: 新モデル学習
            if not self.train_new_model():
                logger.error("❌ 新モデル学習に失敗しました")
                return False
            
            # Phase 5: 性能評価
            if not self.evaluate_new_model():
                logger.error("❌ 新モデル評価に失敗しました")
                return False
            
            # Phase 6: 性能比較・判定
            should_deploy = self.compare_performance()
            
            if should_deploy or self.auto_deploy:
                # Phase 7: 本番適用
                if self.deploy_new_model():
                    self.update_successful = True
                    safe_log_info("🎉 月次モデル更新完了！")
                else:
                    logger.error("❌ 本番適用に失敗しました")
                    return False
            else:
                logger.warning("⚠️ 新モデルの性能が不十分のため、更新を見送りました")
                if not self.auto_deploy:
                    user_choice = input("手動で更新を実行しますか？ (y/N): ")
                    if user_choice.lower() == 'y':
                        if self.deploy_new_model():
                            self.update_successful = True
                            safe_log_info("🎉 手動更新完了！")
                        else:
                            logger.error("❌ 手動更新に失敗しました")
                            return False
            
            # Phase 8: レポート生成
            report = self.generate_update_report()
            
            # レポート保存
            report_file = f"logs/monthly_update_report_{self.update_date.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # レポート表示
            print("\n" + report)
            safe_log_info(f"📄 更新レポート保存: {report_file}")
            
            return self.update_successful
            
        except Exception as e:
            logger.error(f"❌ 月次更新エラー: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="月次モデル更新システム")
    
    parser.add_argument("--new_data", 
                       default="data/usdjpy_ticks_new.csv",
                       help="新しいティックデータのパス")
    
    parser.add_argument("--sample_size", 
                       type=int, 
                       default=1000000,
                       help="学習に使用するサンプル数")
    
    parser.add_argument("--models", 
                       type=int, 
                       default=3,
                       help="アンサンブルモデル数")
    
    parser.add_argument("--epochs", 
                       type=int, 
                       default=30,
                       help="学習エポック数")
    
    parser.add_argument("--backup_old_model", 
                       action="store_true", 
                       default=True,
                       help="旧モデルをバックアップする")
    
    parser.add_argument("--auto_deploy", 
                       action="store_true",
                       help="自動で本番適用する")
    
    parser.add_argument("--rollback", 
                       action="store_true",
                       help="前回モデルにロールバックする")
    
    args = parser.parse_args()
    
    # ロールバック実行
    if args.rollback:
        updater = MonthlyModelUpdater()
        if updater.rollback_to_previous_model():
            print("✅ ロールバック完了")
        else:
            print("❌ ロールバック失敗")
        return
    
    # 通常の月次更新実行
    updater = MonthlyModelUpdater(
        new_data_path=args.new_data,
        sample_size=args.sample_size,
        n_models=args.models,
        epochs=args.epochs,
        backup_old_model=args.backup_old_model,
        auto_deploy=args.auto_deploy
    )
    
    success = updater.run_monthly_update()
    
    if success:
        print("\n🎉 月次モデル更新が正常に完了しました！")
        print("   新しいモデルが本番環境で稼働中です")
        exit(0)
    else:
        print("\n❌ 月次モデル更新に問題が発生しました")
        print("   ログファイルを確認して対処してください")
        exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ ユーザーによって中断されました")
        exit(1)
    except Exception as e:
        print(f"\n❌ システムエラー: {e}")
        import traceback
        traceback.print_exc()
        exit(1)