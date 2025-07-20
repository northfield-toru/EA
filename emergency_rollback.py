"""
緊急ロールバックスクリプト

使用方法:
python emergency_rollback.py --restore_date 202407
python emergency_rollback.py --list_backups
python emergency_rollback.py --auto_restore
"""

import os
import sys
import json
import shutil
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmergencyRollback:
    """緊急ロールバックシステム"""
    
    def __init__(self):
        self.backup_dir = "models/backup"
        self.current_model_path = "models/best_confidence_model.h5"
        self.current_config_path = "config/production_config.json"
        self.mt5_config_path = "config/mt5_config.json"
        
    def list_available_backups(self) -> list:
        """利用可能なバックアップ一覧"""
        backups = []
        
        if not os.path.exists(self.backup_dir):
            logger.warning("❌ バックアップディレクトリが見つかりません")
            return backups
        
        # モデルファイルのバックアップを探す
        model_backups = list(Path(self.backup_dir).glob("best_confidence_model_*.h5"))
        
        for model_file in model_backups:
            # ファイル名から日付を抽出
            filename = model_file.stem
            date_part = filename.split('_')[-1]
            
            try:
                backup_date = datetime.strptime(date_part, "%Y%m")
                
                # 対応する設定ファイルがあるかチェック
                config_file = self.backup_dir + f"/production_config_{date_part}.json"
                perf_file = self.backup_dir + f"/model_performance_{date_part}.json"
                
                backup_info = {
                    'date': date_part,
                    'datetime': backup_date,
                    'model_file': str(model_file),
                    'config_file': config_file if os.path.exists(config_file) else None,
                    'performance_file': perf_file if os.path.exists(perf_file) else None,
                    'model_size': os.path.getsize(model_file) / (1024*1024)  # MB
                }
                
                backups.append(backup_info)
                
            except ValueError:
                logger.warning(f"⚠️ 日付形式が不正なファイル: {model_file}")
                continue
        
        # 日付順でソート（新しい順）
        backups.sort(key=lambda x: x['datetime'], reverse=True)
        
        return backups
    
    def display_backup_list(self, backups: list):
        """バックアップ一覧表示"""
        if not backups:
            print("❌ 利用可能なバックアップがありません")
            return
        
        print("\n📋 利用可能なバックアップ:")
        print("-" * 80)
        print(f"{'No.':<4} {'日付':<8} {'モデルサイズ':<12} {'設定ファイル':<10} {'性能記録':<10}")
        print("-" * 80)
        
        for i, backup in enumerate(backups, 1):
            config_status = "✅" if backup['config_file'] else "❌"
            perf_status = "✅" if backup['performance_file'] else "❌"
            
            print(f"{i:<4} {backup['date']:<8} {backup['model_size']:.1f}MB      "
                  f"{config_status:<10} {perf_status:<10}")
        
        print("-" * 80)
    
    def get_backup_performance(self, backup_info: dict) -> dict:
        """バックアップの性能情報取得"""
        if not backup_info['performance_file']:
            return {}
        
        try:
            with open(backup_info['performance_file'], 'r') as f:
                perf_data = json.load(f)
            
            # current_modelから情報を抽出
            current_model = perf_data.get('current_model', {})
            
            return {
                'ece_improved': current_model.get('ece_improved', 0),
                'ece_improvement': current_model.get('ece_improvement', 0),
                'optimal_temperature': current_model.get('optimal_temperature', 1.0),
                'calibration_success': current_model.get('calibration_success', False),
                'timestamp': current_model.get('timestamp', 'unknown')
            }
            
        except Exception as e:
            logger.warning(f"⚠️ 性能データ読み込みエラー: {e}")
            return {}
    
    def restore_backup(self, backup_info: dict) -> bool:
        """指定されたバックアップを復元"""
        logger.info(f"🔄 バックアップ復元開始: {backup_info['date']}")
        
        try:
            # 現在のファイルをバックアップ
            current_backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if os.path.exists(self.current_model_path):
                emergency_backup_model = f"models/emergency_backup_model_{current_backup_suffix}.h5"
                shutil.copy2(self.current_model_path, emergency_backup_model)
                logger.info(f"📦 現在のモデルを緊急バックアップ: {emergency_backup_model}")
            
            if os.path.exists(self.current_config_path):
                emergency_backup_config = f"config/emergency_backup_config_{current_backup_suffix}.json"
                shutil.copy2(self.current_config_path, emergency_backup_config)
                logger.info(f"📦 現在の設定を緊急バックアップ: {emergency_backup_config}")
            
            # モデルファイル復元
            shutil.copy2(backup_info['model_file'], self.current_model_path)
            logger.info(f"✅ モデルファイル復元: {self.current_model_path}")
            
            # 設定ファイル復元
            if backup_info['config_file'] and os.path.exists(backup_info['config_file']):
                shutil.copy2(backup_info['config_file'], self.current_config_path)
                logger.info(f"✅ 設定ファイル復元: {self.current_config_path}")
                
                # MT5設定も更新
                self._update_mt5_config_from_backup(backup_info)
            else:
                logger.warning("⚠️ 設定ファイルが見つかりません。手動で設定が必要です")
            
            # 復元記録の保存
            self._save_restore_record(backup_info, current_backup_suffix)
            
            logger.info("✅ バックアップ復元完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ 復元エラー: {e}")
            return False
    
    def _update_mt5_config_from_backup(self, backup_info: dict):
        """バックアップからMT5設定を更新"""
        try:
            # バックアップの設定から情報を取得
            with open(backup_info['config_file'], 'r') as f:
                backup_config = json.load(f)
            
            # 性能データから温度パラメータを取得
            perf_data = self.get_backup_performance(backup_info)
            optimal_temperature = perf_data.get('optimal_temperature', 1.0)
            
            # MT5設定を更新
            mt5_config = {
                'model_info': {
                    'restore_date': datetime.now().isoformat(),
                    'restored_from': backup_info['date'],
                    'model_version': f"restored_v{backup_info['date']}"
                },
                'confidence_system': {
                    'optimal_temperature': optimal_temperature,
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
                    'alert_on_unusual_activity': True,
                    'restored_model': True
                }
            }
            
            with open(self.mt5_config_path, 'w') as f:
                json.dump(mt5_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ MT5設定更新完了: 温度={optimal_temperature:.3f}")
            
        except Exception as e:
            logger.warning(f"⚠️ MT5設定更新エラー: {e}")
    
    def _save_restore_record(self, backup_info: dict, emergency_backup_suffix: str):
        """復元記録の保存"""
        try:
            restore_record = {
                'restore_timestamp': datetime.now().isoformat(),
                'restored_from': backup_info['date'],
                'restored_model': backup_info['model_file'],
                'restored_config': backup_info['config_file'],
                'emergency_backup_suffix': emergency_backup_suffix,
                'performance_at_restore': self.get_backup_performance(backup_info)
            }
            
            # 復元履歴ファイルに追加
            restore_history_file = "logs/restore_history.json"
            history = []
            
            if os.path.exists(restore_history_file):
                with open(restore_history_file, 'r') as f:
                    history = json.load(f)
            
            history.append(restore_record)
            
            # 最新10件のみ保持
            history = history[-10:]
            
            with open(restore_history_file, 'w') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ 復元記録保存完了")
            
        except Exception as e:
            logger.warning(f"⚠️ 復元記録保存エラー: {e}")
    
    def auto_restore_latest(self) -> bool:
        """最新のバックアップを自動復元"""
        logger.info("🤖 最新バックアップの自動復元開始...")
        
        backups = self.list_available_backups()
        
        if not backups:
            logger.error("❌ 復元可能なバックアップがありません")
            return False
        
        # 最新のバックアップを選択
        latest_backup = backups[0]
        
        logger.info(f"📅 最新バックアップを復元: {latest_backup['date']}")
        
        # 性能情報表示
        perf_data = self.get_backup_performance(latest_backup)
        if perf_data:
            logger.info(f"📊 復元予定モデルの性能:")
            logger.info(f"   ECE: {perf_data.get('ece_improved', 0):.4f}")
            logger.info(f"   改善幅: {perf_data.get('ece_improvement', 0):+.4f}")
            logger.info(f"   温度: {perf_data.get('optimal_temperature', 1.0):.3f}")
        
        return self.restore_backup(latest_backup)
    
    def interactive_restore(self) -> bool:
        """対話的復元"""
        print("🔄 対話的バックアップ復元")
        
        backups = self.list_available_backups()
        
        if not backups:
            print("❌ 復元可能なバックアップがありません")
            return False
        
        # バックアップ一覧表示
        self.display_backup_list(backups)
        
        # 詳細情報付きで表示
        print("\n📊 各バックアップの詳細:")
        for i, backup in enumerate(backups, 1):
            perf_data = self.get_backup_performance(backup)
            print(f"\n{i}. {backup['date']} ({backup['datetime'].strftime('%Y年%m月')})")
            if perf_data:
                print(f"   ECE: {perf_data.get('ece_improved', 0):.4f}")
                print(f"   改善幅: {perf_data.get('ece_improvement', 0):+.4f}")
                print(f"   温度: {perf_data.get('optimal_temperature', 1.0):.3f}")
                print(f"   状態: {'キャリブレーション成功' if perf_data.get('calibration_success') else 'キャリブレーション課題'}")
        
        # ユーザー選択
        while True:
            try:
                choice = input(f"\n復元するバックアップを選択してください (1-{len(backups)}, q=キャンセル): ")
                
                if choice.lower() == 'q':
                    print("❌ 復元をキャンセルしました")
                    return False
                
                index = int(choice) - 1
                
                if 0 <= index < len(backups):
                    selected_backup = backups[index]
                    
                    # 最終確認
                    confirm = input(f"\n{selected_backup['date']}のバックアップを復元しますか？ (y/N): ")
                    
                    if confirm.lower() == 'y':
                        return self.restore_backup(selected_backup)
                    else:
                    print("❌ 無効な選択です")
                    
            except ValueError:
                print("❌ 数字を入力してください")
    
    def verify_current_system(self) -> dict:
        """現在のシステム状態確認"""
        status = {
            'model_exists': os.path.exists(self.current_model_path),
            'config_exists': os.path.exists(self.current_config_path),
            'mt5_config_exists': os.path.exists(self.mt5_config_path),
            'system_healthy': True,
            'issues': []
        }
        
        # モデルファイル確認
        if not status['model_exists']:
            status['issues'].append("モデルファイルが見つかりません")
            status['system_healthy'] = False
        else:
            model_size = os.path.getsize(self.current_model_path) / (1024*1024)
            if model_size < 1:  # 1MB未満
                status['issues'].append(f"モデルファイルが小さすぎます: {model_size:.1f}MB")
                status['system_healthy'] = False
        
        # 設定ファイル確認
        if not status['config_exists']:
            status['issues'].append("設定ファイルが見つかりません")
            status['system_healthy'] = False
        else:
            try:
                with open(self.current_config_path, 'r') as f:
                    config = json.load(f)
                
                # 重要な設定項目確認
                if 'confidence_system' not in config:
                    status['issues'].append("信頼度システム設定が不足")
                    status['system_healthy'] = False
                
            except json.JSONDecodeError:
                status['issues'].append("設定ファイルが破損しています")
                status['system_healthy'] = False
        
        return status
    
    def emergency_diagnosis(self):
        """緊急診断"""
        print("🚨 緊急システム診断開始...")
        
        # 現在のシステム状態確認
        status = self.verify_current_system()
        
        print(f"\n📊 現在のシステム状態:")
        print(f"   モデルファイル: {'✅' if status['model_exists'] else '❌'}")
        print(f"   設定ファイル: {'✅' if status['config_exists'] else '❌'}")
        print(f"   MT5設定: {'✅' if status['mt5_config_exists'] else '❌'}")
        print(f"   システム健全性: {'✅' if status['system_healthy'] else '❌'}")
        
        if status['issues']:
            print(f"\n⚠️ 検出された問題:")
            for issue in status['issues']:
                print(f"   - {issue}")
        
        # バックアップ状況確認
        backups = self.list_available_backups()
        print(f"\n📦 利用可能なバックアップ: {len(backups)}件")
        
        if backups:
            latest_backup = backups[0]
            print(f"   最新バックアップ: {latest_backup['date']}")
            
            perf_data = self.get_backup_performance(latest_backup)
            if perf_data:
                print(f"   最新バックアップ性能: ECE={perf_data.get('ece_improved', 0):.4f}")
        
        # 推奨アクション
        print(f"\n💡 推奨アクション:")
        if not status['system_healthy']:
            if backups:
                print("   🔄 最新バックアップからの復元を推奨")
                print("   コマンド: python emergency_rollback.py --auto_restore")
            else:
                print("   🚨 バックアップが不足。完全再構築が必要")
        else:
            print("   ✅ システムは正常に動作しています")
            if len(backups) < 3:
                print("   📦 バックアップの増強を推奨")
        
        return status

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="緊急ロールバックシステム")
    
    parser.add_argument("--restore_date", 
                       help="復元する日付 (YYYYMM形式)")
    
    parser.add_argument("--list_backups", 
                       action="store_true",
                       help="利用可能なバックアップ一覧表示")
    
    parser.add_argument("--auto_restore", 
                       action="store_true",
                       help="最新バックアップを自動復元")
    
    parser.add_argument("--interactive", 
                       action="store_true",
                       help="対話的復元")
    
    parser.add_argument("--diagnose", 
                       action="store_true",
                       help="緊急診断実行")
    
    args = parser.parse_args()
    
    rollback = EmergencyRollback()
    
    # 緊急診断
    if args.diagnose:
        status = rollback.emergency_diagnosis()
        return 0 if status['system_healthy'] else 1
    
    # バックアップ一覧表示
    if args.list_backups:
        backups = rollback.list_available_backups()
        rollback.display_backup_list(backups)
        
        # 詳細情報も表示
        if backups:
            print("\n📊 詳細情報:")
            for backup in backups[:3]:  # 最新3件
                perf_data = rollback.get_backup_performance(backup)
                if perf_data:
                    print(f"\n{backup['date']}:")
                    print(f"   ECE: {perf_data.get('ece_improved', 0):.4f}")
                    print(f"   温度: {perf_data.get('optimal_temperature', 1.0):.3f}")
        return 0
    
    # 自動復元
    if args.auto_restore:
        if rollback.auto_restore_latest():
            print("✅ 自動復元完了")
            return 0
        else:
            print("❌ 自動復元失敗")
            return 1
    
    # 対話的復元
    if args.interactive:
        if rollback.interactive_restore():
            print("✅ 対話的復元完了")
            return 0
        else:
            print("❌ 対話的復元失敗")
            return 1
    
    # 日付指定復元
    if args.restore_date:
        backups = rollback.list_available_backups()
        
        # 指定された日付のバックアップを探す
        target_backup = None
        for backup in backups:
            if backup['date'] == args.restore_date:
                target_backup = backup
                break
        
        if target_backup:
            print(f"🔄 {args.restore_date}のバックアップを復元中...")
            
            # 性能情報表示
            perf_data = rollback.get_backup_performance(target_backup)
            if perf_data:
                print(f"📊 復元予定モデルの性能:")
                print(f"   ECE: {perf_data.get('ece_improved', 0):.4f}")
                print(f"   温度: {perf_data.get('optimal_temperature', 1.0):.3f}")
            
            if rollback.restore_backup(target_backup):
                print("✅ 指定日付復元完了")
                return 0
            else:
                print("❌ 指定日付復元失敗")
                return 1
        else:
            print(f"❌ {args.restore_date}のバックアップが見つかりません")
            print("利用可能なバックアップ:")
            rollback.display_backup_list(backups)
            return 1
    
    # 引数なしの場合は使用方法を表示
    print("🔄 緊急ロールバックシステム")
    print("\n使用方法:")
    print("  --list_backups      : バックアップ一覧表示")
    print("  --auto_restore      : 最新バックアップを自動復元")
    print("  --interactive       : 対話的復元")
    print("  --restore_date YYYYMM : 指定日付復元")
    print("  --diagnose          : 緊急診断")
    print("\n例:")
    print("  python emergency_rollback.py --diagnose")
    print("  python emergency_rollback.py --list_backups")
    print("  python emergency_rollback.py --auto_restore")
    print("  python emergency_rollback.py --restore_date 202407")
    
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ ユーザーによって中断されました")
        exit(1)
    except Exception as e:
        print(f"\n❌ システムエラー: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
                        print("❌ 復元をキャンセルしました")
                        return False
                else: