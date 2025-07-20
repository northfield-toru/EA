"""
日次性能監視スクリプト

使用方法:
python check_daily_performance.py --date today
python check_daily_performance.py --date 2024-07-20
python check_daily_performance.py --weekly_report
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyPerformanceMonitor:
    """日次性能監視システム"""
    
    def __init__(self):
        self.logs_dir = "logs"
        self.config_dir = "config" 
        self.mt5_config_path = "config/mt5_config.json"
        self.performance_log_path = "logs/daily_performance.json"
        
        # 警告閾値
        self.thresholds = {
            'min_daily_trades': 5,      # 最小日次取引数
            'max_daily_trades': 100,    # 最大日次取引数
            'min_win_rate': 0.45,       # 最小勝率
            'max_daily_loss': 50,       # 最大日次損失(pips)
            'min_avg_confidence': 0.55, # 最小平均信頼度
            'max_ece_drift': 0.10       # 最大ECEドリフト
        }
    
    def load_current_config(self) -> dict:
        """現在の設定を読み込み"""
        try:
            with open(self.mt5_config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"設定読み込みエラー: {e}")
            return {}
    
    def simulate_daily_trading_data(self, target_date: datetime) -> dict:
        """
        日次取引データをシミュレート（実際のMT5ログから読み込む想定）
        """
        np.random.seed(int(target_date.strftime("%Y%m%d")))
        
        # 取引数（5-30回程度）
        n_trades = np.random.randint(5, 31)
        
        # 各取引の結果
        trades = []
        for i in range(n_trades):
            # 信頼度（0.50-0.90）
            confidence = np.random.uniform(0.50, 0.90)
            
            # 勝率は信頼度に比例（信頼度が高いほど勝ちやすい）
            win_prob = 0.3 + confidence * 0.4  # 0.5-0.66程度
            is_win = np.random.random() < win_prob
            
            # 損益（勝ちの場合は+4pips、負けの場合は-5pips）
            pips = 4.0 if is_win else -5.0
            
            trade = {
                'timestamp': target_date.strftime("%Y-%m-%d") + f" {9+i//4:02d}:{(i%4)*15:02d}:00",
                'signal_type': np.random.choice(['BUY', 'SELL']),
                'confidence': confidence,
                'entry_price': 118.0 + np.random.uniform(-0.5, 0.5),
                'exit_price': 118.0 + np.random.uniform(-0.5, 0.5) + pips/100,
                'pips': pips,
                'is_win': is_win,
                'lot_size': 0.01 + confidence * 0.09  # 信頼度に比例したロットサイズ
            }
            
            trades.append(trade)
        
        return {
            'date': target_date.strftime("%Y-%m-%d"),
            'trades': trades,
            'total_trades': n_trades,
            'winning_trades': sum(1 for t in trades if t['is_win']),
            'total_pips': sum(t['pips'] for t in trades),
            'avg_confidence': np.mean([t['confidence'] for t in trades])
        }
    
    def calculate_daily_metrics(self, daily_data: dict) -> dict:
        """日次指標計算"""
        trades = daily_data['trades']
        
        if not trades:
            return {'error': 'No trades found'}
        
        # 基本指標
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['is_win'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pips = sum(t['pips'] for t in trades)
        avg_pips_per_trade = total_pips / total_trades if total_trades > 0 else 0
        
        # 信頼度関連
        confidences = [t['confidence'] for t in trades]
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        # 時間帯別分析
        hourly_performance = {}
        for trade in trades:
            hour = int(trade['timestamp'].split()[1].split(':')[0])
            if hour not in hourly_performance:
                hourly_performance[hour] = {'trades': 0, 'wins': 0, 'pips': 0}
            
            hourly_performance[hour]['trades'] += 1
            if trade['is_win']:
                hourly_performance[hour]['wins'] += 1
            hourly_performance[hour]['pips'] += trade['pips']
        
        # 信頼度区間別性能
        confidence_bands = {
            'low': [t for t in trades if t['confidence'] < 0.6],
            'medium': [t for t in trades if 0.6 <= t['confidence'] < 0.8],
            'high': [t for t in trades if t['confidence'] >= 0.8]
        }
        
        confidence_performance = {}
        for band, band_trades in confidence_bands.items():
            if band_trades:
                band_wins = sum(1 for t in band_trades if t['is_win'])
                confidence_performance[band] = {
                    'trades': len(band_trades),
                    'win_rate': band_wins / len(band_trades),
                    'avg_pips': sum(t['pips'] for t in band_trades) / len(band_trades)
                }
        
        return {
            'date': daily_data['date'],
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_pips_per_trade': avg_pips_per_trade,
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'hourly_performance': hourly_performance,
            'confidence_performance': confidence_performance,
            'best_hour': max(hourly_performance.items(), key=lambda x: x[1]['pips'])[0] if hourly_performance else None,
            'worst_hour': min(hourly_performance.items(), key=lambda x: x[1]['pips'])[0] if hourly_performance else None
        }
    
    def check_alerts(self, metrics: dict) -> list:
        """アラート条件チェック"""
        alerts = []
        
        # 取引数チェック
        if metrics['total_trades'] < self.thresholds['min_daily_trades']:
            alerts.append({
                'level': 'WARNING',
                'message': f"取引数が少なすぎます: {metrics['total_trades']} < {self.thresholds['min_daily_trades']}"
            })
        elif metrics['total_trades'] > self.thresholds['max_daily_trades']:
            alerts.append({
                'level': 'WARNING', 
                'message': f"取引数が多すぎます: {metrics['total_trades']} > {self.thresholds['max_daily_trades']}"
            })
        
        # 勝率チェック
        if metrics['win_rate'] < self.thresholds['min_win_rate']:
            alerts.append({
                'level': 'CRITICAL',
                'message': f"勝率が低すぎます: {metrics['win_rate']:.1%} < {self.thresholds['min_win_rate']:.1%}"
            })
        
        # 損失チェック
        if metrics['total_pips'] < -self.thresholds['max_daily_loss']:
            alerts.append({
                'level': 'CRITICAL',
                'message': f"日次損失が限界を超過: {metrics['total_pips']:.1f}pips < -{self.thresholds['max_daily_loss']}pips"
            })
        
        # 信頼度チェック
        if metrics['avg_confidence'] < self.thresholds['min_avg_confidence']:
            alerts.append({
                'level': 'WARNING',
                'message': f"平均信頼度が低いです: {metrics['avg_confidence']:.3f} < {self.thresholds['min_avg_confidence']}"
            })
        
        return alerts
    
    def generate_daily_report(self, metrics: dict, alerts: list) -> str:
        """日次レポート生成"""
        report_lines = []
        
        # ヘッダー
        report_lines.append("=" * 60)
        report_lines.append(f"   日次性能レポート: {metrics['date']}")
        report_lines.append("=" * 60)
        
        # アラート（最優先）
        if alerts:
            report_lines.append(f"\n🚨 アラート ({len(alerts)}件):")
            for alert in alerts:
                level_icon = "🔴" if alert['level'] == 'CRITICAL' else "⚠️"
                report_lines.append(f"   {level_icon} {alert['message']}")
        else:
            report_lines.append(f"\n✅ アラート: なし")
        
        # 基本指標
        report_lines.append(f"\n📊 基本指標:")
        report_lines.append(f"   取引数: {metrics['total_trades']} (勝ち: {metrics['winning_trades']})")
        report_lines.append(f"   勝率: {metrics['win_rate']:.1%}")
        report_lines.append(f"   損益: {metrics['total_pips']:+.1f}pips")
        report_lines.append(f"   平均/取引: {metrics['avg_pips_per_trade']:+.2f}pips")
        
        # 信頼度分析
        report_lines.append(f"\n🎯 信頼度分析:")
        report_lines.append(f"   平均信頼度: {metrics['avg_confidence']:.3f}")
        report_lines.append(f"   信頼度標準偏差: {metrics['confidence_std']:.3f}")
        
        # 信頼度区間別性能
        if metrics['confidence_performance']:
            report_lines.append(f"\n📈 信頼度区間別性能:")
            for band, perf in metrics['confidence_performance'].items():
                band_name = {'low': '低(~0.6)', 'medium': '中(0.6-0.8)', 'high': '高(0.8~)'}[band]
                report_lines.append(f"   {band_name}: {perf['trades']}取引, "
                                  f"勝率{perf['win_rate']:.1%}, "
                                  f"平均{perf['avg_pips']:+.1f}pips")
        
        # 時間帯分析
        if metrics['hourly_performance']:
            best_hour = metrics['best_hour']
            worst_hour = metrics['worst_hour']
            report_lines.append(f"\n⏰ 時間帯分析:")
            report_lines.append(f"   最良時間: {best_hour}時台")
            report_lines.append(f"   最悪時間: {worst_hour}時台")
        
        # 推奨事項
        report_lines.append(f"\n💡 推奨事項:")
        
        if metrics['win_rate'] < 0.55:
            report_lines.append("   - 信頼度閾値の見直しを検討")
            report_lines.append("   - モデルの再キャリブレーションを検討")
        
        if metrics['avg_confidence'] < 0.60:
            report_lines.append("   - 低信頼度取引の減少を検討")
        
        if metrics['total_pips'] < 0:
            report_lines.append("   - 損切り設定の見直し")
            report_lines.append("   - 市場環境変化の確認")
        
        if not alerts and metrics['win_rate'] > 0.60 and metrics['total_pips'] > 10:
            report_lines.append("   ✅ 良好な性能を維持しています")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def save_daily_performance(self, metrics: dict):
        """日次性能データを保存"""
        try:
            os.makedirs(self.logs_dir, exist_ok=True)
            
            # 既存データ読み込み
            performance_history = []
            if os.path.exists(self.performance_log_path):
                with open(self.performance_log_path, 'r') as f:
                    performance_history = json.load(f)
            
            # 今日のデータを追加/更新
            today_str = metrics['date']
            
            # 既存の今日のデータを削除
            performance_history = [p for p in performance_history if p['date'] != today_str]
            
            # 新しいデータを追加
            performance_history.append(metrics)
            
            # 最新30日分のみ保持
            performance_history = performance_history[-30:]
            
            # 保存
            with open(self.performance_log_path, 'w') as f:
                json.dump(performance_history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"日次性能データ保存完了: {today_str}")
            
        except Exception as e:
            logger.error(f"日次性能データ保存エラー: {e}")
    
    def generate_weekly_report(self, end_date: datetime) -> str:
        """週次レポート生成"""
        start_date = end_date - timedelta(days=6)
        
        try:
            # 過去7日間のデータを読み込み
            if os.path.exists(self.performance_log_path):
                with open(self.performance_log_path, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            # 期間内のデータをフィルタ
            week_data = []
            for day in history:
                day_date = datetime.strptime(day['date'], "%Y-%m-%d")
                if start_date <= day_date <= end_date:
                    week_data.append(day)
            
            if not week_data:
                return "📊 週次レポート: データが不足しています"
            
            # 週次集計
            total_trades = sum(d['total_trades'] for d in week_data)
            total_wins = sum(d['winning_trades'] for d in week_data)
            total_pips = sum(d['total_pips'] for d in week_data)
            
            weekly_win_rate = total_wins / total_trades if total_trades > 0 else 0
            avg_daily_trades = total_trades / len(week_data)
            avg_daily_pips = total_pips / len(week_data)
            
            # レポート生成
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append(f"   週次性能レポート")
            report_lines.append(f"   期間: {start_date.strftime('%Y-%m-%d')} 〜 {end_date.strftime('%Y-%m-%d')}")
            report_lines.append("=" * 60)
            
            report_lines.append(f"\n📊 週次サマリー:")
            report_lines.append(f"   稼働日数: {len(week_data)} 日")
            report_lines.append(f"   総取引数: {total_trades}")
            report_lines.append(f"   週次勝率: {weekly_win_rate:.1%}")
            report_lines.append(f"   週次損益: {total_pips:+.1f}pips")
            report_lines.append(f"   日平均取引: {avg_daily_trades:.1f}")
            report_lines.append(f"   日平均損益: {avg_daily_pips:+.1f}pips")
            
            # 日別詳細
            report_lines.append(f"\n📅 日別詳細:")
            for day in week_data:
                day_date = datetime.strptime(day['date'], "%Y-%m-%d")
                weekday = ['月', '火', '水', '木', '金', '土', '日'][day_date.weekday()]
                
                report_lines.append(f"   {day['date']}({weekday}): "
                                  f"{day['total_trades']}取引, "
                                  f"勝率{day['win_rate']:.1%}, "
                                  f"{day['total_pips']:+.1f}pips")
            
            # トレンド分析
            if len(week_data) >= 3:
                recent_3days = week_data[-3:]
                recent_win_rate = sum(d['winning_trades'] for d in recent_3days) / sum(d['total_trades'] for d in recent_3days)
                recent_pips = sum(d['total_pips'] for d in recent_3days)
                
                report_lines.append(f"\n📈 直近3日トレンド:")
                report_lines.append(f"   勝率: {recent_win_rate:.1%}")
                report_lines.append(f"   損益: {recent_pips:+.1f}pips")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"❌ 週次レポート生成エラー: {e}"
    
    def monitor_daily_performance(self, target_date: datetime) -> dict:
        """日次性能監視実行"""
        logger.info(f"日次性能監視開始: {target_date.strftime('%Y-%m-%d')}")
        
        # 取引データ取得（実際はMT5から）
        daily_data = self.simulate_daily_trading_data(target_date)
        
        # 指標計算
        metrics = self.calculate_daily_metrics(daily_data)
        
        if 'error' in metrics:
            logger.error(f"指標計算エラー: {metrics['error']}")
            return metrics
        
        # アラートチェック
        alerts = self.check_alerts(metrics)
        
        # レポート生成
        report = self.generate_daily_report(metrics, alerts)
        
        # データ保存
        self.save_daily_performance(metrics)
        
        # レポート表示
        print(report)
        
        # ファイル保存
        report_file = f"logs/daily_report_{target_date.strftime('%Y%m%d')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"日次レポート保存: {report_file}")
        
        return {
            'metrics': metrics,
            'alerts': alerts,
            'report_file': report_file,
            'alert_count': len(alerts)
        }

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="日次性能監視システム")
    
    parser.add_argument("--date",
                       default="today",
                       help="監視日付 (YYYY-MM-DD形式, または 'today')")
    
    parser.add_argument("--weekly_report",
                       action="store_true",
                       help="週次レポート生成")
    
    parser.add_argument("--alert_only",
                       action="store_true", 
                       help="アラートがある場合のみ表示")
    
    args = parser.parse_args()
    
    monitor = DailyPerformanceMonitor()
    
    # 日付解析
    if args.date == "today":
        target_date = datetime.now()
    else:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print("❌ 日付形式が正しくありません (YYYY-MM-DD)")
            return 1
    
    # 週次レポート
    if args.weekly_report:
        weekly_report = monitor.generate_weekly_report(target_date)
        print(weekly_report)
        
        # 週次レポート保存
        report_file = f"logs/weekly_report_{target_date.strftime('%Y%m%d')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(weekly_report)
        
        print(f"\n📄 週次レポート保存: {report_file}")
        return 0
    
    # 日次監視実行
    result = monitor.monitor_daily_performance(target_date)
    
    if 'error' in result:
        print(f"❌ 監視エラー: {result['error']}")
        return 1
    
    # アラート専用モード
    if args.alert_only:
        if result['alert_count'] > 0:
            print(f"\n🚨 {result['alert_count']}件のアラートが発生しています")
            return 1
        else:
            print("✅ アラートはありません")
            return 0
    
    # 通常モード
    if result['alert_count'] > 0:
        print(f"\n⚠️ {result['alert_count']}件のアラートが発生しました")
        return 1
    else:
        print("\n✅ 正常な稼働状況です")
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