"""
USDJPY スキャルピングEA用 モデル評価
実トレード指標、リスク分析、詳細レポート生成
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.calibration import calibration_curve
import json
from datetime import datetime
from typing import Dict, List, Tuple
import os

# 自作モジュール
from utils import USDJPYUtils
from model import ScalpingCNNLSTM

class ScalpingEvaluator:
    """スキャルピングモデル評価クラス"""
    
    def __init__(self, 
                 model: ScalpingCNNLSTM,
                 profit_pips: float = 8.0,
                 loss_pips: float = 4.0,
                 spread_pips: float = 0.7):
        """
        Args:
            model: 学習済みモデル
            profit_pips: 利確pips
            loss_pips: 損切りpips  
            spread_pips: スプレッドpips
        """
        self.model = model
        self.profit_pips = profit_pips
        self.loss_pips = loss_pips
        self.spread_pips = spread_pips
        self.utils = USDJPYUtils()
        
        print(f"評価器初期化: 利確{profit_pips}pips, 損切{loss_pips}pips, スプレッド{spread_pips}pips")
    
    def comprehensive_evaluation(self, 
                                X_test: np.array, 
                                y_test: np.array,
                                test_timestamps: pd.Index = None) -> Dict:
        """
        包括的モデル評価
        Args:
            X_test: テスト用特徴量
            y_test: テスト用ラベル（raw）
            test_timestamps: テストデータのタイムスタンプ
        Returns:
            dict: 評価結果
        """
        print("=== 包括的モデル評価開始 ===")
        
        # 予測実行
        pred_proba, pred_class = self.model.predict(X_test)
        
        # 基本分類指標
        classification_metrics = self._calculate_classification_metrics(y_test, pred_class, pred_proba)
        
        # 実トレード指標
        trading_metrics = self._calculate_detailed_trading_metrics(y_test, pred_class, pred_proba)
        
        # リスク分析
        risk_metrics = self._calculate_risk_metrics(y_test, pred_class, pred_proba)
        
        # 時系列分析（タイムスタンプがある場合）
        temporal_metrics = {}
        if test_timestamps is not None:
            temporal_metrics = self._calculate_temporal_metrics(
                y_test, pred_class, pred_proba, test_timestamps
            )
        
        # 信頼度分析
        confidence_metrics = self._calculate_confidence_metrics(y_test, pred_class, pred_proba)
        
        # 結果統合
        evaluation_results = {
            'test_samples': len(X_test),
            'classification_metrics': classification_metrics,
            'trading_metrics': trading_metrics,
            'risk_metrics': risk_metrics,
            'temporal_metrics': temporal_metrics,
            'confidence_metrics': confidence_metrics,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # サマリー表示
        self._print_evaluation_summary(evaluation_results)
        
        return evaluation_results
    
    def _calculate_classification_metrics(self, y_true: np.array, y_pred: np.array, pred_proba: np.array) -> Dict:
        """分類性能指標計算"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
        
        # 基本指標
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # クラス別指標
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # 加重平均
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # 混同行列
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': classification_report(
                y_true, y_pred, target_names=['NO_TRADE', 'BUY', 'SELL']
            )
        }
    
    def _calculate_detailed_trading_metrics(self, y_true: np.array, y_pred: np.array, pred_proba: np.array) -> Dict:
        """詳細な実トレード指標計算"""
        
        # シグナル分析
        buy_signals = (y_pred == 1)
        sell_signals = (y_pred == 2)
        no_trade_signals = (y_pred == 0)
        
        metrics = {
            'total_signals': len(y_pred),
            'buy_signals': buy_signals.sum(),
            'sell_signals': sell_signals.sum(),
            'no_trade_signals': no_trade_signals.sum()
        }
        
        # BUY トレード分析
        if metrics['buy_signals'] > 0:
            buy_correct = (y_true[buy_signals] == 1).sum()
            buy_wrong_notrade = (y_true[buy_signals] == 0).sum()
            buy_wrong_sell = (y_true[buy_signals] == 2).sum()
            
            metrics['buy_win_rate'] = buy_correct / metrics['buy_signals']
            metrics['buy_correct'] = buy_correct
            metrics['buy_wrong_notrade'] = buy_wrong_notrade
            metrics['buy_wrong_sell'] = buy_wrong_sell
            
            # BUY利益計算
            buy_profit = (buy_correct * self.profit_pips - 
                         (buy_wrong_notrade + buy_wrong_sell) * self.loss_pips)
            metrics['buy_total_profit'] = buy_profit
            metrics['buy_profit_per_trade'] = buy_profit / metrics['buy_signals']
        else:
            metrics.update({
                'buy_win_rate': 0, 'buy_correct': 0, 'buy_wrong_notrade': 0,
                'buy_wrong_sell': 0, 'buy_total_profit': 0, 'buy_profit_per_trade': 0
            })
        
        # SELL トレード分析
        if metrics['sell_signals'] > 0:
            sell_correct = (y_true[sell_signals] == 2).sum()
            sell_wrong_notrade = (y_true[sell_signals] == 0).sum()
            sell_wrong_buy = (y_true[sell_signals] == 1).sum()
            
            metrics['sell_win_rate'] = sell_correct / metrics['sell_signals']
            metrics['sell_correct'] = sell_correct
            metrics['sell_wrong_notrade'] = sell_wrong_notrade
            metrics['sell_wrong_buy'] = sell_wrong_buy
            
            # SELL利益計算
            sell_profit = (sell_correct * self.profit_pips - 
                          (sell_wrong_notrade + sell_wrong_buy) * self.loss_pips)
            metrics['sell_total_profit'] = sell_profit
            metrics['sell_profit_per_trade'] = sell_profit / metrics['sell_signals']
        else:
            metrics.update({
                'sell_win_rate': 0, 'sell_correct': 0, 'sell_wrong_notrade': 0,
                'sell_wrong_buy': 0, 'sell_total_profit': 0, 'sell_profit_per_trade': 0
            })
        
        # 総合トレード指標
        total_trades = metrics['buy_signals'] + metrics['sell_signals']
        if total_trades > 0:
            total_profit = metrics['buy_total_profit'] + metrics['sell_total_profit']
            metrics['total_profit'] = total_profit
            metrics['profit_per_trade'] = total_profit / total_trades
            metrics['overall_win_rate'] = (metrics['buy_correct'] + metrics['sell_correct']) / total_trades
            
            # 取引頻度
            metrics['trading_frequency'] = total_trades / len(y_pred)
        else:
            metrics.update({
                'total_profit': 0, 'profit_per_trade': 0, 
                'overall_win_rate': 0, 'trading_frequency': 0
            })
        
        # スプレッドコスト考慮
        spread_cost = total_trades * self.spread_pips
        metrics['profit_after_spread'] = metrics['total_profit'] - spread_cost
        metrics['profit_per_trade_after_spread'] = metrics['profit_after_spread'] / total_trades if total_trades > 0 else 0
        
        return metrics
    
    def _calculate_risk_metrics(self, y_true: np.array, y_pred: np.array, pred_proba: np.array) -> Dict:
        """リスク分析指標"""
        
        # 連続損失分析
        trades_mask = (y_pred == 1) | (y_pred == 2)
        if trades_mask.sum() == 0:
            return {'no_trades': True}
        
        trade_results = []
        for i in range(len(y_pred)):
            if trades_mask[i]:
                if y_pred[i] == y_true[i]:  # 正解
                    trade_results.append(self.profit_pips)
                else:  # 不正解
                    trade_results.append(-self.loss_pips)
        
        if not trade_results:
            return {'no_valid_trades': True}
        
        # 累積損益計算
        cumulative_pnl = np.cumsum(trade_results)
        
        # ドローダウン計算
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown)
        
        # 連続負けの最大数
        losing_streaks = []
        current_streak = 0
        for result in trade_results:
            if result < 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    losing_streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            losing_streaks.append(current_streak)
        
        max_losing_streak = max(losing_streaks) if losing_streaks else 0
        
        # 利益の標準偏差（ボラティリティ）
        profit_std = np.std(trade_results)
        
        # シャープレシオ近似
        mean_profit = np.mean(trade_results)
        sharpe_ratio = mean_profit / profit_std if profit_std > 0 else 0
        
        # 勝率別のリスク
        win_rate = np.mean(np.array(trade_results) > 0)
        
        return {
            'max_drawdown': max_drawdown,
            'max_losing_streak': max_losing_streak,
            'profit_volatility': profit_std,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades_analyzed': len(trade_results),
            'cumulative_pnl': cumulative_pnl.tolist()
        }
    
    def _calculate_temporal_metrics(self, y_true: np.array, y_pred: np.array, pred_proba: np.array, timestamps: pd.Index) -> Dict:
        """時系列分析指標"""
        
        # タイムスタンプをDataFrameに変換
        df = pd.DataFrame({
            'timestamp': timestamps,
            'y_true': y_true,
            'y_pred': y_pred,
            'correct': y_true == y_pred
        })
        
        # 時間帯別分析
        df['hour'] = df['timestamp'].dt.hour
        hourly_accuracy = df.groupby('hour')['correct'].mean()
        
        # 曜日別分析
        df['weekday'] = df['timestamp'].dt.weekday
        daily_accuracy = df.groupby('weekday')['correct'].mean()
        
        # 月別分析（データが複数月ある場合）
        df['month'] = df['timestamp'].dt.month
        monthly_accuracy = df.groupby('month')['correct'].mean()
        
        # トレード頻度の時間変動
        trades_mask = (df['y_pred'] == 1) | (df['y_pred'] == 2)
        hourly_trade_freq = df[trades_mask].groupby('hour').size()
        
        return {
            'hourly_accuracy': hourly_accuracy.to_dict(),
            'daily_accuracy': daily_accuracy.to_dict(),
            'monthly_accuracy': monthly_accuracy.to_dict(),
            'hourly_trade_frequency': hourly_trade_freq.to_dict()
        }
    
    def _calculate_confidence_metrics(self, y_true: np.array, y_pred: np.array, pred_proba: np.array) -> Dict:
        """予測信頼度分析"""
        
        # 最大確率（信頼度）
        max_proba = np.max(pred_proba, axis=1)
        
        # 信頼度別の精度
        confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        confidence_analysis = {}
        
        for threshold in confidence_thresholds:
            high_conf_mask = max_proba >= threshold
            if high_conf_mask.sum() > 0:
                accuracy = np.mean(y_true[high_conf_mask] == y_pred[high_conf_mask])
                coverage = high_conf_mask.sum() / len(y_pred)
                
                confidence_analysis[f'threshold_{threshold}'] = {
                    'accuracy': accuracy,
                    'coverage': coverage,
                    'samples': high_conf_mask.sum()
                }
        
        # キャリブレーション分析（各クラス別）
        calibration_results = {}
        for class_idx in range(pred_proba.shape[1]):
            class_name = ['NO_TRADE', 'BUY', 'SELL'][class_idx]
            y_binary = (y_true == class_idx).astype(int)
            
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_binary, pred_proba[:, class_idx], n_bins=10
                )
                calibration_results[class_name] = {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                }
            except:
                calibration_results[class_name] = {'error': 'Unable to calculate calibration'}
        
        return {
            'mean_confidence': np.mean(max_proba),
            'confidence_std': np.std(max_proba),
            'confidence_analysis': confidence_analysis,
            'calibration_results': calibration_results
        }
    
    def _print_evaluation_summary(self, results: Dict):
        """評価結果サマリー表示"""
        print("\n" + "="*60)
        print("           モデル評価サマリー")
        print("="*60)
        
        # 基本指標
        cls_metrics = results['classification_metrics']
        print(f"📊 分類性能:")
        print(f"  精度: {cls_metrics['accuracy']:.3f}")
        print(f"  加重F1スコア: {cls_metrics['f1_weighted']:.3f}")
        print(f"  バランス精度: {cls_metrics['balanced_accuracy']:.3f}")
        
        # トレード指標
        trade_metrics = results['trading_metrics']
        print(f"\n💰 トレード性能:")
        print(f"  総シグナル数: {trade_metrics['buy_signals'] + trade_metrics['sell_signals']:,}")
        print(f"  BUY勝率: {trade_metrics['buy_win_rate']:.1%} ({trade_metrics['buy_signals']}シグナル)")
        print(f"  SELL勝率: {trade_metrics['sell_win_rate']:.1%} ({trade_metrics['sell_signals']}シグナル)")
        print(f"  総利益: {trade_metrics['total_profit']:.1f} pips")
        print(f"  1トレード利益: {trade_metrics['profit_per_trade']:.2f} pips")
        print(f"  スプレッド後利益: {trade_metrics['profit_after_spread']:.1f} pips")
        
        # リスク指標
        risk_metrics = results['risk_metrics']
        if 'max_drawdown' in risk_metrics:
            print(f"\n⚠️  リスク分析:")
            print(f"  最大ドローダウン: {risk_metrics['max_drawdown']:.1f} pips")
            print(f"  最大連続負け: {risk_metrics['max_losing_streak']} 回")
            print(f"  シャープレシオ: {risk_metrics['sharpe_ratio']:.2f}")
            print(f"  勝率: {risk_metrics['win_rate']:.1%}")
        
        # 信頼度分析
        conf_metrics = results['confidence_metrics']
        print(f"\n🎯 信頼度分析:")
        print(f"  平均信頼度: {conf_metrics['mean_confidence']:.3f}")
        
        if 'threshold_0.7' in conf_metrics['confidence_analysis']:
            high_conf = conf_metrics['confidence_analysis']['threshold_0.7']
            print(f"  高信頼度(>0.7)精度: {high_conf['accuracy']:.3f}")
            print(f"  高信頼度カバレッジ: {high_conf['coverage']:.1%}")
        
        print("="*60)
    
    def generate_evaluation_report(self, results: Dict, output_dir: str = "evaluation_results"):
        """評価レポート生成"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON形式で詳細結果保存
        with open(f"{output_dir}/evaluation_results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # テキストレポート生成
        self._generate_text_report(results, f"{output_dir}/evaluation_report_{timestamp}.txt")
        
        # 可視化レポート生成
        self._generate_visualization_report(results, output_dir, timestamp)
        
        print(f"評価レポート生成完了: {output_dir}")
    
    def _generate_text_report(self, results: Dict, filepath: str):
        """テキストレポート生成"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("USDJPY スキャルピングEA モデル評価レポート\n")
            f.write("="*60 + "\n")
            f.write(f"評価日時: {results['evaluation_timestamp']}\n")
            f.write(f"テストサンプル数: {results['test_samples']:,}\n\n")
            
            # 分類性能
            cls_metrics = results['classification_metrics']
            f.write("【分類性能】\n")
            f.write(f"精度: {cls_metrics['accuracy']:.4f}\n")
            f.write(f"加重F1スコア: {cls_metrics['f1_weighted']:.4f}\n")
            f.write(f"バランス精度: {cls_metrics['balanced_accuracy']:.4f}\n\n")
            
            f.write("クラス別詳細:\n")
            classes = ['NO_TRADE', 'BUY', 'SELL']
            for i, class_name in enumerate(classes):
                f.write(f"  {class_name}:\n")
                f.write(f"    適合率: {cls_metrics['precision_per_class'][i]:.4f}\n")
                f.write(f"    再現率: {cls_metrics['recall_per_class'][i]:.4f}\n")
                f.write(f"    F1スコア: {cls_metrics['f1_per_class'][i]:.4f}\n")
                f.write(f"    サポート: {cls_metrics['support_per_class'][i]}\n\n")
            
            # トレード性能
            trade_metrics = results['trading_metrics']
            f.write("【トレード性能】\n")
            f.write(f"BUYシグナル: {trade_metrics['buy_signals']} (勝率: {trade_metrics['buy_win_rate']:.1%})\n")
            f.write(f"SELLシグナル: {trade_metrics['sell_signals']} (勝率: {trade_metrics['sell_win_rate']:.1%})\n")
            f.write(f"総利益: {trade_metrics['total_profit']:.2f} pips\n")
            f.write(f"1トレード当たり利益: {trade_metrics['profit_per_trade']:.3f} pips\n")
            f.write(f"スプレッド後利益: {trade_metrics['profit_after_spread']:.2f} pips\n")
            f.write(f"取引頻度: {trade_metrics['trading_frequency']:.1%}\n\n")
            
            # リスク分析
            risk_metrics = results['risk_metrics']
            if 'max_drawdown' in risk_metrics:
                f.write("【リスク分析】\n")
                f.write(f"最大ドローダウン: {risk_metrics['max_drawdown']:.2f} pips\n")
                f.write(f"最大連続負け: {risk_metrics['max_losing_streak']} 回\n")
                f.write(f"利益ボラティリティ: {risk_metrics['profit_volatility']:.3f}\n")
                f.write(f"シャープレシオ: {risk_metrics['sharpe_ratio']:.3f}\n")
                f.write(f"勝率: {risk_metrics['win_rate']:.1%}\n\n")
            
            # 時系列分析
            if results['temporal_metrics']:
                f.write("【時系列分析】\n")
                temporal = results['temporal_metrics']
                
                if 'hourly_accuracy' in temporal:
                    f.write("時間帯別精度:\n")
                    for hour, acc in temporal['hourly_accuracy'].items():
                        f.write(f"  {hour:2d}時: {acc:.3f}\n")
                    f.write("\n")
    
    def _generate_visualization_report(self, results: Dict, output_dir: str, timestamp: str):
        """可視化レポート生成"""
        try:
            # 混同行列
            self._plot_confusion_matrix(results['classification_metrics'], output_dir, timestamp)
            
            # 累積損益グラフ
            if 'cumulative_pnl' in results['risk_metrics']:
                self._plot_cumulative_pnl(results['risk_metrics'], output_dir, timestamp)
            
            # 時間帯別精度
            if results['temporal_metrics'] and 'hourly_accuracy' in results['temporal_metrics']:
                self._plot_hourly_performance(results['temporal_metrics'], output_dir, timestamp)
            
            # 信頼度分析
            self._plot_confidence_analysis(results['confidence_metrics'], output_dir, timestamp)
            
        except Exception as e:
            print(f"可視化生成エラー: {e}")
    
    def _plot_confusion_matrix(self, cls_metrics: Dict, output_dir: str, timestamp: str):
        """混同行列プロット"""
        plt.figure(figsize=(8, 6))
        conf_matrix = np.array(cls_metrics['confusion_matrix'])
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['NO_TRADE', 'BUY', 'SELL'],
                   yticklabels=['NO_TRADE', 'BUY', 'SELL'])
        plt.title('混同行列')
        plt.ylabel('実際のラベル')
        plt.xlabel('予測ラベル')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_{timestamp}.png", dpi=300)
        plt.close()
    
    def _plot_cumulative_pnl(self, risk_metrics: Dict, output_dir: str, timestamp: str):
        """累積損益プロット"""
        plt.figure(figsize=(12, 6))
        cumulative_pnl = risk_metrics['cumulative_pnl']
        
        plt.plot(cumulative_pnl, linewidth=2)
        plt.title('累積損益曲線')
        plt.xlabel('トレード回数')
        plt.ylabel('累積損益 (pips)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 最大ドローダウンの表示
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_dd_idx = np.argmax(drawdown)
        
        plt.fill_between(range(len(cumulative_pnl)), running_max, cumulative_pnl, 
                        alpha=0.3, color='red', label=f'ドローダウン (最大: {np.max(drawdown):.1f}pips)')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cumulative_pnl_{timestamp}.png", dpi=300)
        plt.close()
    
    def _plot_hourly_performance(self, temporal_metrics: Dict, output_dir: str, timestamp: str):
        """時間帯別パフォーマンス"""
        plt.figure(figsize=(12, 6))
        
        hourly_acc = temporal_metrics['hourly_accuracy']
        hours = sorted(hourly_acc.keys())
        accuracies = [hourly_acc[h] for h in hours]
        
        plt.bar(hours, accuracies, alpha=0.7)
        plt.title('時間帯別予測精度')
        plt.xlabel('時刻')
        plt.ylabel('精度')
        plt.xticks(hours)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=np.mean(accuracies), color='r', linestyle='--', 
                   label=f'平均精度: {np.mean(accuracies):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hourly_performance_{timestamp}.png", dpi=300)
        plt.close()
    
    def _plot_confidence_analysis(self, conf_metrics: Dict, output_dir: str, timestamp: str):
        """信頼度分析プロット"""
        plt.figure(figsize=(10, 6))
        
        conf_analysis = conf_metrics['confidence_analysis']
        thresholds = []
        accuracies = []
        coverages = []
        
        for key, values in conf_analysis.items():
            threshold = float(key.split('_')[1])
            thresholds.append(threshold)
            accuracies.append(values['accuracy'])
            coverages.append(values['coverage'])
        
        if thresholds:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 精度 vs 信頼度閾値
            ax1.plot(thresholds, accuracies, 'o-', linewidth=2, markersize=6)
            ax1.set_xlabel('信頼度閾値')
            ax1.set_ylabel('精度')
            ax1.set_title('信頼度閾値 vs 精度')
            ax1.grid(True, alpha=0.3)
            
            # カバレッジ vs 信頼度閾値
            ax2.plot(thresholds, coverages, 'o-', color='orange', linewidth=2, markersize=6)
            ax2.set_xlabel('信頼度閾値')
            ax2.set_ylabel('カバレッジ')
            ax2.set_title('信頼度閾値 vs カバレッジ')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/confidence_analysis_{timestamp}.png", dpi=300)
        
        plt.close()


def evaluate_model_from_file(model_path: str, 
                           test_data_path: str,
                           output_dir: str = "evaluation_results") -> Dict:
    """
    保存されたモデルの評価実行
    Args:
        model_path: モデルファイルパス
        test_data_path: テストデータパス
        output_dir: 出力ディレクトリ
    Returns:
        dict: 評価結果
    """
    print("=== 保存モデル評価実行 ===")
    
    # モデル読み込み
    model = ScalpingCNNLSTM()
    model.load_model(model_path)
    
    # 評価器初期化
    evaluator = ScalpingEvaluator(model)
    
    # テストデータ準備（実装は使用ケースに応じて調整）
    # この部分は実際のデータ形式に合わせて実装する必要があります
    
    print("評価用データ準備...")
    # X_test, y_test, timestamps = prepare_test_data(test_data_path)
    
    # 評価実行
    # results = evaluator.comprehensive_evaluation(X_test, y_test, timestamps)
    
    # レポート生成
    # evaluator.generate_evaluation_report(results, output_dir)
    
    print("モデル評価完了")
    # return results


if __name__ == "__main__":
    # 評価テスト
    print("=== モデル評価システム テスト ===")
    
    try:
        # ダミーデータでテスト
        n_samples = 1000
        n_classes = 3
        
        # ダミー予測結果生成
        y_true = np.random.randint(0, n_classes, n_samples)
        y_pred = np.random.randint(0, n_classes, n_samples)
        pred_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
        
        # ダミーモデル作成
        from model import ScalpingCNNLSTM
        dummy_model = ScalpingCNNLSTM()
        
        # 評価器テスト
        evaluator = ScalpingEvaluator(dummy_model)
        
        # 包括的評価テスト
        results = evaluator.comprehensive_evaluation(
            X_test=np.random.randn(n_samples, 30, 87),  # ダミー特徴量
            y_test=y_true
        )
        
        print("評価システムテスト完了")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()