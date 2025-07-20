"""
信頼度キャリブレーション可視化システム
ChatGPT質問への回答: accuracy vs confidence の詳細分析と可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import pandas as pd
from typing import Dict, List, Tuple
import os

class CalibrationVisualizationSystem:
    """キャリブレーション可視化システム"""
    
    def __init__(self, style='scientific'):
        """
        Args:
            style: 'scientific', 'business', 'presentation'
        """
        self.style = style
        self._setup_plotting_style()
        
    def _setup_plotting_style(self):
        """プロット스タイル設定"""
        if self.style == 'scientific':
            plt.style.use('seaborn-v0_8-whitegrid')
        elif self.style == 'business':
            plt.style.use('seaborn-v0_8-darkgrid')
        else:
            plt.style.use('default')
        
        # 日本語フォント設定（オプション）
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.dpi'] = 100
    
    def comprehensive_calibration_analysis(self, y_true: np.array, 
                                         y_proba: np.array,
                                         model_name: str = "ScalpingModel") -> Dict:
        """
        包括的キャリブレーション分析
        
        Args:
            y_true: 実際のラベル (0/1)
            y_proba: 予測確率
            model_name: モデル名
        
        Returns:
            分析結果辞書
        """
        
        print(f"📊 {model_name} キャリブレーション分析開始")
        
        # メトリクス計算
        metrics = self._calculate_calibration_metrics(y_true, y_proba)
        
        # 可視化作成
        fig = plt.figure(figsize=(20, 12))
        
        # 1. キャリブレーション曲線
        ax1 = plt.subplot(2, 4, 1)
        self._plot_calibration_curve(y_true, y_proba, ax1)
        
        # 2. 信頼度ヒストグラム
        ax2 = plt.subplot(2, 4, 2)
        self._plot_confidence_histogram(y_proba, ax2)
        
        # 3. 信頼度 vs 精度散布図
        ax3 = plt.subplot(2, 4, 3)
        self._plot_confidence_accuracy_scatter(y_true, y_proba, ax3)
        
        # 4. ビン別信頼度分析
        ax4 = plt.subplot(2, 4, 4)
        self._plot_binned_confidence_analysis(y_true, y_proba, ax4)
        
        # 5. 予測確率分布（クラス別）
        ax5 = plt.subplot(2, 4, 5)
        self._plot_prediction_distribution_by_class(y_true, y_proba, ax5)
        
        # 6. ECE/MCE メトリクス
        ax6 = plt.subplot(2, 4, 6)
        self._plot_ece_mce_breakdown(y_true, y_proba, ax6)
        
        # 7. 信頼度 vs パフォーマンス関係
        ax7 = plt.subplot(2, 4, 7)
        self._plot_confidence_performance_curve(y_true, y_proba, ax7)
        
        # 8. メトリクスサマリー
        ax8 = plt.subplot(2, 4, 8)
        self._plot_metrics_summary(metrics, ax8)
        
        # メインタイトルも明示的にフォント指定
        plt.suptitle(f'{model_name} - 包括的キャリブレーション分析', fontsize=16, fontfamily='Meiryo')
        plt.tight_layout()
        plt.savefig(f'{model_name}_calibration_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 問題診断
        diagnosis = self._diagnose_calibration_issues(metrics)
        
        return {
            'metrics': metrics,
            'diagnosis': diagnosis,
            'recommendations': self._generate_recommendations(diagnosis)
        }
    
    def _calculate_calibration_metrics(self, y_true: np.array, y_proba: np.array) -> Dict:
        """キャリブレーションメトリクス計算"""
        
        # 基本メトリクス
        brier_score = brier_score_loss(y_true, y_proba)
        
        # キャリブレーション曲線
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10, strategy='uniform'
        )
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(y_true, y_proba, n_bins=10)
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(y_true, y_proba, n_bins=10)
        
        # Average Confidence
        avg_confidence = np.mean(y_proba)
        
        # Accuracy
        y_pred = (y_proba >= 0.5).astype(int)
        accuracy = np.mean(y_true == y_pred)
        
        # Confidence-Accuracy 相関
        confidence_accuracy_corr = np.corrcoef(y_proba, y_true)[0, 1]
        
        # Over/Under confidence分析
        overconfidence = self._calculate_overconfidence(y_true, y_proba)
        
        return {
            'brier_score': brier_score,
            'ece': ece,
            'mce': mce,
            'avg_confidence': avg_confidence,
            'accuracy': accuracy,
            'confidence_accuracy_correlation': confidence_accuracy_corr,
            'overconfidence': overconfidence,
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value
            }
        }
    
    def _calculate_ece(self, y_true: np.array, y_proba: np.array, n_bins: int = 10) -> float:
        """Expected Calibration Error計算"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.astype(float).mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].astype(float).mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(self, y_true: np.array, y_proba: np.array, n_bins: int = 10) -> float:
        """Maximum Calibration Error計算"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                error = abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error
    
    def _calculate_overconfidence(self, y_true: np.array, y_proba: np.array) -> Dict:
        """過信度分析"""
        
        # 高信頼度領域での分析
        high_conf_mask = y_proba >= 0.8
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = y_true[high_conf_mask].mean()
            high_conf_avg_prediction = y_proba[high_conf_mask].mean()
            high_conf_overconfidence = high_conf_avg_prediction - high_conf_accuracy
        else:
            high_conf_overconfidence = 0
        
        # 中信頼度領域での分析
        mid_conf_mask = (y_proba >= 0.5) & (y_proba < 0.8)
        if mid_conf_mask.sum() > 0:
            mid_conf_accuracy = y_true[mid_conf_mask].mean()
            mid_conf_avg_prediction = y_proba[mid_conf_mask].mean()
            mid_conf_overconfidence = mid_conf_avg_prediction - mid_conf_accuracy
        else:
            mid_conf_overconfidence = 0
        
        return {
            'high_confidence_region': high_conf_overconfidence,
            'mid_confidence_region': mid_conf_overconfidence,
            'overall': np.mean(y_proba) - np.mean(y_true)
        }
    
    def _plot_calibration_curve(self, y_true: np.array, y_proba: np.array, ax):
        """キャリブレーション曲線プロット"""
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10
        )
        
        # 完璧なキャリブレーション線
        ax.plot([0, 1], [0, 1], 'k--', label='完璧なキャリブレーション', alpha=0.5)
        
        # 実際のキャリブレーション
        ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                linewidth=2, markersize=8, label='モデルキャリブレーション')
        
        # 信頼区間（簡易版）
        ax.fill_between(mean_predicted_value, 
                       fraction_of_positives - 0.05, 
                       fraction_of_positives + 0.05, 
                       alpha=0.2)
        
        # 明示的なフォント指定
        ax.set_xlabel('平均予測確率', fontfamily='Meiryo')
        ax.set_ylabel('実際の陽性率', fontfamily='Meiryo')
        ax.set_title('キャリブレーション曲線', fontfamily='Meiryo')
        
        # 凡例のフォント指定
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_fontfamily('Meiryo')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_confidence_histogram(self, y_proba: np.array, ax):
        """信頼度ヒストグラム"""
        
        ax.hist(y_proba, bins=20, alpha=0.7, density=True, edgecolor='black')
        ax.axvline(np.mean(y_proba), color='red', linestyle='--', 
                  label=f'平均: {np.mean(y_proba):.3f}')
        ax.axvline(np.median(y_proba), color='orange', linestyle='--', 
                  label=f'中央値: {np.median(y_proba):.3f}')
        
        # 明示的なフォント指定
        ax.set_xlabel('予測確率', fontfamily='Meiryo')
        ax.set_ylabel('密度', fontfamily='Meiryo')
        ax.set_title('信頼度分布', fontfamily='Meiryo')
        
        # 凡例のフォント指定
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_fontfamily('Meiryo')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_confidence_accuracy_scatter(self, y_true: np.array, y_proba: np.array, ax):
        """信頼度 vs 精度散布図"""
        
        # ビン化して散布図作成
        bins = np.linspace(0, 1, 21)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_count = mask.sum()
            else:
                bin_accuracy = 0
                bin_count = 0
            
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(bin_count)
        
        # サンプル数に比例したサイズで散布図
        sizes = [max(20, count / 5) for count in bin_counts]
        scatter = ax.scatter(bin_centers, bin_accuracies, s=sizes, alpha=0.6, c=bin_counts, cmap='viridis')
        
        # 完璧な線
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='完璧なキャリブレーション')
        
        ax.set_xlabel('信頼度レベル', fontfamily='Meiryo')
        ax.set_ylabel('実際の精度', fontfamily='Meiryo')
        ax.set_title('信頼度 vs 精度', fontfamily='Meiryo')
        plt.colorbar(scatter, ax=ax, label='サンプル数')
        
        # 凡例のフォント指定
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_fontfamily('Meiryo')
            
        ax.grid(True, alpha=0.3)
    
    def _plot_binned_confidence_analysis(self, y_true: np.array, y_proba: np.array, ax):
        """ビン別信頼度分析"""
        
        # 10ビンでの分析
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        bin_data = []
        for i in range(n_bins):
            mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
            if mask.sum() > 0:
                bin_confidence = y_proba[mask].mean()
                bin_accuracy = y_true[mask].mean()
                bin_count = mask.sum()
                calibration_error = abs(bin_confidence - bin_accuracy)
            else:
                bin_confidence = bin_centers[i]
                bin_accuracy = 0
                bin_count = 0
                calibration_error = 0
            
            bin_data.append({
                'bin_center': bin_centers[i],
                'confidence': bin_confidence,
                'accuracy': bin_accuracy,
                'count': bin_count,
                'error': calibration_error
            })
        
        # エラーバープロット
        x_pos = range(n_bins)
        confidences = [d['confidence'] for d in bin_data]
        accuracies = [d['accuracy'] for d in bin_data]
        errors = [d['error'] for d in bin_data]
        
        bars = ax.bar(x_pos, errors, alpha=0.7, color=['red' if e > 0.1 else 'orange' if e > 0.05 else 'green' for e in errors])
        
        ax.set_xlabel('信頼度ビン', fontfamily='Meiryo')
        ax.set_ylabel('キャリブレーションエラー', fontfamily='Meiryo')
        ax.set_title('ビン別キャリブレーションエラー', fontfamily='Meiryo')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{d["bin_center"]:.1f}' for d in bin_data], rotation=45)
        
        # 色の説明
        ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='警告 (0.05)')
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='重大 (0.10)')
        
        # 凡例のフォント指定
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_fontfamily('Meiryo')
            
        ax.grid(True, alpha=0.3)
    
    def _plot_prediction_distribution_by_class(self, y_true: np.array, y_proba: np.array, ax):
        """クラス別予測確率分布"""
        
        # クラス0（実際にNegative）
        negative_probs = y_proba[y_true == 0]
        # クラス1（実際にPositive）
        positive_probs = y_proba[y_true == 1]
        
        ax.hist(negative_probs, bins=20, alpha=0.5, label=f'実際にネガティブ (n={len(negative_probs)})', 
               color='red', density=True)
        ax.hist(positive_probs, bins=20, alpha=0.5, label=f'実際にポジティブ (n={len(positive_probs)})', 
               color='blue', density=True)
        
        ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='決定閾値')
        
        ax.set_xlabel('予測確率', fontfamily='Meiryo')
        ax.set_ylabel('密度', fontfamily='Meiryo')
        ax.set_title('真のクラス別予測分布', fontfamily='Meiryo')
        
        # 凡例のフォント指定
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_fontfamily('Meiryo')
            
        ax.grid(True, alpha=0.3)
    
    def _plot_ece_mce_breakdown(self, y_true: np.array, y_proba: np.array, ax):
        """ECE/MCE詳細分析"""
        
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        
        bin_errors = []
        bin_weights = []
        
        for i in range(n_bins):
            mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
            
            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_proba[mask].mean()
                bin_weight = mask.sum() / len(y_proba)
                bin_error = abs(bin_confidence - bin_accuracy)
            else:
                bin_error = 0
                bin_weight = 0
            
            bin_errors.append(bin_error)
            bin_weights.append(bin_weight)
        
        # 加重エラー（ECEへの寄与）
        weighted_errors = [e * w for e, w in zip(bin_errors, bin_weights)]
        
        x_pos = range(n_bins)
        
        # エラーバー
        bars1 = ax.bar(x_pos, bin_errors, alpha=0.5, label='Bin Error', color='lightcoral')
        bars2 = ax.bar(x_pos, weighted_errors, alpha=0.8, label='Weighted Error (ECE contribution)', color='darkred')
        
        ax.set_xlabel('Confidence Bin')
        ax.set_ylabel('Calibration Error')
        ax.set_title('ECE/MCE Breakdown by Bin')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ECE/MCE値を表示
        ece = sum(weighted_errors)
        mce = max(bin_errors)
        ax.text(0.02, 0.98, f'ECE: {ece:.4f}\nMCE: {mce:.4f}', 
               transform=ax.transAxes, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_confidence_performance_curve(self, y_true: np.array, y_proba: np.array, ax):
        """信頼度 vs パフォーマンス曲線"""
        
        thresholds = np.linspace(0.5, 0.95, 20)
        performances = []
        sample_counts = []
        
        for threshold in thresholds:
            mask = y_proba >= threshold
            
            if mask.sum() > 0:
                accuracy = y_true[mask].mean()
                count = mask.sum()
            else:
                accuracy = 0
                count = 0
            
            performances.append(accuracy)
            sample_counts.append(count)
        
        # メインプロット: 信頼度 vs 精度
        ax.plot(thresholds, performances, 'o-', linewidth=2, markersize=6, color='blue', label='Accuracy')
        
        # セカンダリY軸: サンプル数
        ax2 = ax.twinx()
        ax2.bar(thresholds, sample_counts, alpha=0.3, width=0.01, color='orange', label='Sample Count')
        
        ax.set_xlabel('信頼度閾値', fontfamily='Meiryo')
        ax.set_ylabel('精度', color='blue')
        ax2.set_ylabel('サンプル数', color='orange')
        ax.set_title('パフォーマンス vs 信頼度閾値', fontfamily='Meiryo')
        
        # 0.58, 0.59の特別マーキング（ChatGPTの問題）
        if 0.58 in thresholds:
            idx_58 = np.argmin(np.abs(thresholds - 0.58))
            ax.scatter(0.58, performances[idx_58], s=100, color='red', marker='o', zorder=5, label='問題点 0.58')
        
        if 0.59 in thresholds:
            idx_59 = np.argmin(np.abs(thresholds - 0.59))
            ax.scatter(0.59, performances[idx_59], s=100, color='darkred', marker='s', zorder=5, label='問題点 0.59')
        
        # 凡例のフォント指定
        legend1 = ax.legend(loc='upper left')
        legend2 = ax2.legend(loc='upper right')
        for text in legend1.get_texts():
            text.set_fontfamily('Meiryo')
        for text in legend2.get_texts():
            text.set_fontfamily('Meiryo')
            
        ax.grid(True, alpha=0.3)
    
    def _plot_metrics_summary(self, metrics: Dict, ax):
        """メトリクスサマリー表示"""
        
        ax.axis('off')
        
        # メトリクスのテキスト表示
        metrics_text = f"""
📊 キャリブレーションメトリクス

🎯 基本指標:
  • 精度: {metrics['accuracy']:.3f}
  • ブライアスコア: {metrics['brier_score']:.4f}
  • 平均信頼度: {metrics['avg_confidence']:.3f}

📈 キャリブレーション:
  • ECE: {metrics['ece']:.4f}
  • MCE: {metrics['mce']:.4f}
  • 信頼度-精度相関: {metrics['confidence_accuracy_correlation']:.3f}

⚠️ 過信分析:
  • 高信頼度域: {metrics['overconfidence']['high_confidence_region']:+.3f}
  • 中信頼度域: {metrics['overconfidence']['mid_confidence_region']:+.3f}
  • 全体: {metrics['overconfidence']['overall']:+.3f}
        """
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    def _diagnose_calibration_issues(self, metrics: Dict) -> Dict:
        """キャリブレーション問題の診断"""
        
        issues = []
        severity = "normal"
        
        # ECE分析
        if metrics['ece'] > 0.1:
            issues.append("深刻なキャリブレーション不良 (ECE > 0.1)")
            severity = "critical"
        elif metrics['ece'] > 0.05:
            issues.append("中程度のキャリブレーション不良 (ECE > 0.05)")
            severity = "warning"
        
        # 過信分析
        high_overconf = metrics['overconfidence']['high_confidence_region']
        if high_overconf > 0.15:
            issues.append("高信頼度域での深刻な過信")
            severity = "critical"
        elif high_overconf > 0.1:
            issues.append("高信頼度域での過信傾向")
            if severity == "normal":
                severity = "warning"
        
        # 相関分析
        if metrics['confidence_accuracy_correlation'] < 0.5:
            issues.append("信頼度と精度の低相関")
            if severity == "normal":
                severity = "warning"
        
        return {
            'issues': issues,
            'severity': severity,
            'primary_issue': issues[0] if issues else "問題なし"
        }
    
    def _generate_recommendations(self, diagnosis: Dict) -> List[str]:
        """診断結果に基づく推奨事項生成"""
        
        recommendations = []
        
        if diagnosis['severity'] == "critical":
            recommendations.extend([
                "🚨 緊急対応必要: モデル再学習を強く推奨",
                "🔧 温度スケーリング + Platt Scaling の即座実装",
                "📊 キャリブレーション重視損失関数への変更"
            ])
        
        if "キャリブレーション不良" in str(diagnosis['issues']):
            recommendations.extend([
                "🌡️ 温度スケーリングによる校正実装",
                "📈 Isotonic Regression の追加検討",
                "🎯 キャリブレーション専用データセットでの再訓練"
            ])
        
        if "過信" in str(diagnosis['issues']):
            recommendations.extend([
                "🔄 データ拡張による過信防止",
                "⚖️ ラベルスムージング技法の適用",
                "📉 ドロップアウト率の増加検討"
            ])
        
        if "低相関" in str(diagnosis['issues']):
            recommendations.extend([
                "🏗️ モデルアーキテクチャの見直し",
                "🎲 不確実性推定機能の追加",
                "📏 レンジベース閾値システムの導入"
            ])
        
        # 一般的推奨事項
        recommendations.extend([
            "📊 定期的なキャリブレーション監視の実装",
            "🔍 Bootstrap法による安定性検証",
            "📈 複数通貨ペアでの交差検証"
        ])
        
        return recommendations

def create_chatgpt_response_visualization():
    """ChatGPT質問への回答用可視化デモ"""
    
    print("📊 ChatGPT質問への回答: 信頼度キャリブレーション可視化")
    print("=" * 70)
    
    # デモ用のサンプルデータ生成（実際のPhase 4データに置き換え）
    np.random.seed(42)
    n_samples = 5000
    
    # 0.58→0.59問題を再現するサンプルデータ
    confidences = np.random.beta(2, 3, n_samples) * 0.6 + 0.3
    
    # 人工的な急激変化を注入（0.59付近で急激に性能向上）
    performance_boost = np.where(confidences >= 0.59, 
                                np.random.normal(0.15, 0.05, n_samples), 
                                0)
    
    base_accuracy = 0.58 + performance_boost
    y_true = np.random.binomial(1, base_accuracy, n_samples)
    
    print(f"📝 デモデータ特性:")
    print(f"  サンプル数: {n_samples:,}")
    print(f"  信頼度範囲: {confidences.min():.3f} - {confidences.max():.3f}")
    print(f"  実際の精度: {y_true.mean():.3f}")
    
    # 可視化システム初期化
    viz_system = CalibrationVisualizationSystem(style='scientific')
    
    # 包括的分析実行
    analysis_results = viz_system.comprehensive_calibration_analysis(
        y_true, confidences, model_name="Phase4_ScalpingModel"
    )
    
    print(f"\n🎯 分析結果サマリー:")
    print(f"  ECE: {analysis_results['metrics']['ece']:.4f}")
    print(f"  MCE: {analysis_results['metrics']['mce']:.4f}")
    print(f"  信頼度-精度相関: {analysis_results['metrics']['confidence_accuracy_correlation']:.3f}")
    
    print(f"\n⚠️ 検出された問題:")
    for issue in analysis_results['diagnosis']['issues']:
        print(f"  • {issue}")
    
    print(f"\n💡 推奨対策:")
    for i, rec in enumerate(analysis_results['recommendations'][:5], 1):
        print(f"  {i}. {rec}")
    
    return analysis_results

# 推奨キャリブレーションメトリクス
RECOMMENDED_CALIBRATION_METRICS = {
    'primary_metrics': {
        'ECE (Expected Calibration Error)': {
            'formula': 'Σ |confidence - accuracy| * bin_weight',
            'ideal_value': '< 0.05',
            'critical_threshold': '> 0.1',
            'interpretation': '信頼度と実際の精度の乖離度'
        },
        'MCE (Maximum Calibration Error)': {
            'formula': 'max(|confidence - accuracy|) across bins',
            'ideal_value': '< 0.1',
            'critical_threshold': '> 0.2',
            'interpretation': '最悪ケースでの校正エラー'
        }
    },
    
    'secondary_metrics': {
        'Brier Score': {
            'formula': 'Σ (predicted_prob - actual)²',
            'ideal_value': '< 0.15',
            'interpretation': '確率予測の全体的品質'
        },
        'Confidence-Accuracy Correlation': {
            'formula': 'corr(predicted_confidence, actual_accuracy)',
            'ideal_value': '> 0.8',
            'interpretation': '信頼度の信頼性指標'
        }
    },
    
    'visualization_recommendations': {
        'essential_plots': [
            'Calibration Curve (reliability diagram)',
            'Confidence Histogram',
            'Confidence vs Accuracy Scatter',
            'Per-bin Calibration Error'
        ],
        'advanced_plots': [
            'Prediction Distribution by Class',
            'ECE/MCE Breakdown',
            'Confidence-Performance Curve',
            'Bootstrap Confidence Intervals'
        ]
    }
}

def demonstrate_calibration_metrics():
    """キャリブレーションメトリクスのデモンストレーション"""
    
    print("📊 推奨キャリブレーションメトリクス:")
    print("=" * 50)
    
    for category, metrics in RECOMMENDED_CALIBRATION_METRICS.items():
        if category == 'visualization_recommendations':
            continue
            
        print(f"\n{category.upper().replace('_', ' ')}:")
        
        for metric_name, details in metrics.items():
            print(f"\n  {metric_name}:")
            for key, value in details.items():
                print(f"    {key}: {value}")
    
    print(f"\n🎨 推奨可視化:")
    viz_rec = RECOMMENDED_CALIBRATION_METRICS['visualization_recommendations']
    
    print(f"\n  必須プロット:")
    for plot in viz_rec['essential_plots']:
        print(f"    • {plot}")
    
    print(f"\n  高度なプロット:")
    for plot in viz_rec['advanced_plots']:
        print(f"    • {plot}")

if __name__ == "__main__":
    # キャリブレーション可視化デモ実行
    results = create_chatgpt_response_visualization()
    
    # メトリクス説明
    demonstrate_calibration_metrics()
    
    print(f"\n✅ ChatGPT質問への完全回答:")
    print(f"1. 段階的フィルター: ±0.015重複幅、中央重視重み配分")
    print(f"2. データ拡張: 30%ノイズ + 10%シャッフル + 40%スムージング")
    print(f"3. 可視化: ECE/MCE + キャリブレーション曲線 + 8種類の詳細分析")