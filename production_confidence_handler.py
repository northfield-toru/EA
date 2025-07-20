"""
実運用向け信頼度ハンドラー
MT5連携を想定した軽量かつ高速な信頼度処理システム

主な特徴:
1. 3GBメモリ制約でも動作する軽量設計
2. リアルタイム予測での低レイテンシー
3. JSON形式でのMT5連携対応
4. 動的しきい値調整とロットサイズ最適化
"""

import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
import joblib
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConfidenceConfig:
    """信頼度設定"""
    base_threshold: float = 0.58
    optimal_range_lower: float = 0.58
    optimal_range_upper: float = 0.61
    min_confidence_for_trade: float = 0.50
    max_confidence_for_scaling: float = 0.90
    lot_size_base: float = 0.01
    lot_size_max: float = 0.10
    calibration_enabled: bool = True
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'base_threshold': self.base_threshold,
            'optimal_range_lower': self.optimal_range_lower,
            'optimal_range_upper': self.optimal_range_upper,
            'min_confidence_for_trade': self.min_confidence_for_trade,
            'max_confidence_for_scaling': self.max_confidence_for_scaling,
            'lot_size_base': self.lot_size_base,
            'lot_size_max': self.lot_size_max,
            'calibration_enabled': self.calibration_enabled
        }

class LightweightCalibrator:
    """軽量キャリブレーター（実運用向け）"""
    
    def __init__(self, method: str = 'temperature'):
        """
        Args:
            method: 'temperature', 'linear', or 'none'
        """
        self.method = method
        self.temperature = 1.0
        self.slope = 1.0
        self.intercept = 0.0
        self.is_fitted = False
        
    def fit(self, confidences: np.ndarray, accuracies: np.ndarray):
        """
        軽量キャリブレーション学習
        Args:
            confidences: 予測信頼度
            accuracies: 実際の精度（0 or 1）
        """
        if self.method == 'temperature':
            # 温度スケーリング（簡易版）
            def temperature_loss(temp):
                calibrated = 1 / (1 + np.exp(-np.log(confidences / (1 - confidences)) / temp))
                return np.mean((calibrated - accuracies) ** 2)
            
            # 簡易最適化
            best_temp = 1.0
            best_loss = float('inf')
            
            for temp in np.linspace(0.1, 5.0, 50):
                try:
                    loss = temperature_loss(temp)
                    if loss < best_loss:
                        best_loss = loss
                        best_temp = temp
                except:
                    continue
            
            self.temperature = best_temp
            
        elif self.method == 'linear':
            # 線形キャリブレーション
            X = np.column_stack([confidences, np.ones(len(confidences))])
            coeffs = np.linalg.lstsq(X, accuracies, rcond=None)[0]
            self.slope = coeffs[0]
            self.intercept = coeffs[1]
        
        self.is_fitted = True
        logger.info(f"軽量キャリブレーション完了: method={self.method}")
        
    def predict(self, confidences: np.ndarray) -> np.ndarray:
        """
        キャリブレーション適用
        Args:
            confidences: 生の信頼度
        Returns:
            キャリブレーション済み信頼度
        """
        if not self.is_fitted or self.method == 'none':
            return confidences
        
        if self.method == 'temperature':
            # 温度スケーリング適用
            logits = np.log(confidences / (1 - confidences + 1e-8))
            scaled_logits = logits / self.temperature
            return 1 / (1 + np.exp(-scaled_logits))
        
        elif self.method == 'linear':
            # 線形変換
            calibrated = self.slope * confidences + self.intercept
            return np.clip(calibrated, 0.01, 0.99)
        
        return confidences
    
    def save(self, filepath: str):
        """キャリブレーターを保存"""
        calibrator_data = {
            'method': self.method,
            'temperature': self.temperature,
            'slope': self.slope,
            'intercept': self.intercept,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'w') as f:
            json.dump(calibrator_data, f, indent=2)
    
    def load(self, filepath: str):
        """キャリブレーターを読み込み"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.method = data['method']
        self.temperature = data['temperature']
        self.slope = data['slope']
        self.intercept = data['intercept']
        self.is_fitted = data['is_fitted']

class DynamicThresholdOptimizer:
    """動的しきい値最適化"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.prediction_history = []
        self.outcome_history = []
        self.confidence_history = []
        
    def add_result(self, prediction: int, confidence: float, actual_outcome: int):
        """
        取引結果を追加
        Args:
            prediction: 予測（0=NO_TRADE, 1=TRADE）
            confidence: 信頼度
            actual_outcome: 実際の結果（0=失敗, 1=成功）
        """
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        self.outcome_history.append(actual_outcome)
        
        # 履歴サイズ制限
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
            self.confidence_history.pop(0)
            self.outcome_history.pop(0)
    
    def optimize_threshold(self, target_win_rate: float = 0.65) -> float:
        """
        目標勝率に基づく最適しきい値計算
        Args:
            target_win_rate: 目標勝率
        Returns:
            最適しきい値
        """
        if len(self.prediction_history) < 20:
            return 0.58  # デフォルト値
        
        # 信頼度レンジでの勝率計算
        confidences = np.array(self.confidence_history)
        outcomes = np.array(self.outcome_history)
        predictions = np.array(self.prediction_history)
        
        # TRADE予測のみを対象
        trade_mask = predictions == 1
        if trade_mask.sum() < 10:
            return 0.58
        
        trade_confidences = confidences[trade_mask]
        trade_outcomes = outcomes[trade_mask]
        
        best_threshold = 0.58
        best_score = -999
        
        # しきい値を変化させて最適値を探索
        for threshold in np.linspace(0.50, 0.90, 41):
            high_conf_mask = trade_confidences >= threshold
            
            if high_conf_mask.sum() >= 5:  # 最低5サンプル
                win_rate = np.mean(trade_outcomes[high_conf_mask])
                trade_count = high_conf_mask.sum()
                
                # スコア = 勝率への近さ - ペナルティ（取引数減少）
                win_rate_score = 1.0 - abs(win_rate - target_win_rate)
                volume_penalty = max(0, (20 - trade_count) * 0.02)  # 取引数減少ペナルティ
                
                score = win_rate_score - volume_penalty
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        
        return best_threshold

class ProductionConfidenceHandler:
    """実運用信頼度ハンドラー（MT5連携対応）"""
    
    def __init__(self, config_path: str = "config/production_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # コンポーネント初期化
        self.calibrator = LightweightCalibrator(
            method=self.config.get('calibration_method', 'temperature')
        )
        self.threshold_optimizer = DynamicThresholdOptimizer(
            history_size=self.config.get('history_size', 100)
        )
        
        # 統計情報
        self.stats = {
            'total_predictions': 0,
            'trade_signals': 0,
            'successful_trades': 0,
            'current_win_rate': 0.0,
            'current_threshold': self.config['base_threshold']
        }
        
        # キャリブレーター読み込み（存在する場合）
        calibrator_path = self.config.get('calibrator_save_path', 'models/production_calibrator.json')
        if os.path.exists(calibrator_path):
            try:
                self.calibrator.load(calibrator_path)
                logger.info(f"キャリブレーター読み込み完了: {calibrator_path}")
            except Exception as e:
                logger.warning(f"キャリブレーター読み込み失敗: {e}")
    
    def _load_config(self) -> Dict:
        """設定読み込み"""
        default_config = ConfidenceConfig().to_dict()
        default_config.update({
            'calibration_method': 'temperature',
            'history_size': 100,
            'auto_threshold_update': True,
            'min_samples_for_update': 20,
            'calibrator_save_path': 'models/production_calibrator.json',
            'stats_save_path': 'logs/production_stats.json'
        })
        
        try:
            with open(self.config_path, 'r') as f:
                user_config = json.load(f)
            return {**default_config, **user_config}
        except FileNotFoundError:
            logger.info(f"設定ファイル作成: {self.config_path}")
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def process_prediction(self, raw_prediction: np.ndarray, 
                         return_json: bool = True) -> Dict:
        """
        リアルタイム予測処理（MT5連携用）
        Args:
            raw_prediction: 生のモデル予測確率 [n_classes]
            return_json: JSON形式で返すか
        Returns:
            処理済み予測結果
        """
        # 1. 基本信頼度計算
        raw_confidence = np.max(raw_prediction)
        predicted_class = np.argmax(raw_prediction)
        
        # 2. キャリブレーション適用
        if self.config['calibration_enabled']:
            calibrated_confidence = self.calibrator.predict(np.array([raw_confidence]))[0]
        else:
            calibrated_confidence = raw_confidence
        
        # 3. 現在のしきい値取得
        current_threshold = self.stats['current_threshold']
        
        # 4. 取引判定
        should_trade = calibrated_confidence >= current_threshold
        
        # 5. 最適レンジ判定（急峻性対策）
        in_optimal_range = (
            self.config['optimal_range_lower'] <= calibrated_confidence <= 
            self.config['optimal_range_upper']
        )
        
        # 6. ロットサイズ計算
        lot_size = self._calculate_lot_size(calibrated_confidence)
        
        # 7. リスク調整済み判定
        risk_adjusted_trade = should_trade and (
            in_optimal_range or calibrated_confidence >= 0.70
        )
        
        # 8. 結果構築
        result = {
            'timestamp': datetime.now().isoformat(),
            'raw_confidence': float(raw_confidence),
            'calibrated_confidence': float(calibrated_confidence),
            'predicted_class': int(predicted_class),
            'should_trade': bool(risk_adjusted_trade),
            'lot_size': float(lot_size),
            'threshold_info': {
                'current_threshold': float(current_threshold),
                'in_optimal_range': bool(in_optimal_range),
                'optimal_range': [
                    self.config['optimal_range_lower'],
                    self.config['optimal_range_upper']
                ]
            },
            'metadata': {
                'calibration_enabled': self.config['calibration_enabled'],
                'calibration_method': self.calibrator.method,
                'total_predictions': self.stats['total_predictions'],
                'current_win_rate': self.stats['current_win_rate']
            }
        }
        
        # 9. 統計更新
        self.stats['total_predictions'] += 1
        if risk_adjusted_trade:
            self.stats['trade_signals'] += 1
        
        return result if return_json else result
    
    def _calculate_lot_size(self, confidence: float) -> float:
        """
        信頼度ベースのロットサイズ計算
        Args:
            confidence: キャリブレーション済み信頼度
        Returns:
            推奨ロットサイズ
        """
        base_lot = self.config['lot_size_base']
        max_lot = self.config['lot_size_max']
        min_conf = self.config['min_confidence_for_trade']
        max_conf = self.config['max_confidence_for_scaling']
        
        if confidence < min_conf:
            return 0.0
        
        # 線形スケーリング
        confidence_ratio = (confidence - min_conf) / (max_conf - min_conf)
        confidence_ratio = np.clip(confidence_ratio, 0.0, 1.0)
        
        # 最適レンジでのボーナス
        if (self.config['optimal_range_lower'] <= confidence <= 
            self.config['optimal_range_upper']):
            confidence_ratio *= 1.2  # 20%ボーナス
        
        lot_size = base_lot + (max_lot - base_lot) * confidence_ratio
        return round(lot_size, 2)
    
    def add_trade_result(self, prediction: int, confidence: float, 
                        profit_pips: float, success: bool):
        """
        取引結果を追加（学習用）
        Args:
            prediction: 予測クラス
            confidence: 使用した信頼度
            profit_pips: 実現損益（pips）
            success: 成功かどうか
        """
        # しきい値最適化に追加
        trade_signal = 1 if prediction != 0 else 0
        outcome = 1 if success else 0
        
        self.threshold_optimizer.add_result(trade_signal, confidence, outcome)
        
        # 統計更新
        if trade_signal == 1:
            if success:
                self.stats['successful_trades'] += 1
            
            # 勝率計算
            total_trades = (self.stats['trade_signals'] if 
                          self.stats['trade_signals'] > 0 else 1)
            self.stats['current_win_rate'] = self.stats['successful_trades'] / total_trades
        
        # 動的しきい値更新
        if (self.config['auto_threshold_update'] and 
            len(self.threshold_optimizer.prediction_history) >= 
            self.config['min_samples_for_update']):
            
            new_threshold = self.threshold_optimizer.optimize_threshold()
            if abs(new_threshold - self.stats['current_threshold']) > 0.01:
                logger.info(f"しきい値更新: {self.stats['current_threshold']:.3f} → {new_threshold:.3f}")
                self.stats['current_threshold'] = new_threshold
    
    def get_mt5_signal(self, raw_prediction: np.ndarray) -> str:
        """
        MT5用のJSON信号生成
        Args:
            raw_prediction: 生の予測確率
        Returns:
            JSON文字列
        """
        result = self.process_prediction(raw_prediction)
        
        # MT5用に簡素化
        mt5_signal = {
            'action': 'BUY' if result['predicted_class'] == 1 and result['should_trade'] else 'NO_TRADE',
            'confidence': result['calibrated_confidence'],
            'lot_size': result['lot_size'],
            'timestamp': result['timestamp'],
            'in_optimal_range': result['threshold_info']['in_optimal_range']
        }
        
        return json.dumps(mt5_signal, separators=(',', ':'))  # 最小化JSON
    
    def save_stats(self):
        """統計情報保存"""
        stats_with_config = {
            'stats': self.stats,
            'config': self.config,
            'last_updated': datetime.now().isoformat()
        }
        
        stats_path = self.config['stats_save_path']
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        
        with open(stats_path, 'w') as f:
            json.dump(stats_with_config, f, indent=2)
    
    def export_calibration(self):
        """キャリブレーター保存"""
        if self.calibrator.is_fitted:
            save_path = self.config['calibrator_save_path']
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.calibrator.save(save_path)
            logger.info(f"キャリブレーター保存: {save_path}")

class ConfidenceMonitor:
    """信頼度モニタリング（運用監視）"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.recent_confidences = []
        self.recent_outcomes = []
        
    def add_observation(self, confidence: float, outcome: bool):
        """観測データ追加"""
        self.recent_confidences.append(confidence)
        self.recent_outcomes.append(int(outcome))
        
        # ウィンドウサイズ制限
        if len(self.recent_confidences) > self.window_size:
            self.recent_confidences.pop(0)
            self.recent_outcomes.pop(0)
    
    def get_calibration_health(self) -> Dict:
        """キャリブレーション健全性チェック"""
        if len(self.recent_confidences) < 10:
            return {'status': 'insufficient_data', 'samples': len(self.recent_confidences)}
        
        confidences = np.array(self.recent_confidences)
        outcomes = np.array(self.recent_outcomes)
        
        # 簡易ECE計算
        n_bins = 5
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        ece = 0
        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_confidence = confidences[mask].mean()
                bin_accuracy = outcomes[mask].mean()
                bin_weight = mask.sum() / len(confidences)
                ece += abs(bin_confidence - bin_accuracy) * bin_weight
        
        # 警告レベル判定
        if ece > 0.15:
            status = 'poor_calibration'
        elif ece > 0.08:
            status = 'moderate_calibration'
        else:
            status = 'good_calibration'
        
        return {
            'status': status,
            'ece': ece,
            'samples': len(self.recent_confidences),
            'mean_confidence': confidences.mean(),
            'mean_accuracy': outcomes.mean(),
            'recommendation': self._get_recommendation(status, ece)
        }
    
    def _get_recommendation(self, status: str, ece: float) -> str:
        """推奨アクション"""
        if status == 'poor_calibration':
            return 'recalibration_required'
        elif status == 'moderate_calibration':
            return 'monitor_closely'
        else:
            return 'continue_current_settings'

# ユーティリティ関数群
def create_production_config(base_threshold: float = 0.58,
                           optimal_range_width: float = 0.03,
                           output_path: str = "config/production_config.json") -> Dict:
    """実運用設定ファイル作成"""
    config = ConfidenceConfig(
        base_threshold=base_threshold,
        optimal_range_lower=base_threshold,
        optimal_range_upper=base_threshold + optimal_range_width
    ).to_dict()
    
    config.update({
        'calibration_method': 'temperature',
        'auto_threshold_update': True,
        'min_samples_for_update': 20,
        'history_size': 100
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config

def quick_mt5_integration_test():
    """MT5連携テスト"""
    handler = ProductionConfidenceHandler()
    
    # サンプル予測
    sample_predictions = [
        np.array([0.3, 0.7]),  # TRADE、中程度信頼度
        np.array([0.1, 0.9]),  # TRADE、高信頼度
        np.array([0.8, 0.2]),  # NO_TRADE
        np.array([0.4, 0.6])   # TRADE、低信頼度
    ]
    
    print("=== MT5連携テスト ===")
    for i, pred in enumerate(sample_predictions):
        signal = handler.get_mt5_signal(pred)
        print(f"予測 {i+1}: {signal}")
    
    return handler

if __name__ == "__main__":
    # テスト実行
    print("=== 実運用信頼度ハンドラー テスト ===")
    
    # 設定作成
    config = create_production_config()
    print(f"設定作成完了: base_threshold={config['base_threshold']}")
    
    # MT5連携テスト
    handler = quick_mt5_integration_test()
    
    # 統計保存テスト
    handler.save_stats()
    print("統計保存完了")
    
    print("✅ テスト完了")