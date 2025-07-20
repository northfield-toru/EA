"""
統合改善信頼度システム（完全版）
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

class TemperatureScaling:
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, probs, y_true):
        def objective(temp):
            # 温度適用
            scaled_probs = probs / temp
            scaled_probs = np.exp(scaled_probs) / np.sum(np.exp(scaled_probs), axis=1, keepdims=True)
            
            # ECE計算
            max_probs = np.max(scaled_probs, axis=1)
            pred_classes = np.argmax(scaled_probs, axis=1)
            
            ece = 0
            for i in range(10):
                bin_lower = i / 10
                bin_upper = (i + 1) / 10
                in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
                
                if in_bin.sum() > 0:
                    bin_acc = np.mean(pred_classes[in_bin] == y_true[in_bin])
                    bin_conf = max_probs[in_bin].mean()
                    bin_weight = in_bin.sum() / len(max_probs)
                    ece += abs(bin_conf - bin_acc) * bin_weight
            
            return ece
        
        result = minimize_scalar(objective, bounds=(0.1, 5.0), method='bounded')
        self.temperature = result.x
        return self
    
    def predict_proba(self, probs):
        scaled_probs = probs / self.temperature
        return np.exp(scaled_probs) / np.sum(np.exp(scaled_probs), axis=1, keepdims=True)

class SmoothThresholdFilter:
    def __init__(self, base_threshold=0.58, range_width=0.03):
        self.base_threshold = base_threshold
        self.upper_threshold = base_threshold + range_width
    
    def get_smooth_weight(self, confidence):
        if confidence < self.base_threshold:
            return 0.0
        elif confidence > self.upper_threshold:
            return 1.0
        else:
            # 線形補間でスムーズな変化
            progress = (confidence - self.base_threshold) / (self.upper_threshold - self.base_threshold)
            return progress
    
    def apply_filter(self, predictions, confidences):
        weights = np.array([self.get_smooth_weight(c) for c in confidences])
        return predictions, weights

def calculate_ece(y_true, y_prob, n_bins=10):
    """ECE計算"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    max_probs = np.max(y_prob, axis=1)
    pred_labels = np.argmax(y_prob, axis=1)
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        prop_in_bin = in_bin.sum() / len(max_probs)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(pred_labels[in_bin] == y_true[in_bin])
            avg_confidence_in_bin = max_probs[in_bin].mean()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def run_complete_confidence_improvement(predictions, true_labels):
    """完全な信頼度改善を実行"""
    
    print("🔧 信頼度改善システム実行中...")
    
    # 1. 改善前の評価
    original_ece = calculate_ece(true_labels, predictions)
    print(f"改善前ECE: {original_ece:.4f}")
    
    # 2. 温度スケーリング適用
    temp_scaler = TemperatureScaling()
    temp_scaler.fit(predictions, true_labels)
    
    calibrated_probs = temp_scaler.predict_proba(predictions)
    calibrated_ece = calculate_ece(true_labels, calibrated_probs)
    
    print(f"温度スケーリング後ECE: {calibrated_ece:.4f}")
    print(f"最適温度: {temp_scaler.temperature:.3f}")
    
    # 3. スムーズフィルター適用
    smooth_filter = SmoothThresholdFilter()
    max_confidences = np.max(calibrated_probs, axis=1)
    _, smooth_weights = smooth_filter.apply_filter(calibrated_probs, max_confidences)
    
    # 4. 結果評価
    improvement = original_ece - calibrated_ece
    success = improvement > 0.02  # 2%以上の改善で成功
    
    results = {
        'original_ece': original_ece,
        'improved_ece': calibrated_ece,
        'improvement': improvement,
        'optimal_temperature': temp_scaler.temperature,
        'calibrated_probs': calibrated_probs,
        'smooth_weights': smooth_weights,
        'success': success
    }
    
    print(f"信頼度改善: {improvement:+.4f}")
    print(f"改善成功: {'YES' if success else 'NO'}")
    
    return results
