#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
モデル破綻診断スクリプト
予測分布・信頼度・ラベル分布を詳細分析
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime

def analyze_model_breakdown():
    """モデル破綻の詳細分析"""
    
    print("🔍 モデル破綻診断開始")
    print("="*60)
    
    # 最新の評価結果を読み込み
    try:
        with open('models/evaluation_report/performance_summary_20250723_154859.json', 'r') as f:
            results = json.load(f)
        
        print("📊 基本統計:")
        print(f"   テスト精度: {results['test_accuracy']:.1%}")
        print(f"   テスト損失: {results['test_loss']:.2f}")
        
        print("\n🎯 F1スコア分析:")
        for class_name, f1 in results['f1_scores'].items():
            print(f"   {class_name:8}: {f1:.3f}")
        
        print("\n🔄 閾値性能分析:")
        thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
        for thresh in thresholds:
            if thresh in results['threshold_performance']:
                perf = results['threshold_performance'][thresh]
                print(f"   閾値{thresh}: 精度={perf['accuracy']:.3f}, "
                      f"F1={perf['f1_score']:.3f}, "
                      f"予測数={perf['num_predictions']:,}")
        
        # 問題診断
        print("\n⚠️ 診断結果:")
        
        # 1. 閾値性能が全て同じかチェック
        accuracies = [results['threshold_performance'][t]['accuracy'] 
                     for t in ['0.1', '0.3', '0.5', '0.7', '0.9']]
        if len(set(accuracies)) == 1:
            print("   🚨 信頼度固定問題: 全閾値で同じ性能")
        
        # 2. F1スコアがゼロのクラスをチェック
        zero_f1_classes = [name for name, f1 in results['f1_scores'].items() if f1 == 0.0]
        if zero_f1_classes:
            print(f"   🚨 予測ゼロクラス: {', '.join(zero_f1_classes)}")
        
        # 3. 精度が低すぎるかチェック
        if results['test_accuracy'] < 0.4:
            print(f"   🚨 精度低下問題: {results['test_accuracy']:.1%} < 40%")
        
        return results
        
    except FileNotFoundError:
        print("❌ 評価結果ファイルが見つかりません")
        return None

def suggest_solutions():
    """解決策の提案"""
    
    print("\n💡 解決策提案:")
    print("="*60)
    
    print("🎯 アプローチ1: ラベリング戦略変更")
    print("   - TP/SL条件をさらに緩和（TP:10pips, SL:5pips）")
    print("   - future_windowを300-500に拡大")
    print("   - NO_TRADEクラスを60-70%に増加")
    
    print("\n🎯 アプローチ2: モデル構造変更")
    print("   - Conv1D → LSTM/Transformerに変更")
    print("   - シーケンス長を短縮（30→15）")
    print("   - ドロップアウト率を削減（0.4→0.2）")
    
    print("\n🎯 アプローチ3: データ戦略変更")
    print("   - クラス重み調整強化")
    print("   - アンダーサンプリング実行")
    print("   - 異なる時間帯でのデータ分割")
    
    print("\n🎯 アプローチ4: 評価手法変更")
    print("   - 2クラス分類（BUY vs SELL）に簡素化")
    print("   - 回帰問題として再定義")
    print("   - アンサンブル手法の導入")

def generate_next_config():
    """次回テスト用設定生成"""
    
    print("\n🚀 次回テスト設定:")
    print("="*60)
    
    config_v4 = {
        "trading": {
            "pair": "USDJPY",
            "spread_pips": 0.7,
            "pip_value": 0.01,
            "tp_pips": 10.0,    # さらに緩和
            "sl_pips": 5.0,     # 厳格化
            "future_window": 300  # 大幅拡大
        },
        "model": {
            "architecture": "lstm",     # Conv1D → LSTM
            "sequence_length": 15,      # 30 → 15
            "dropout_rate": 0.2,        # 0.4 → 0.2
            "learning_rate": 0.0001,    # より低い学習率
            "epochs": 15,               # 早期収束
            "early_stopping_patience": 3
        }
    }
    
    print("📋 主要変更点:")
    print("   - アーキテクチャ: Conv1D → LSTM")
    print("   - TP/SL比率: 10:5 = 2:1")
    print("   - 未来窓: 200 → 300ティック")
    print("   - シーケンス長: 30 → 15")
    print("   - 学習率: 0.0003 → 0.0001")
    
    return config_v4

def main():
    """メイン実行"""
    
    print("🧪 USDJPYスキャルピングAI 破綻診断システム")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 分析実行
    results = analyze_model_breakdown()
    
    if results:
        # 解決策提案
        suggest_solutions()
        
        # 次回設定生成
        next_config = generate_next_config()
        
        print("\n✅ 診断完了")
        print("次回はLSTMベースのアーキテクチャで再挑戦を推奨")
    else:
        print("❌ 診断に必要なファイルが不足しています")

if __name__ == "__main__":
    main()