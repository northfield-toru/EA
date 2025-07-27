#!/usr/bin/env python3
"""
学習問題診断スクリプト
accuracy低下の根本原因を特定
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def diagnose_learning_problem():
    """学習問題の診断"""
    
    print("=" * 60)
    print("🔍 学習問題診断開始")
    print("=" * 60)
    print()
    
    # 1. データ基本分析
    diagnose_data_quality()
    
    # 2. ラベル分布分析
    diagnose_label_distribution()
    
    # 3. 特徴量分析
    diagnose_features()
    
    # 4. future_window分析
    diagnose_future_window()
    
    # 5. 推奨設定出力
    recommend_settings()

def diagnose_data_quality():
    """データ品質診断"""
    
    print("📊 1. データ品質診断")
    print("-" * 30)
    
    try:
        # ティックデータ読み込み
        df = pd.read_csv('data/usdjpy_ticks.csv', sep='\t')
        df.columns = [col.strip('<>') for col in df.columns]
        df['MID'] = (df['BID'] + df['ASK']) / 2
        
        print(f"データ件数: {len(df):,}")
        print(f"期間: {df['DATE'].min()} - {df['DATE'].max()}")
        
        # 価格変動分析
        price_changes = df['MID'].pct_change().dropna()
        price_changes_pips = price_changes / 0.01  # pips換算
        
        print(f"価格変動統計:")
        print(f"  平均変動: {price_changes_pips.mean():.4f} pips")
        print(f"  標準偏差: {price_changes_pips.std():.4f} pips")
        print(f"  最大変動: {price_changes_pips.max():.4f} pips")
        print(f"  最小変動: {price_changes_pips.min():.4f} pips")
        
        # スプレッド分析
        actual_spread = (df['ASK'] - df['BID']) / 0.01
        print(f"実際のスプレッド:")
        print(f"  平均: {actual_spread.mean():.2f} pips")
        print(f"  中央値: {actual_spread.median():.2f} pips")
        print(f"  範囲: {actual_spread.min():.2f} - {actual_spread.max():.2f} pips")
        
        # 設定スプレッドとの比較
        config_spread = 0.7
        if abs(actual_spread.median() - config_spread) > 0.3:
            print(f"⚠️  WARNING: 設定スプレッド({config_spread})と実際({actual_spread.median():.2f})に差異")
        else:
            print(f"✅ スプレッド設定は適切")
        
        print()
        
    except Exception as e:
        print(f"❌ データ品質診断エラー: {e}")
        print()

def diagnose_label_distribution():
    """ラベル分布診断"""
    
    print("🏷️  2. ラベル分布診断")
    print("-" * 30)
    
    try:
        # 最新のラベル分布確認
        label_files = []
        for root, dirs, files in os.walk('logs'):
            for file in files:
                if file == 'label_distribution.csv':
                    label_files.append(os.path.join(root, file))
        
        if not label_files:
            print("❌ ラベル分布ファイルが見つかりません")
            print()
            return
        
        # 最新ファイル取得
        latest_file = max(label_files, key=os.path.getmtime)
        df_labels = pd.read_csv(latest_file)
        
        print("現在のラベル分布:")
        for _, row in df_labels.iterrows():
            print(f"  {row['Label']}: {row['Count']:,} ({row['Percentage']:.1f}%)")
        
        # 問題判定
        buy_ratio = df_labels[df_labels['Label'] == 'BUY']['Percentage'].iloc[0]
        sell_ratio = df_labels[df_labels['Label'] == 'SELL']['Percentage'].iloc[0]
        no_trade_ratio = df_labels[df_labels['Label'] == 'NO_TRADE']['Percentage'].iloc[0]
        
        issues = []
        if buy_ratio < 5 or sell_ratio < 5:
            issues.append(f"BUY/SELL比率が低すぎる ({buy_ratio:.1f}%/{sell_ratio:.1f}%)")
        
        if no_trade_ratio > 90:
            issues.append(f"NO_TRADE比率が高すぎる ({no_trade_ratio:.1f}%)")
        
        if abs(buy_ratio - sell_ratio) > 5:
            issues.append(f"BUY/SELL不均衡 (差: {abs(buy_ratio - sell_ratio):.1f}%)")
        
        if issues:
            print("⚠️  ラベル分布の問題:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("✅ ラベル分布は適切")
        
        print()
        
    except Exception as e:
        print(f"❌ ラベル分布診断エラー: {e}")
        print()

def diagnose_features():
    """特徴量診断"""
    
    print("🎯 3. 特徴量診断")
    print("-" * 30)
    
    try:
        # 重要特徴量ファイル確認
        feature_files = []
        for root, dirs, files in os.walk('logs'):
            for file in files:
                if file == 'important_features.json':
                    feature_files.append(os.path.join(root, file))
        
        if not feature_files:
            print("❌ 重要特徴量ファイルが見つかりません")
            print("   特徴量-ラベル相関分析が未実行の可能性")
            print()
            return
        
        # 最新ファイル取得
        latest_file = max(feature_files, key=os.path.getmtime)
        with open(latest_file, 'r') as f:
            features = json.load(f)
        
        if not features:
            print("❌ 有効な特徴量が見つかりません")
            print("   → 特徴量とラベルに相関がない可能性が高い")
            print()
            return
        
        print("重要特徴量 (上位5個):")
        for i, feat in enumerate(features[:5]):
            print(f"  {i+1}. {feat['feature']}: 分離度={feat['separation']:.3f}")
        
        # 分離度分析
        top_separation = features[0]['separation'] if features else 0
        
        if top_separation < 0.1:
            print("❌ 致命的: 最高分離度が0.1未満")
            print("   → 特徴量がラベルを全く識別できていない")
        elif top_separation < 0.3:
            print("⚠️  警告: 最高分離度が0.3未満")
            print("   → 特徴量の識別能力が低い")
        else:
            print("✅ 特徴量の分離度は適切")
        
        print()
        
    except Exception as e:
        print(f"❌ 特徴量診断エラー: {e}")
        print()

def diagnose_future_window():
    """future_window診断"""
    
    print("⏱️  4. future_window診断")
    print("-" * 30)
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        future_window = config['data']['future_window']
        tp_pips = config['trading']['tp_pips']
        sl_pips = config['trading']['sl_pips']
        
        print(f"現在設定:")
        print(f"  future_window: {future_window}")
        print(f"  TP: {tp_pips} pips")
        print(f"  SL: {sl_pips} pips")
        
        # ティックデータで分析
        df = pd.read_csv('data/usdjpy_ticks.csv', sep='\t')
        df.columns = [col.strip('<>') for col in df.columns]
        df['MID'] = (df['BID'] + df['ASK']) / 2
        
        # 100ティックでの価格変動分析
        sample_indices = np.random.choice(len(df) - future_window, 1000)
        movements = []
        
        for idx in sample_indices:
            current_price = df['MID'].iloc[idx]
            future_prices = df['MID'].iloc[idx+1:idx+future_window+1]
            
            max_move_up = ((future_prices.max() - current_price) / 0.01)
            max_move_down = ((current_price - future_prices.min()) / 0.01)
            
            movements.append({
                'max_up': max_move_up,
                'max_down': max_move_down
            })
        
        movements_df = pd.DataFrame(movements)
        
        print(f"{future_window}ティック内での最大変動:")
        print(f"  上昇: 平均{movements_df['max_up'].mean():.2f} pips (最大{movements_df['max_up'].max():.2f})")
        print(f"  下降: 平均{movements_df['max_down'].mean():.2f} pips (最大{movements_df['max_down'].max():.2f})")
        
        # TP到達可能性
        tp_reachable = (movements_df['max_up'] >= tp_pips).mean() * 100
        sl_reachable = (movements_df['max_down'] >= sl_pips).mean() * 100
        
        print(f"到達可能性:")
        print(f"  TP({tp_pips}pips): {tp_reachable:.1f}%")
        print(f"  SL({sl_pips}pips): {sl_reachable:.1f}%")
        
        if tp_reachable < 20 or sl_reachable < 20:
            print("⚠️  WARNING: TP/SL到達率が低い")
            print("   → future_windowが短すぎる可能性")
        else:
            print("✅ future_window設定は適切")
        
        print()
        
    except Exception as e:
        print(f"❌ future_window診断エラー: {e}")
        print()

def recommend_settings():
    """推奨設定"""
    
    print("💡 5. 推奨設定")
    print("-" * 30)
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        recommendations = []
        
        # データ分析結果に基づく推奨
        df = pd.read_csv('data/usdjpy_ticks.csv', sep='\t')
        df.columns = [col.strip('<>') for col in df.columns]
        df['MID'] = (df['BID'] + df['ASK']) / 2
        
        # 価格変動分析
        price_changes_pips = df['MID'].pct_change().dropna() / 0.01
        volatility = price_changes_pips.std()
        
        # 推奨future_window
        if volatility < 0.5:
            recommended_window = 200
            recommendations.append(f"future_window: {recommended_window} (低ボラティリティ対応)")
        elif volatility > 2.0:
            recommended_window = 50
            recommendations.append(f"future_window: {recommended_window} (高ボラティリティ対応)")
        else:
            recommended_window = 100
        
        # 推奨TP/SL
        if volatility < 0.5:
            recommendations.append("TP/SL: 2.0/3.0 pips (低ボラティリティ)")
        elif volatility > 2.0:
            recommendations.append("TP/SL: 6.0/8.0 pips (高ボラティリティ)")
        
        # 推奨サンプリング
        if len(df) > 1000000:
            recommendations.append("sample_rate: 3-5 (大容量データ)")
        
        # 推奨学習設定
        recommendations.append("learning_rate: 0.0001 (超保守的)")
        recommendations.append("batch_size: 128 (さらに安定)")
        recommendations.append("loss_type: 'focal' (クラス不均衡対応)")
        
        print("推奨設定変更:")
        for rec in recommendations:
            print(f"  - {rec}")
        
        # 設定ファイル生成
        recommended_config = config.copy()
        recommended_config['data']['future_window'] = recommended_window
        recommended_config['model']['learning_rate'] = 0.0001
        recommended_config['model']['batch_size'] = 128
        recommended_config['model']['loss_type'] = 'focal'
        
        with open('config_recommended.json', 'w') as f:
            json.dump(recommended_config, f, indent=2)
        
        print()
        print("✅ 推奨設定をconfig_recommended.jsonに保存しました")
        
    except Exception as e:
        print(f"❌ 推奨設定生成エラー: {e}")
    
    print()
    print("=" * 60)
    print("🎯 診断完了")
    print("=" * 60)

if __name__ == "__main__":
    diagnose_learning_problem()