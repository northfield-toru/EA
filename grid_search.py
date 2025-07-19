#!/usr/bin/env python3
"""
Phase 2E: 並列グリッドサーチスクリプト
各パラメータセットを個別プロセスで学習・評価
"""

import os
import sys
import json
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Tuple

# 自作モジュール
from train import ParameterOptimizedTrainer

def train_single_parameter_worker(args: Tuple) -> Dict:
    """
    並列処理用ワーカー関数
    """
    data_path, tp_pips, sl_pips, trade_threshold, sample_size, epochs, output_dir = args
    
    try:
        optimizer = ParameterOptimizedTrainer(data_path, output_dir)
        result = optimizer.train_single_parameter_set(
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            trade_threshold=trade_threshold,
            sample_size=sample_size,
            epochs=epochs
        )
        return result
        
    except Exception as e:
        return {
            'parameters': {'tp_pips': tp_pips, 'sl_pips': sl_pips, 'trade_threshold': trade_threshold},
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def run_parallel_grid_search(data_path: str,
                            sample_size: int = 500000,
                            epochs: int = 25,
                            max_workers: int = None) -> Dict:
    """
    並列グリッドサーチ実行
    """
    print("⚡" * 30)
    print("    Phase 2E: 並列グリッドサーチ")
    print("⚡" * 30)
    
    # パラメータ組み合わせ
    tp_options = [4.0, 5.0, 6.0]
    sl_options = [3.0, 4.0]
    trade_threshold_options = [0.20, 0.25, 0.30, 0.35]
    
    # 全組み合わせ生成
    parameter_combinations = []
    for tp in tp_options:
        for sl in sl_options:
            for tr in trade_threshold_options:
                parameter_combinations.append((data_path, tp, sl, tr, sample_size, epochs, "phase2e_parallel_results"))
    
    total_combinations = len(parameter_combinations)
    
    # ワーカー数設定
    if max_workers is None:
        max_workers = min(mp.cpu_count() - 1, total_combinations)
    
    print(f"並列実行設定:")
    print(f"  組み合わせ数: {total_combinations}")
    print(f"  ワーカー数: {max_workers}")
    print(f"  予想実行時間: {(total_combinations * epochs * 2) // max_workers // 60} 分")
    
    # 並列実行
    print(f"\n⚡ 並列学習開始...")
    
    with mp.Pool(processes=max_workers) as pool:
        results = pool.map(train_single_parameter_worker, parameter_combinations)
    
    # 結果集計
    all_results = {}
    best_result = None
    best_profit = -999
    best_param_id = None
    successful_count = 0
    
    for i, (_, tp, sl, tr, _, _, _) in enumerate(parameter_combinations):
        param_id = f"TP{tp}_SL{sl}_TR{tr:.2f}"
        result = results[i]
        all_results[param_id] = result
        
        if 'eval_results' in result:
            successful_count += 1
            profit = result['eval_results'].get('expected_profit_per_trade', -999)
            
            if profit > best_profit:
                best_profit = profit
                best_result = result
                best_param_id = param_id
    
    # 結果表示
    print(f"\n⚡ 並列グリッドサーチ完了")
    print(f"成功率: {successful_count}/{total_combinations} ({successful_count/total_combinations:.1%})")
    
    if best_result:
        print(f"\n🏆 最優秀結果: {best_param_id}")
        print(f"利益: {best_profit:+.2f}pips/トレード")
        
        if best_profit > 0:
            print("🎉 並列処理で利益化達成！")
    
    # 統合結果保存
    summary = {
        'phase': '2E_parallel_grid_search',
        'total_combinations': total_combinations,
        'successful_results': successful_count,
        'max_workers': max_workers,
        'best_param_id': best_param_id,
        'best_profit': best_profit,
        'all_results': all_results,
        'execution_time': datetime.now().isoformat()
    }
    
    os.makedirs("phase2e_parallel_results", exist_ok=True)
    with open('phase2e_parallel_results/parallel_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return summary

if __name__ == "__main__":
    if len(sys.argv) < 2:
        data_path = "data/usdjpy_ticks.csv"
    else:
        data_path = sys.argv[1]
    
    # 並列グリッドサーチ実行
    results = run_parallel_grid_search(
        data_path=data_path,
        sample_size=300000,  # 並列処理用に軽量化
        epochs=20,           # 並列処理用に軽量化
        max_workers=4        # CPU数に応じて調整
    )
    
    print(f"\n📊 並列グリッドサーチ完了")
    if results['best_profit'] > 0:
        print(f"🎉 利益化成功: {results['best_param_id']}")
    else:
        print(f"📈 最良結果: {results['best_profit']:+.2f}pips")