"""
クイック デバッグ - 3値分類の問題を直接特定
"""

import pandas as pd
import numpy as np

# 最小限のテスト
print("=== 3値分類エラー直接検証 ===")

# 1. 簡単なテストデータ作成
print("1. テストデータ作成...")
test_data = pd.DataFrame({
    'close': [157.0, 157.05, 157.02, 157.08, 157.03, 157.06, 157.01, 157.09, 157.04, 157.07]
})
print(f"テストデータ: {test_data.shape}")
print(test_data['close'].values)

# 2. labeling.py から直接インポートしてテスト
print("\n2. ラベル生成関数直接テスト...")

try:
    from labeling import ScalpingLabeler
    print("ScalpingLabeler インポート成功")
    
    # 3. ラベラー作成
    labeler = ScalpingLabeler(
        profit_pips=3.0,   # より緩い条件
        loss_pips=3.0,
        lookforward_ticks=5,  # 短い前方参照
        use_flexible_conditions=True
    )
    print("ScalpingLabeler 作成成功")
    
    # 4. 3値分類ラベル生成を直接呼び出し
    print("\n3. create_labels_vectorized 直接呼び出し...")
    result = labeler.create_labels_vectorized(test_data)
    
    print(f"結果の型: {type(result)}")
    if result is not None:
        print(f"結果の長さ: {len(result)}")
        print(f"結果の値: {result.values}")
        print(f"結果の分布: {result.value_counts().to_dict()}")
        print("✅ 3値分類 成功！")
    else:
        print("❌ 3値分類で None が返されました")
    
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
except Exception as e:
    print(f"❌ エラー発生: {e}")
    import traceback
    traceback.print_exc()

# 5. 2値分類も確認
print("\n4. 2値分類も確認...")
try:
    binary_result = labeler.create_binary_labels_vectorized(test_data)
    if binary_result is not None:
        print(f"2値分類成功: 長さ={len(binary_result)}, 分布={binary_result.value_counts().to_dict()}")
    else:
        print("❌ 2値分類で None")
except Exception as e:
    print(f"❌ 2値分類エラー: {e}")

print("\n=== デバッグ完了 ===")