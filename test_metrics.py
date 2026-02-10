import numpy as np
from scripts.eval_downstream_metrics import calculate_pointing_accuracy, simulate_turn_reduction

# Test 1: Perfect prediction
y_true = np.array([[0, 1, 0, 0, 0, 0]]) # Missing 'what' (idx 1)
y_pred = np.array([[0.1, 0.9, 0.1, 0.1, 0.1, 0.1]]) # Confidence 0.9 on 'what'

acc = calculate_pointing_accuracy(y_true, y_pred)
print(f"Test 1 Acc (Exp 1.0): {acc}")

sim = simulate_turn_reduction(y_true, y_pred, threshold=0.5)
print(f"Test 1 Sim (Exp < 2.0): {sim}")

# Test 2: Low confidence (fallback)
y_pred_low = np.array([[0.1, 0.4, 0.1, 0.1, 0.1, 0.1]]) # Confidence 0.4 (< 0.5)
sim_low = simulate_turn_reduction(y_true, y_pred_low, threshold=0.5)
print(f"Test 2 Sim (Exp 2.0): {sim_low}")

# Test 3: Wrong prediction
y_pred_wrong = np.array([[0.1, 0.1, 0.9, 0.1, 0.1, 0.1]]) # Predicts 'when' (idx 2)
sim_wrong = simulate_turn_reduction(y_true, y_pred_wrong, threshold=0.5)
print(f"Test 3 Sim (Exp 3.0): {sim_wrong}")
