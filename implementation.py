import pandas as pd
import numpy as np

def topsis(input_file_name, weights, impacts, result_file_name):
    if any(arg is None for arg in [input_file_name, weights, impacts, result_file_name]):
        print("Error: Missing one or more required parameters.")
        return

    try:
        df = pd.read_csv(input_file_name)
    except FileNotFoundError:
        print(f"Error: File '{input_file_name}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if df.shape[1] < 3:
        print("Error: Input file must contain at least three columns.")
        return

    # Check if weights and impacts are separated by commas
    if ',' not in weights:
        print("Error: Weights must be separated by commas.")
        return
    if ',' not in impacts:
        print("Error: Impacts must be separated by commas.")
        return

    try:
        weights = [float(w.strip()) for w in weights.split(",")]
    except Exception:
        print("Error: Weights must be numeric")
        return
    impacts=[i.strip() for i in impacts.split(",")]

    num_cols = df.shape[1] - 1
    if not len(weights) == len(impacts) == num_cols:
        print("Error: Number of weights, impacts, and columns (from 2nd to last) must be the same.")
        return

    for imp in impacts:
        if imp not in ['+', '-']:
            print("Error: Impacts must be either '+' or '-'.")
            return

    data = df.iloc[:, 1:]
    def is_numeric(x):
        try:
            float(x)
            return True
        except Exception:
            return False
    if not np.all(data.map(is_numeric)):
        print("Error: All values from 2nd to last columns must be numeric.")
        return
    data = data.apply(pd.to_numeric)

    norm = np.sqrt((data ** 2).sum())
    norm_data = data / norm

    weighted_data = norm_data * weights

    ideal_best = []
    ideal_worst = []
    for i, imp in enumerate(impacts):
        if imp == '+':
            ideal_best.append(weighted_data.iloc[:, i].max())
            ideal_worst.append(weighted_data.iloc[:, i].min())
        else:
            ideal_best.append(weighted_data.iloc[:, i].min())
            ideal_worst.append(weighted_data.iloc[:, i].max())

    s_pos = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    s_neg = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))

    score = s_neg / (s_pos + s_neg)
    df['Topsis Score'] = score
    df['Rank'] = df['Topsis Score'].rank(ascending=False, method='max').astype(int)

    try:
        df.to_csv(result_file_name, index=False)
        print(f"Result written to {result_file_name}")
    except Exception as e:
        print(f"Error writing result file: {e}")

topsis("data.csv", "1,2,1,4,3", "-,+,-,-,+", "result.csv")