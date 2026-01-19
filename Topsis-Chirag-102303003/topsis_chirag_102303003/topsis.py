import pandas as pd
import numpy as np


class TopsisError(Exception):
    pass


def parse_weights_impacts(weights: str, impacts: str, n_cols: int):
    if "," not in weights or "," not in impacts:
        raise TopsisError("Weights and Impacts must be comma-separated values.")

    w = weights.split(",")
    im = impacts.split(",")

    if len(w) != len(im):
        raise TopsisError("Number of weights must be equal to number of impacts.")

    if len(w) != n_cols:
        raise TopsisError("Number of weights/impacts must match number of numeric columns (2nd to last).")

    try:
        w = list(map(float, w))
    except:
        raise TopsisError("Weights must be numeric values only.")

    for x in im:
        x = x.strip()
        if x not in ["+", "-"]:
            raise TopsisError("Impacts must be either '+' or '-' only.")

    return np.array(w, dtype=float), [x.strip() for x in im]


def topsis(input_file: str, weights: str, impacts: str, output_file: str):
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        raise TopsisError("Input file not found.")
    except Exception as e:
        raise TopsisError(f"Error reading input file: {str(e)}")

    if df.shape[1] < 3:
        raise TopsisError("Input file must contain three or more columns.")

    data = df.iloc[:, 1:].copy()

    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise TopsisError("From 2nd to last columns must contain numeric values only.")

    n_cols = data.shape[1]

    w, im = parse_weights_impacts(weights, impacts, n_cols)

    norm = data / np.sqrt((data ** 2).sum(axis=0))

    weighted = norm * w

    ideal_best = []
    ideal_worst = []

    for j in range(n_cols):
        col_vals = weighted.iloc[:, j]
        if im[j] == "+":
            ideal_best.append(col_vals.max())
            ideal_worst.append(col_vals.min())
        else:
            ideal_best.append(col_vals.min())
            ideal_worst.append(col_vals.max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    score = dist_worst / (dist_best + dist_worst)

    df["Topsis Score"] = score
    df["Rank"] = df["Topsis Score"].rank(ascending=False, method="dense").astype(int)

    df.to_excel(output_file, index=False)
    return df
