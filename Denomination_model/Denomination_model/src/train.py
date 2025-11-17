# denomination_split/train.py
import os
import argparse
import pandas as pd
from Denomination_model.Denomination_model.src.train import make_features, train_random_forest

def main(data_path, save_path):
    print("Loading data from:", data_path)
    df = pd.read_csv(data_path)
    print(f"Total rows in dataset: {len(df)}")

    X, y, idx = make_features(df)
    print("Feature matrix shape:", X.shape)
    print("Target matrix shape:", y.shape)
    print("Training random forest multi-output regressor...")

    model, metrics, X_test, y_test, preds = train_random_forest(X, y, save_path=save_path)

    print("Training finished. Metrics:")
    print("Overall MAE (counts):", metrics['overall_mae'])
    print("Per denomination MAE:")
    for d, m in metrics['per_denom_mae'].items():
        print(f"  {d}: {m:.3f}")

    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/kirana.csv", help="path to kirana.csv")
    parser.add_argument("--out", default="../models/denomination_rf.joblib", help="where to save model")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    main(args.data, args.out)
