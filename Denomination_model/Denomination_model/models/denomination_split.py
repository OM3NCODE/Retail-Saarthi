# denomination_split/model.py
import json
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

DENOMS = [500, 200, 100, 50, 20, 10, 5, 2, 1]

def parse_breakdown_field(s):
    """Parse fields like '{"50":1,"20":1}' or NaN into dict of ints."""
    if pd.isna(s):
        return {}
    if isinstance(s, dict):
        return s
    try:
        # Some rows might contain single quotes or be valid JSON strings
        return json.loads(s)
    except Exception:
        try:
            return eval(s)  # fallback (careful) if it's a python dict string
        except Exception:
            return {}

def build_targets(df):
    """Return numpy array shape (n_samples, n_denoms) with counts per denomination"""
    rows = []
    for v in df['change_breakdown'].apply(parse_breakdown_field):
        counts = [int(v.get(str(d), 0)) for d in DENOMS]
        rows.append(counts)
    return np.array(rows, dtype=int)

def extract_tender_features(df):
    """
    Create columns indicating counts of tendered denominations (what customer gave),
    and other useful numeric features.
    """
    # Parse tendered_breakdown to columns
    tcols = {f"tended_{d}": [] for d in DENOMS}
    for tb in df['tendered_breakdown'].apply(parse_breakdown_field):
        for d in DENOMS:
            tcols[f"tended_{d}"].append(int(tb.get(str(d), 0)))
    tdf = pd.DataFrame(tcols)

    return tdf

def make_features(df):
    """Return features DataFrame X and target array y."""
    # Filter to cash txns with change > 0 and where we have a change_breakdown
    df2 = df[(df['payment_method'] == 'cash') & (df['change_given'].notna())]
    # sometimes change_given is 0.0; only care where change>0
    df2 = df2[df2['change_given'] > 0]

    if df2.empty:
        raise ValueError("No suitable cash transactions with positive change found.")

    # Basic numeric features
    dts = pd.to_datetime(df2['timestamp'])
    df2 = df2.copy()
    df2['hour'] = dts.dt.hour
    df2['weekday'] = dts.dt.weekday
    # monetary features
    df2['total_amount'] = df2['total_amount'].astype(float)
    # If tendered_amount exists, include it
    df2['tendered_amount'] = pd.to_numeric(df2['tendered_amount'], errors='coerce').fillna(0.0)

    filed_tender = extract_tender_features(df2)
    # Optionally include store_type encoded
    store_dummies = pd.get_dummies(df2['store_type'].fillna('unknown'), prefix='store')

    X = pd.concat([
        df2[['total_amount', 'tendered_amount', 'hour', 'weekday']].reset_index(drop=True),
        filed_tender.reset_index(drop=True),
        store_dummies.reset_index(drop=True)
    ], axis=1).fillna(0)

    y = build_targets(df2)

    return X, y, df2.index  # also return original indices (optional)
    
def train_random_forest(X, y, save_path=None, test_size=0.2, random_state=42):
    """Train a multi-output RandomForestRegressor and return model + metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    base = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    # round and clip to non-negative integers
    preds_rounded = np.clip(np.rint(preds), 0, None).astype(int)

    per_denom_mae = mean_absolute_error(y_test, preds_rounded, multioutput='raw_values')
    overall_mae = mean_absolute_error(y_test, preds_rounded)

    metrics = {
        'per_denom_mae': dict(zip(DENOMS, per_denom_mae.tolist())),
        'overall_mae': float(overall_mae)
    }

    if save_path:
        joblib.dump({'model': model, 'DENOMS': DENOMS, 'features': list(X.columns)}, save_path)

    return model, metrics, X_test, y_test, preds_rounded
