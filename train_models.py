"""
EventShield — Model Training & Validation
Reproduces all three models and their cross-validated metrics.

Run: python train_models.py
Outputs: models.pkl, metrics.json

All metrics reported are 5-fold cross-validated out-of-fold scores.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score,
                             f1_score, average_precision_score)
from sklearn.preprocessing import LabelEncoder
import pickle, json
import warnings
warnings.filterwarnings('ignore')


def main():
    df = pd.read_csv('event_data.csv')
    df['start_dt'] = pd.to_datetime(df['start_datetime'], errors='coerce', utc=True)
    df['closed_dt'] = pd.to_datetime(df['closed_datetime'], errors='coerce', utc=True)
    df['hour'] = df['start_dt'].dt.hour
    df['dow'] = df['start_dt'].dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 21) | (df['hour'] <= 5)).astype(int)
    df['resolution_min'] = (df['closed_dt'] - df['start_dt']).dt.total_seconds() / 60
    df.loc[df['resolution_min'] <= 0, 'resolution_min'] = np.nan

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {}

    # ── Shared feature prep for classifiers ──
    res = df.dropna(subset=['resolution_min', 'hour', 'dow']).copy()
    res = res[(res['resolution_min'] >= 1) & (res['resolution_min'] < 50000)]
    res['is_long'] = (res['resolution_min'] > 120).astype(int)

    le_cause, le_corr = LabelEncoder(), LabelEncoder()
    le_stat, le_veh = LabelEncoder(), LabelEncoder()
    res['cause_enc'] = le_cause.fit_transform(res['event_cause'])
    res['corr_enc'] = le_corr.fit_transform(res['corridor'].fillna('Unknown'))
    res['stat_enc'] = le_stat.fit_transform(res['police_station'].fillna('Unknown'))
    res['veh_enc'] = le_veh.fit_transform(res['veh_type'].fillna('Unknown'))

    feats = ['hour', 'dow', 'is_weekend', 'is_night', 'cause_enc',
             'corr_enc', 'stat_enc', 'veh_enc', 'latitude', 'longitude']
    X = res[feats].values

    # ── Model 1: Prolonged-event classifier (>2h) ──
    y_long = res['is_long'].values
    oof = cross_val_predict(GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
                            X, y_long, cv=kf, method='predict_proba')[:, 1]
    clf_long = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42).fit(X, y_long)
    metrics['long_duration'] = {
        'roc_auc': round(roc_auc_score(y_long, oof), 4),
        'pr_auc': round(average_precision_score(y_long, oof), 4),
        'f1': round(f1_score(y_long, (oof > 0.5).astype(int)), 4),
        'positive_rate': round(float(y_long.mean()), 4), 'n': int(len(y_long))}

    # ── Model 2: High-impact classifier ──
    high_causes = ['accident', 'protest', 'vip_movement', 'public_event', 'procession']
    res['high_impact'] = ((res['requires_road_closure']) | (res['is_long'] == 1) |
                          (res['event_cause'].isin(high_causes))).astype(int)
    y_hi = res['high_impact'].values
    oof_hi = cross_val_predict(GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
                               X, y_hi, cv=kf, method='predict_proba')[:, 1]
    clf_hi = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42).fit(X, y_hi)
    metrics['high_impact'] = {
        'roc_auc': round(roc_auc_score(y_hi, oof_hi), 4),
        'pr_auc': round(average_precision_score(y_hi, oof_hi), 4),
        'f1': round(f1_score(y_hi, (oof_hi > 0.5).astype(int)), 4),
        'positive_rate': round(float(y_hi.mean()), 4), 'n': int(len(y_hi)),
        'feature_importance': [(f, round(float(i), 4))
                               for f, i in sorted(zip(feats, clf_hi.feature_importances_),
                                                  key=lambda x: -x[1])]}

    # ── Model 3: Spatial-temporal hotspot forecaster ──
    df['hour_bucket'] = pd.cut(df['hour'], bins=[0, 6, 10, 16, 20, 24],
                               labels=['night', 'morning', 'midday', 'evening', 'lateeve'],
                               include_lowest=True)
    agg = df.dropna(subset=['hour', 'police_station']).groupby(
        ['police_station', 'dow', 'hour_bucket']).size().reset_index(name='event_count')
    le_ps, le_hb = LabelEncoder(), LabelEncoder()
    agg['ps_enc'] = le_ps.fit_transform(agg['police_station'])
    agg['hb_enc'] = le_hb.fit_transform(agg['hour_bucket'].astype(str))
    Xa = agg[['ps_enc', 'dow', 'hb_enc']].values
    ya = agg['event_count'].values
    ya_log = np.log1p(ya)
    oof_c = np.expm1(cross_val_predict(GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
                                       Xa, ya_log, cv=kf))
    reg_count = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42).fit(Xa, ya_log)
    base_mae = mean_absolute_error(ya, np.full_like(ya, ya.mean(), dtype=float))
    fc_mae = mean_absolute_error(ya, oof_c)
    metrics['event_forecast'] = {
        'mae': round(fc_mae, 3), 'r2': round(r2_score(ya, oof_c), 4),
        'baseline_mae': round(base_mae, 3),
        'improvement_pct': round((1 - fc_mae / base_mae) * 100, 1), 'n': int(len(ya))}

    # ── Persist ──
    models = {'clf_long': clf_long, 'clf_hi': clf_hi, 'reg_count': reg_count,
              'le_cause': le_cause, 'le_corr': le_corr, 'le_stat': le_stat,
              'le_veh': le_veh, 'le_ps': le_ps, 'le_hb': le_hb, 'feats': feats}
    pickle.dump(models, open('models.pkl', 'wb'))
    json.dump(metrics, open('metrics.json', 'w'), indent=2)

    print("=== Cross-Validated Metrics ===")
    print(json.dumps(metrics, indent=2))
    print("\nSaved: models.pkl, metrics.json")


if __name__ == '__main__':
    main()
