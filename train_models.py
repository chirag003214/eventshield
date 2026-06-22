"""
EventShield — Model Training & Validation
Reproduces all three models and their cross-validated metrics.

Run: python train_models.py
Outputs: models.pkl, metrics.json

All metrics reported are 5-fold cross-validated out-of-fold scores.
"""
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (average_precision_score, f1_score,
                             mean_absolute_error, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


def build_models(df):
    """Train all three EventShield models in-process and return (models, metrics)."""
    df = df.copy()
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
    event_df = df.dropna(subset=['resolution_min', 'hour', 'dow']).copy()
    event_df = event_df[(event_df['resolution_min'] >= 1) & (event_df['resolution_min'] < 50000)]
    event_df['is_long'] = (event_df['resolution_min'] > 120).astype(int)

    le_cause, le_corr = LabelEncoder(), LabelEncoder()
    le_stat, le_veh = LabelEncoder(), LabelEncoder()
    event_df['cause_enc'] = le_cause.fit_transform(event_df['event_cause'])
    event_df['corr_enc'] = le_corr.fit_transform(event_df['corridor'].fillna('Unknown'))
    event_df['stat_enc'] = le_stat.fit_transform(event_df['police_station'].fillna('Unknown'))
    event_df['veh_enc'] = le_veh.fit_transform(event_df['veh_type'].fillna('Unknown'))

    feats = ['hour', 'dow', 'is_weekend', 'is_night', 'cause_enc',
             'corr_enc', 'stat_enc', 'veh_enc', 'latitude', 'longitude']
    X_clf = event_df[feats].values

    # ── Model 1: Prolonged-event classifier (>2h) ──
    y_long = event_df['is_long'].values
    oof_long = cross_val_predict(
        GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
        X_clf, y_long, cv=kf, method='predict_proba')[:, 1]
    clf_long = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, random_state=42).fit(X_clf, y_long)
    metrics['long_duration'] = {
        'roc_auc': round(roc_auc_score(y_long, oof_long), 4),
        'pr_auc': round(average_precision_score(y_long, oof_long), 4),
        'f1': round(f1_score(y_long, (oof_long > 0.5).astype(int)), 4),
        'positive_rate': round(float(y_long.mean()), 4), 'n': int(len(y_long))}

    # ── Model 2: High-impact classifier ──
    high_causes = ['accident', 'protest', 'vip_movement', 'public_event', 'procession']
    event_df['high_impact'] = (
        (event_df['requires_road_closure']) |
        (event_df['is_long'] == 1) |
        (event_df['event_cause'].isin(high_causes))
    ).astype(int)
    y_high = event_df['high_impact'].values
    oof_high = cross_val_predict(
        GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
        X_clf, y_high, cv=kf, method='predict_proba')[:, 1]
    clf_hi = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, random_state=42).fit(X_clf, y_high)
    metrics['high_impact'] = {
        'roc_auc': round(roc_auc_score(y_high, oof_high), 4),
        'pr_auc': round(average_precision_score(y_high, oof_high), 4),
        'f1': round(f1_score(y_high, (oof_high > 0.5).astype(int)), 4),
        'positive_rate': round(float(y_high.mean()), 4), 'n': int(len(y_high)),
        'feature_importance': [(f, round(float(i), 4))
                               for f, i in sorted(zip(feats, clf_hi.feature_importances_),
                                                  key=lambda x: -x[1])]}

    # ── Model 3: Spatial-temporal hotspot forecaster ──
    df['hour_bucket'] = pd.cut(df['hour'], bins=[0, 6, 10, 16, 20, 24],
                               labels=['night', 'morning', 'midday', 'evening', 'lateeve'],
                               include_lowest=True)
    station_window_agg = df.dropna(subset=['hour', 'police_station']).groupby(
        ['police_station', 'dow', 'hour_bucket']).size().reset_index(name='event_count')
    le_ps, le_hb = LabelEncoder(), LabelEncoder()
    station_window_agg['ps_enc'] = le_ps.fit_transform(station_window_agg['police_station'])
    station_window_agg['hb_enc'] = le_hb.fit_transform(
        station_window_agg['hour_bucket'].astype(str))
    X_forecast = station_window_agg[['ps_enc', 'dow', 'hb_enc']].values
    y_count = station_window_agg['event_count'].values
    y_count_log = np.log1p(y_count)
    oof_forecast = np.expm1(cross_val_predict(
        GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
        X_forecast, y_count_log, cv=kf))
    reg_count = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, random_state=42).fit(X_forecast, y_count_log)
    base_mae = mean_absolute_error(y_count, np.full_like(y_count, y_count.mean(), dtype=float))
    forecast_mae = mean_absolute_error(y_count, oof_forecast)
    metrics['event_forecast'] = {
        'mae': round(forecast_mae, 3), 'r2': round(r2_score(y_count, oof_forecast), 4),
        'baseline_mae': round(base_mae, 3),
        'improvement_pct': round((1 - forecast_mae / base_mae) * 100, 1),
        'n': int(len(y_count))}

    models = {'clf_long': clf_long, 'clf_hi': clf_hi, 'reg_count': reg_count,
              'le_cause': le_cause, 'le_corr': le_corr, 'le_stat': le_stat,
              'le_veh': le_veh, 'le_ps': le_ps, 'le_hb': le_hb, 'feats': feats}
    return models, metrics


def main():
    """CLI entrypoint: train models and persist models.pkl + metrics.json for offline inspection."""
    df = pd.read_csv('event_data.csv')
    models, metrics = build_models(df)
    pickle.dump(models, open('models.pkl', 'wb'))
    json.dump(metrics, open('metrics.json', 'w'), indent=2)

    print("=== Cross-Validated Metrics ===")
    print(json.dumps(metrics, indent=2))
    print("\nSaved: models.pkl, metrics.json")


if __name__ == '__main__':
    main()
