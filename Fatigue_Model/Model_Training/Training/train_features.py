# import os
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import LeaveOneGroupOut
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
# from sklearn.linear_model import RidgeCV, HuberRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# def train_stacking_model(X_df, y, groups, save_dir):
#     # Clean feature matrix
#     X_df = pd.DataFrame(X_df)
#     X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     X_df.fillna(0, inplace=True)

#     # Drop highly correlated features
#     corr = X_df.corr().abs()
#     upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#     drop_cols = [c for c in upper.columns if any(upper[c] > 0.95)]
#     if drop_cols:
#         print(f"🧹 Dropped collinear features: {drop_cols}")
#         X_df.drop(columns=drop_cols, inplace=True, errors="ignore")

#     logo = LeaveOneGroupOut()
#     fold_results = []

#     for train_idx, test_idx in logo.split(X_df, y, groups):
#         X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]

#         # Skip degenerate folds
#         if np.std(y_test) < 1e-6:
#             continue

#         # Standardize features
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test  = scaler.transform(X_test)

#         # Base models
#         base_learners = [
#             ("rf", RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)),
#             ("gb", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)),
#             ("ridge", RidgeCV(alphas=[0.1, 1, 10]))
#         ]

#         # Stacked meta learner (robust)
#         stack_model = StackingRegressor(
#             estimators=base_learners,
#             final_estimator=HuberRegressor(epsilon=1.35),
#             n_jobs=-1
#         )

#         stack_model.fit(X_train, y_train)
#         y_pred = stack_model.predict(X_test)

#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         r2 = r2_score(y_test, y_pred)

#         # --- Accuracy within tolerance margins ---
#         abs_errors = np.abs(y_pred - y_test)
#         acc_01 = np.mean(abs_errors <= 0.1)
#         acc_025 = np.mean(abs_errors <= 0.25)
#         acc_05 = np.mean(abs_errors <= 0.5)

#         subj = np.unique(groups[test_idx])[0]
#         fold_results.append({
#             "subject": subj,
#             "RMSE": rmse,
#             "R2": r2,
#             "Acc@0.1": acc_01,
#             "Acc@0.25": acc_025,
#             "Acc@0.5": acc_05
#         })

#         print(f"🧪 Subject {subj}: RMSE={rmse:.3f}, R²={r2:.3f}, "
#               f"Acc@0.1={acc_01:.3f}, Acc@0.25={acc_025:.3f}, Acc@0.5={acc_05:.3f}")

#     # Save fold results
#     df_results = pd.DataFrame(fold_results)
#     df_results.to_csv(os.path.join(save_dir, "cross_subject_results.csv"), index=False)

#     print(f"\n📄 Saved fold results → {save_dir}/cross_subject_results.csv")
#     print(f"📊 Mean RMSE={df_results.RMSE.mean():.3f}, Mean R²={df_results.R2.mean():.3f}, "
#           f"Mean Acc@0.1={df_results['Acc@0.1'].mean():.3f}, "
#           f"Mean Acc@0.25={df_results['Acc@0.25'].mean():.3f}, "
#           f"Mean Acc@0.5={df_results['Acc@0.5'].mean():.3f}")

#     # Save final model on full data
#     full_scaler = StandardScaler().fit(X_df)
#     X_scaled = full_scaler.transform(X_df)
#     final_model = StackingRegressor(
#         estimators=[
#             ("rf", RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)),
#             ("gb", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)),
#             ("ridge", RidgeCV(alphas=[0.1, 1, 10]))
#         ],
#         final_estimator=HuberRegressor(epsilon=1.35),
#         n_jobs=-1
#     )
#     final_model.fit(X_scaled, y)

#     import joblib
#     joblib.dump(final_model, os.path.join(save_dir, "stacking_model.pkl"))
#     print(f"💾 Saved stacking model → {save_dir}/stacking_model.pkl")

#     # Save feature importances
#     fi = pd.Series(
#         final_model.estimators_[0].feature_importances_,
#         index=X_df.columns
#     ).sort_values(ascending=False)
#     fi.to_csv(os.path.join(save_dir, "feature_importance.csv"))
#     print(f"📄 Saved feature importances → {save_dir}/feature_importance.csv")

import os
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.svm import SVR
from xgboost import XGBRegressor
import joblib


def preprocess_features(X: pd.DataFrame, y: np.ndarray, top_frac: float = 0.8) -> pd.DataFrame:
    """Clean, impute, and select high-information features."""
    X = X.copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X[:] = SimpleImputer(strategy="median").fit_transform(X)

    # Mutual information feature selection
    mi = mutual_info_regression(X, y)
    keep_idx = np.argsort(mi)[-int(len(mi) * top_frac):]
    X = X.iloc[:, keep_idx]
    print(f"🔍 Selected top {len(keep_idx)} informative features out of {len(mi)}")
    return X


def evaluate_fold(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    abs_err = np.abs(y_pred - y_true)
    return {
        "RMSE": rmse,
        "R2": r2,
        "Acc@0.1": np.mean(abs_err <= 0.1),
        "Acc@0.25": np.mean(abs_err <= 0.25),
        "Acc@0.5": np.mean(abs_err <= 0.5),
    }


def build_stacking_model() -> StackingRegressor:
    """Strong, diverse ensemble for regression tasks."""
    base_learners = [
        ("rf", RandomForestRegressor(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1)),
        ("xgb", XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            reg_lambda=1.0, tree_method="hist")),
        ("svr", SVR(C=10, kernel="rbf")),
        ("ridge", RidgeCV(alphas=[0.1, 1, 10]))
    ]

    meta = ElasticNetCV(l1_ratio=[.3, .5, .7, .9], alphas=[0.001, 0.01, 0.1, 1], cv=5, n_jobs=-1)

    return StackingRegressor(estimators=base_learners, final_estimator=meta, n_jobs=-1)


def train_stacking_model(X_df: pd.DataFrame, y: np.ndarray, groups: np.ndarray, save_dir: str) -> None:
    """Train & evaluate an optimized stacking ensemble using Leave-One-Group-Out CV."""
    os.makedirs(save_dir, exist_ok=True)
    X_df = preprocess_features(pd.DataFrame(X_df), y)
    logo = LeaveOneGroupOut()
    results = []

    for train_idx, test_idx in logo.split(X_df, y, groups):
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if np.std(y_test) < 1e-6:
            continue

        scaler = RobustScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        model = build_stacking_model()

        # Optional: quick randomized hyperparameter tuning
        param_dist = {
            'rf__max_depth': [8, 10, 12, None],
            'rf__n_estimators': [300, 400, 500],
            'xgb__learning_rate': [0.03, 0.05, 0.07],
            'final_estimator__l1_ratio': [0.3, 0.5, 0.7, 0.9],
            'final_estimator__alphas': [
                [0.001, 0.01, 0.1],
                [0.01, 0.1, 1.0],
                [0.1, 1.0, 10.0],
            ],
        }
        tuner = RandomizedSearchCV(model, param_distributions=param_dist,
                                   n_iter=10, scoring='r2', cv=3, n_jobs=-1)
        tuner.fit(X_train, y_train)
        best_model = tuner.best_estimator_

        y_pred = best_model.predict(X_test)
        metrics = evaluate_fold(y_test, y_pred)
        subj = np.unique(groups[test_idx])[0]
        metrics["subject"] = subj
        results.append(metrics)

        print(f"🧪 Subject {subj}: RMSE={metrics['RMSE']:.3f}, R²={metrics['R2']:.3f}, "
              f"Acc@0.1={metrics['Acc@0.1']:.3f}, Acc@0.25={metrics['Acc@0.25']:.3f}, "
              f"Acc@0.5={metrics['Acc@0.5']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "cross_subject_results.csv"), index=False)
    print(f"\n📄 Saved CV results → {save_dir}/cross_subject_results.csv")
    print(f"📊 Mean RMSE={df.RMSE.mean():.3f}, R²={df.R2.mean():.3f}")

    # Retrain final model on all data
    final_scaler = RobustScaler().fit(X_df)
    X_scaled = final_scaler.transform(X_df)
    final_model = build_stacking_model()
    final_model.fit(X_scaled, y)

    joblib.dump(final_model, os.path.join(save_dir, "stacking_model.pkl"))
    joblib.dump(final_scaler, os.path.join(save_dir, "scaler.pkl"))
    print(f"💾 Saved final model + scaler → {save_dir}")

    # Feature importances (only from tree-based base models)
    fi = pd.Series(
        final_model.estimators_[0].feature_importances_,
        index=X_df.columns
    ).sort_values(ascending=False)
    fi.to_csv(os.path.join(save_dir, "feature_importance.csv"))
    print(f"📈 Saved feature importances → {save_dir}/feature_importance.csv")