import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_stacking_model(X_df, y, groups, save_dir):
    # Clean feature matrix
    X_df = pd.DataFrame(X_df)
    X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_df.fillna(0, inplace=True)

    # Drop highly correlated features
    corr = X_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > 0.95)]
    if drop_cols:
        print(f"🧹 Dropped collinear features: {drop_cols}")
        X_df.drop(columns=drop_cols, inplace=True, errors="ignore")

    logo = LeaveOneGroupOut()
    fold_results = []

    for train_idx, test_idx in logo.split(X_df, y, groups):
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Skip degenerate folds
        if np.std(y_test) < 1e-6:
            continue

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Base models
        base_learners = [
            ("rf", RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)),
            ("gb", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)),
            ("ridge", RidgeCV(alphas=[0.1, 1, 10]))
        ]

        # Stacked meta learner (robust)
        stack_model = StackingRegressor(
            estimators=base_learners,
            final_estimator=HuberRegressor(epsilon=1.35),
            n_jobs=-1
        )

        stack_model.fit(X_train, y_train)
        y_pred = stack_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # --- Accuracy within tolerance margins ---
        abs_errors = np.abs(y_pred - y_test)
        acc_01 = np.mean(abs_errors <= 0.1)
        acc_025 = np.mean(abs_errors <= 0.25)
        acc_05 = np.mean(abs_errors <= 0.5)

        subj = np.unique(groups[test_idx])[0]
        fold_results.append({
            "subject": subj,
            "RMSE": rmse,
            "R2": r2,
            "Acc@0.1": acc_01,
            "Acc@0.25": acc_025,
            "Acc@0.5": acc_05
        })

        print(f"🧪 Subject {subj}: RMSE={rmse:.3f}, R²={r2:.3f}, "
              f"Acc@0.1={acc_01:.3f}, Acc@0.25={acc_025:.3f}, Acc@0.5={acc_05:.3f}")

    # Save fold results
    df_results = pd.DataFrame(fold_results)
    df_results.to_csv(os.path.join(save_dir, "cross_subject_results.csv"), index=False)

    print(f"\n📄 Saved fold results → {save_dir}/cross_subject_results.csv")
    print(f"📊 Mean RMSE={df_results.RMSE.mean():.3f}, Mean R²={df_results.R2.mean():.3f}, "
          f"Mean Acc@0.1={df_results['Acc@0.1'].mean():.3f}, "
          f"Mean Acc@0.25={df_results['Acc@0.25'].mean():.3f}, "
          f"Mean Acc@0.5={df_results['Acc@0.5'].mean():.3f}")

    # Save final model on full data
    full_scaler = StandardScaler().fit(X_df)
    X_scaled = full_scaler.transform(X_df)
    final_model = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)),
            ("gb", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)),
            ("ridge", RidgeCV(alphas=[0.1, 1, 10]))
        ],
        final_estimator=HuberRegressor(epsilon=1.35),
        n_jobs=-1
    )
    final_model.fit(X_scaled, y)

    import joblib
    joblib.dump(final_model, os.path.join(save_dir, "stacking_model.pkl"))
    print(f"💾 Saved stacking model → {save_dir}/stacking_model.pkl")

    # Save feature importances
    fi = pd.Series(
        final_model.estimators_[0].feature_importances_,
        index=X_df.columns
    ).sort_values(ascending=False)
    fi.to_csv(os.path.join(save_dir, "feature_importance.csv"))
    print(f"📄 Saved feature importances → {save_dir}/feature_importance.csv")