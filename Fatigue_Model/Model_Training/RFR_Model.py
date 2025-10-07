from sklearn.ensemble import RandomForestRegressor

class RFRModel:
    def build_rfr_model(n_estimators=200, max_depth=None, random_state=42):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        return model