from src.features.build_features import features_modelo_B

def treinar_modelo(df_modelo_atual):
    X_final = df_modelo_atual.loc[
        df_modelo_atual["ano_campeonato"] >= 2012,
        features_modelo_B
    ]

    y_final = df_modelo_atual.loc[
        df_modelo_atual["ano_campeonato"] >= 2012,
        "target"
    ]
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    modelo_final = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "logistica",
            LogisticRegression(
                C=0.01,
                max_iter=1000
            )
        )
    ])

    modelo_final.fit(
        X_final,
        y_final
    )
    return modelo_final
