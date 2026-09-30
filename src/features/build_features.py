import numpy as np
import pandas as pd
features_modelo_B = [
    # Elo sazonal
    "elo_mandante_sazonal",
    "elo_visitante_sazonal",
    "elo_sazonal_diff",

    # EWM geral
    "pontos_ewm_mandante",
    "gols_ewm_mandante",
    "gols_sofridos_ewm_mandante",
    "vitoria_ewm_mandante",

    "pontos_ewm_visitante",
    "gols_ewm_visitante",
    "gols_sofridos_ewm_visitante",
    "vitoria_ewm_visitante",

    # EWM por mando
    "pontos_ewm_mando_mandante",
    "gols_ewm_mando_mandante",
    "gols_sofridos_ewm_mando_mandante",
    "vitoria_ewm_mando_mandante",

    "pontos_ewm_mando_visitante",
    "gols_ewm_mando_visitante",
    "gols_sofridos_ewm_mando_visitante",
    "vitoria_ewm_mando_visitante",

    # Diferenças
    "pontos_ewm_diff",
    "gols_ewm_diff",
    "gols_sofridos_ewm_diff",
    "vitoria_ewm_diff",

    "pontos_ewm_mando_diff",
    "gols_ewm_mando_diff",
    "gols_sofridos_ewm_mando_diff",
    "vitoria_ewm_mando_diff"
]
def cria_df_times(df_jogos):

    mandante = pd.DataFrame({
        "ano_campeonato": df_jogos["ano_campeonato"],
        "data": df_jogos["data"],
        "rodada": df_jogos["rodada"],

        "time": df_jogos["time_mandante"],
        "adversario": df_jogos["time_visitante"],

        "casa": 1,

        "gols_marcados": df_jogos["gols_mandante"],
        "gols_sofridos": df_jogos["gols_visitante"]
    })

    visitante = pd.DataFrame({
        "ano_campeonato": df_jogos["ano_campeonato"],
        "data": df_jogos["data"],
        "rodada": df_jogos["rodada"],

        "time": df_jogos["time_visitante"],
        "adversario": df_jogos["time_mandante"],

        "casa": 0,

        "gols_marcados": df_jogos["gols_visitante"],
        "gols_sofridos": df_jogos["gols_mandante"]
    })

    df_times = pd.concat(
        [mandante, visitante],
        ignore_index=True
    )

    df_times["pontos"] = np.select(
        [
            df_times["gols_marcados"] > df_times["gols_sofridos"],
            df_times["gols_marcados"] == df_times["gols_sofridos"]
        ],
        [
            3,
            1
        ],
        default=0
    )

    df_times["saldo_gols"] = (
        df_times["gols_marcados"]
        - df_times["gols_sofridos"]
    )

    df_times["vitoria"] = (
        df_times["gols_marcados"]
        > df_times["gols_sofridos"]
    ).astype(int)

    df_times = df_times.sort_values(
        ["time", "data", "rodada"]
    ).reset_index(drop=True)

    return df_times
def cria_df_modelo_B(df_times, df_jogos):

    df_modelo = df_jogos.copy()

    colunas_ewm = [
        "pontos_ewm",
        "gols_ewm",
        "gols_sofridos_ewm",
        "vitoria_ewm",
        "pontos_ewm_mando",
        "gols_ewm_mando",
        "gols_sofridos_ewm_mando",
        "vitoria_ewm_mando"
    ]

    # dados do mandante
    mandante = df_times[
        df_times["casa"] == 1
    ][
        [
            "ano_campeonato",
            "data",
            "rodada",
            "time", "adversario"
        ] + colunas_ewm
    ].copy()

    mandante = mandante.rename(
        columns={
            "time": "time_mandante",
            "adversario": "time_visitante",
            **{
                col: f"{col}_mandante"
                for col in colunas_ewm
            }
        }
    )

    # dados do visitante
    visitante = df_times[
        df_times["casa"] == 0
    ][
        [
            "ano_campeonato",
            "data",
            "rodada",
            "time", "adversario"
        ] + colunas_ewm
    ].copy()

    visitante = visitante.rename(
        columns={
            "time": "time_visitante",
            "adversario": "time_mandante",
            **{
                col: f"{col}_visitante"
                for col in colunas_ewm
            }
        }
    )

    # junta mandante
    df_modelo = df_modelo.merge(
        mandante,
        on=[
            "ano_campeonato",
            "data",
            "rodada",
            "time_mandante", "time_visitante"
        ],
        how="left", validate="one_to_one"
    )

    # junta visitante
    df_modelo = df_modelo.merge(
        visitante,
        on=[
            "ano_campeonato",
            "data",
            "rodada",
            "time_visitante", "time_mandante"
        ],
        how="left", validate="one_to_one"
    )

    return df_modelo
def cria_elo_sazonal(df, K=18, alpha=0.75):

    df = df.sort_values(["data", "rodada"]).copy()

    elos = {}
    ano_anterior = None

    for i, row in df.iterrows():

        ano_atual = row["ano_campeonato"]

        # Quando muda a temporada, aproxima todos os Elos de 1500
        if ano_anterior is not None and ano_atual != ano_anterior:

            for time in elos:
                elos[time] = 1500 + alpha * (elos[time] - 1500)

        mandante = row["time_mandante"]
        visitante = row["time_visitante"]

        if mandante not in elos:
            elos[mandante] = 1500

        if visitante not in elos:
            elos[visitante] = 1500

        elo_mandante = elos[mandante]
        elo_visitante = elos[visitante]

        # Elo ANTES do jogo
        df.loc[i, "elo_mandante_sazonal"] = elo_mandante
        df.loc[i, "elo_visitante_sazonal"] = elo_visitante

        # Expectativa
        E_mandante = 1 / (
            1 + 10 ** ((elo_visitante - elo_mandante) / 400)
        )

        E_visitante = 1 - E_mandante

        # Resultado real
        if row["gols_mandante"] > row["gols_visitante"]:
            S_mandante = 1
            S_visitante = 0

        elif row["gols_mandante"] == row["gols_visitante"]:
            S_mandante = 0.5
            S_visitante = 0.5

        else:
            S_mandante = 0
            S_visitante = 1

        # Atualiza depois da partida
        elos[mandante] = (
            elo_mandante +
            K * (S_mandante - E_mandante)
        )

        elos[visitante] = (
            elo_visitante +
            K * (S_visitante - E_visitante)
        )

        ano_anterior = ano_atual

    return df

def construir_features(df_completo):
    df_times_completo = cria_df_times(df_completo)
    df_times_completo = df_times_completo.sort_values(
        ["time", "data", "rodada"]
    ).copy()

    grupo_time = df_times_completo.groupby("time")

    df_times_completo["pontos_ewm"] = grupo_time["pontos"].transform(
        lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
    )

    df_times_completo["gols_ewm"] = grupo_time["gols_marcados"].transform(
        lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
    )

    df_times_completo["gols_sofridos_ewm"] = grupo_time["gols_sofridos"].transform(
        lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
    )

    df_times_completo["vitoria_ewm"] = grupo_time["vitoria"].transform(
        lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
    )
    grupo_mando = df_times_completo.groupby(["time", "casa"])

    df_times_completo["pontos_ewm_mando"] = grupo_mando["pontos"].transform(
        lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
    )

    df_times_completo["gols_ewm_mando"] = grupo_mando["gols_marcados"].transform(
        lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
    )

    df_times_completo["gols_sofridos_ewm_mando"] = grupo_mando["gols_sofridos"].transform(
        lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
    )

    df_times_completo["vitoria_ewm_mando"] = grupo_mando["vitoria"].transform(
        lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
    )
    df_modelo_atual = cria_df_modelo_B(
        df_times_completo,
        df_completo.copy()
    )

    df_modelo_atual = cria_elo_sazonal(
        df_modelo_atual,
        K=18,
        alpha=0.75
    )
    df_modelo_atual["elo_sazonal_diff"] = (
        df_modelo_atual["elo_mandante_sazonal"] - df_modelo_atual["elo_visitante_sazonal"]
    )
    for nome in ["pontos", "gols", "gols_sofridos", "vitoria"]:
        for sufixo in ["ewm", "ewm_mando"]:
            base = f"{nome}_{sufixo}"
            df_modelo_atual[f"{base}_diff"] = (
                df_modelo_atual[f"{base}_mandante"] - df_modelo_atual[f"{base}_visitante"]
            )
    df_modelo_atual["target"] = np.select(
        [df_modelo_atual["gols_mandante"] > df_modelo_atual["gols_visitante"],
         df_modelo_atual["gols_mandante"] == df_modelo_atual["gols_visitante"]],
        ["H", "D"], default="A"
    )
    return (df_times_completo, df_modelo_atual)
