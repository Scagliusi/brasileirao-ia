import pandas as pd
def calcula_elo_atual(df, K=18, alpha=0.75):
    jogos = df.sort_values(["data", "rodada"]).copy()

    elos = {}
    ano_anterior = None

    for _, row in jogos.iterrows():
        ano_atual = row["ano_campeonato"]

        # regressão à média quando muda a temporada
        if ano_anterior is not None and ano_atual != ano_anterior:
            for time in elos:
                elos[time] = 1500 + alpha * (elos[time] - 1500)

        mandante = row["time_mandante"]
        visitante = row["time_visitante"]

        if mandante not in elos:
            elos[mandante] = 1500

        if visitante not in elos:
            elos[visitante] = 1500

        elo_m = elos[mandante]
        elo_v = elos[visitante]

        esperado_m = 1 / (1 + 10 ** ((elo_v - elo_m) / 400))
        esperado_v = 1 - esperado_m

        if row["gols_mandante"] > row["gols_visitante"]:
            resultado_m = 1
            resultado_v = 0

        elif row["gols_mandante"] < row["gols_visitante"]:
            resultado_m = 0
            resultado_v = 1

        else:
            resultado_m = 0.5
            resultado_v = 0.5

        elos[mandante] = elo_m + K * (resultado_m - esperado_m)
        elos[visitante] = elo_v + K * (resultado_v - esperado_v)

        ano_anterior = ano_atual

    return elos

def preparar_estados(df_completo, df_times_completo, ano_atual=2026):
    estado_tabela = (
        df_times_completo[
            df_times_completo["ano_campeonato"] == ano_atual
        ]
        .groupby("time")
        .agg(
            jogos=("time", "size"),
            pontos=("pontos", "sum"),
            vitorias=("vitoria", "sum"),
            gols_pro=("gols_marcados", "sum"),
            gols_contra=("gols_sofridos", "sum")
        )
        .reset_index()
    )

    estado_tabela["saldo_gols"] = (
        estado_tabela["gols_pro"]
        - estado_tabela["gols_contra"]
    )

    estado_tabela = estado_tabela.sort_values(
        ["pontos", "vitorias", "saldo_gols", "gols_pro"],
        ascending=False
    ).reset_index(drop=True)

    estado_tabela
    estado_ewm = (
        df_times_completo
        .sort_values(["time", "data", "rodada"])
        .copy()
    )

    grupo_time = estado_ewm.groupby("time")

    estado_ewm["pontos_ewm_atual"] = grupo_time["pontos"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean()
    )

    estado_ewm["gols_ewm_atual"] = grupo_time["gols_marcados"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean()
    )

    estado_ewm["gols_sofridos_ewm_atual"] = grupo_time["gols_sofridos"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean()
    )

    estado_ewm["vitoria_ewm_atual"] = grupo_time["vitoria"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean()
    )
    estado_ewm_atual = (
        estado_ewm
        .groupby("time")
        .tail(1)[
            [
                "time",
                "pontos_ewm_atual",
                "gols_ewm_atual",
                "gols_sofridos_ewm_atual",
                "vitoria_ewm_atual"
            ]
        ]
        .reset_index(drop=True)
    )
    times_atuais = estado_tabela["time"]

    estado_ewm_atual = estado_ewm_atual[
        estado_ewm_atual["time"].isin(times_atuais)
    ].reset_index(drop=True)
    estado_ewm_mando = (
        df_times_completo
        .sort_values(["time", "casa", "data", "rodada"])
        .copy()
    )

    grupo_mando = estado_ewm_mando.groupby(["time", "casa"])

    estado_ewm_mando["pontos_ewm_mando_atual"] = grupo_mando["pontos"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean()
    )

    estado_ewm_mando["gols_ewm_mando_atual"] = grupo_mando["gols_marcados"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean()
    )

    estado_ewm_mando["gols_sofridos_ewm_mando_atual"] = grupo_mando["gols_sofridos"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean()
    )

    estado_ewm_mando["vitoria_ewm_mando_atual"] = grupo_mando["vitoria"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean()
    )
    estado_ewm_mando_atual = (
        estado_ewm_mando
        .groupby(["time", "casa"])
        .tail(1)[
            [
                "time",
                "casa",
                "pontos_ewm_mando_atual",
                "gols_ewm_mando_atual",
                "gols_sofridos_ewm_mando_atual",
                "vitoria_ewm_mando_atual"
            ]
        ]
    )
    estado_ewm_mando_atual = estado_ewm_mando_atual[
        estado_ewm_mando_atual["time"].isin(estado_tabela["time"])
    ].reset_index(drop=True)
    elos_atuais = calcula_elo_atual(df_completo)
    estado_elo_atual = (
        pd.DataFrame(
            elos_atuais.items(),
            columns=["time", "elo_atual"]
        )
        .query("time in @estado_tabela.time")
        .reset_index(drop=True)
    )
    estado_inicial = (
        estado_tabela
        .merge(
            estado_ewm_atual,
            on="time",
            how="left"
        )
        .merge(
            estado_elo_atual,
            on="time",
            how="left"
        )
    )

    estado_inicial
    return (estado_tabela, estado_inicial, estado_ewm_mando_atual)
