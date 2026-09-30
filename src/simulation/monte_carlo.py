import numpy as np
import pandas as pd
from src.features.build_features import features_modelo_B

def montar_features_jogo(
    mandante,
    visitante,
    estado_simulacao,
    estado_mando_simulacao
):

    # Estado geral dos dois times
    dados_mandante = estado_simulacao[
        estado_simulacao["time"] == mandante
    ].iloc[0]

    dados_visitante = estado_simulacao[
        estado_simulacao["time"] == visitante
    ].iloc[0]


    # Estado específico de mando
    # mandante joga em casa -> casa = 1
    mando_mandante = estado_mando_simulacao[
        (estado_mando_simulacao["time"] == mandante) &
        (estado_mando_simulacao["casa"] == 1)
    ].iloc[0]

    # visitante joga fora -> casa = 0
    mando_visitante = estado_mando_simulacao[
        (estado_mando_simulacao["time"] == visitante) &
        (estado_mando_simulacao["casa"] == 0)
    ].iloc[0]


    # Montar features
    features = {

        # ELO
        "elo_mandante_sazonal":
            dados_mandante["elo_atual"],

        "elo_visitante_sazonal":
            dados_visitante["elo_atual"],

        "elo_sazonal_diff":
            dados_mandante["elo_atual"]
            - dados_visitante["elo_atual"],


        # EWM geral mandante
        "pontos_ewm_mandante":
            dados_mandante["pontos_ewm_atual"],

        "gols_ewm_mandante":
            dados_mandante["gols_ewm_atual"],

        "gols_sofridos_ewm_mandante":
            dados_mandante["gols_sofridos_ewm_atual"],

        "vitoria_ewm_mandante":
            dados_mandante["vitoria_ewm_atual"],


        # EWM geral visitante
        "pontos_ewm_visitante":
            dados_visitante["pontos_ewm_atual"],

        "gols_ewm_visitante":
            dados_visitante["gols_ewm_atual"],

        "gols_sofridos_ewm_visitante":
            dados_visitante["gols_sofridos_ewm_atual"],

        "vitoria_ewm_visitante":
            dados_visitante["vitoria_ewm_atual"],


        # EWM mando mandante
        "pontos_ewm_mando_mandante":
            mando_mandante["pontos_ewm_mando_atual"],

        "gols_ewm_mando_mandante":
            mando_mandante["gols_ewm_mando_atual"],

        "gols_sofridos_ewm_mando_mandante":
            mando_mandante["gols_sofridos_ewm_mando_atual"],

        "vitoria_ewm_mando_mandante":
            mando_mandante["vitoria_ewm_mando_atual"],


        # EWM mando visitante
        "pontos_ewm_mando_visitante":
            mando_visitante["pontos_ewm_mando_atual"],

        "gols_ewm_mando_visitante":
            mando_visitante["gols_ewm_mando_atual"],

        "gols_sofridos_ewm_mando_visitante":
            mando_visitante["gols_sofridos_ewm_mando_atual"],

        "vitoria_ewm_mando_visitante":
            mando_visitante["vitoria_ewm_mando_atual"],
    }

    # Diferenças gerais
    features["pontos_ewm_diff"] = (
        features["pontos_ewm_mandante"]
        - features["pontos_ewm_visitante"]
    )

    features["gols_ewm_diff"] = (
        features["gols_ewm_mandante"]
        - features["gols_ewm_visitante"]
    )

    features["gols_sofridos_ewm_diff"] = (
        features["gols_sofridos_ewm_mandante"]
        - features["gols_sofridos_ewm_visitante"]
    )

    features["vitoria_ewm_diff"] = (
        features["vitoria_ewm_mandante"]
        - features["vitoria_ewm_visitante"]
    )


    # Diferenças por mando
    features["pontos_ewm_mando_diff"] = (
        features["pontos_ewm_mando_mandante"]
        - features["pontos_ewm_mando_visitante"]
    )

    features["gols_ewm_mando_diff"] = (
        features["gols_ewm_mando_mandante"]
        - features["gols_ewm_mando_visitante"]
    )

    features["gols_sofridos_ewm_mando_diff"] = (
        features["gols_sofridos_ewm_mando_mandante"]
        - features["gols_sofridos_ewm_mando_visitante"]
    )

    features["vitoria_ewm_mando_diff"] = (
        features["vitoria_ewm_mando_mandante"]
        - features["vitoria_ewm_mando_visitante"]
    )


    # vira DataFrame com 1 linha
    X_jogo = pd.DataFrame(
        [features],
        columns=features_modelo_B
    )

    return X_jogo

def sortear_resultado(classes, probabilidades, rng):
    return rng.choice(
        classes,
        p=probabilidades
    )

def calcular_lambdas(
    mandante,
    visitante,
    estado_simulacao,
    estado_mando_simulacao
):

    # dados gerais
    dados_m = estado_simulacao[
        estado_simulacao["time"] == mandante
    ].iloc[0]

    dados_v = estado_simulacao[
        estado_simulacao["time"] == visitante
    ].iloc[0]

    # dados por mando
    mando_m = estado_mando_simulacao[
        (estado_mando_simulacao["time"] == mandante) &
        (estado_mando_simulacao["casa"] == 1)
    ].iloc[0]

    mando_v = estado_mando_simulacao[
        (estado_mando_simulacao["time"] == visitante) &
        (estado_mando_simulacao["casa"] == 0)
    ].iloc[0]

    # ataque recente
    ataque_m = (
        dados_m["gols_ewm_atual"]
        + mando_m["gols_ewm_mando_atual"]
    ) / 2

    ataque_v = (
        dados_v["gols_ewm_atual"]
        + mando_v["gols_ewm_mando_atual"]
    ) / 2

    # defesa adversária
    defesa_v = (
        dados_v["gols_sofridos_ewm_atual"]
        + mando_v["gols_sofridos_ewm_mando_atual"]
    ) / 2

    defesa_m = (
        dados_m["gols_sofridos_ewm_atual"]
        + mando_m["gols_sofridos_ewm_mando_atual"]
    ) / 2

    # gols esperados
    lambda_m = (ataque_m + defesa_v) / 2
    lambda_v = (ataque_v + defesa_m) / 2

    return lambda_m, lambda_v

def sortear_placar(resultado, lambda_m, lambda_v, rng):

    if resultado not in {'H', 'D', 'A'}:
        raise ValueError('Resultado inválido.')
    if not np.isfinite([lambda_m, lambda_v]).all() or min(lambda_m, lambda_v) < 0:
        raise ValueError('Médias de gols inválidas; confira os estados.')
    if (resultado == 'H' and lambda_m == 0) or (resultado == 'A' and lambda_v == 0):
        raise ValueError('Placar sorteado impossível com média de gols zero.')
    for _ in range(100000):

        gols_m = rng.poisson(lambda_m)
        gols_v = rng.poisson(lambda_v)

        # vitória mandante
        if resultado == "H" and gols_m > gols_v:
            return gols_m, gols_v

        # empate
        if resultado == "D" and gols_m == gols_v:
            return gols_m, gols_v

        # vitória visitante
        if resultado == "A" and gols_m < gols_v:
            return gols_m, gols_v
    raise RuntimeError("Não foi possível sortear placar compatível em 100000 tentativas.")

def atualizar_estado_jogo(
    mandante,
    visitante,
    gols_m,
    gols_v,
    estado_simulacao,
    estado_mando_simulacao,
    span=5,
    K=18
):
    # alpha usado pela EWM(span=5, adjust=False)
    alpha = 2 / (span + 1)

    # localizar os dois times no estado geral
    idx_m = estado_simulacao[
        estado_simulacao["time"] == mandante
    ].index[0]

    idx_v = estado_simulacao[
        estado_simulacao["time"] == visitante
    ].index[0]

    # --------------------------------
    # 1. Resultado: pontos e vitória
    # --------------------------------

    if gols_m > gols_v:
        pontos_m, pontos_v = 3, 0
        vitoria_m, vitoria_v = 1, 0

        resultado_elo_m = 1
        resultado_elo_v = 0

    elif gols_m < gols_v:
        pontos_m, pontos_v = 0, 3
        vitoria_m, vitoria_v = 0, 1

        resultado_elo_m = 0
        resultado_elo_v = 1

    else:
        pontos_m, pontos_v = 1, 1
        vitoria_m, vitoria_v = 0, 0

        resultado_elo_m = 0.5
        resultado_elo_v = 0.5


    # --------------------------------
    # 2. Atualizar classificação
    # --------------------------------

    estado_simulacao.at[idx_m, "jogos"] += 1
    estado_simulacao.at[idx_v, "jogos"] += 1

    estado_simulacao.at[idx_m, "pontos"] += pontos_m
    estado_simulacao.at[idx_v, "pontos"] += pontos_v

    estado_simulacao.at[idx_m, "vitorias"] += vitoria_m
    estado_simulacao.at[idx_v, "vitorias"] += vitoria_v

    estado_simulacao.at[idx_m, "gols_pro"] += gols_m
    estado_simulacao.at[idx_m, "gols_contra"] += gols_v

    estado_simulacao.at[idx_v, "gols_pro"] += gols_v
    estado_simulacao.at[idx_v, "gols_contra"] += gols_m

    estado_simulacao.at[idx_m, "saldo_gols"] = (
        estado_simulacao.at[idx_m, "gols_pro"]
        - estado_simulacao.at[idx_m, "gols_contra"]
    )

    estado_simulacao.at[idx_v, "saldo_gols"] = (
        estado_simulacao.at[idx_v, "gols_pro"]
        - estado_simulacao.at[idx_v, "gols_contra"]
    )


    # --------------------------------
    # 3. Atualizar EWM geral
    # nova_ewm = alpha * valor_novo
    #          + (1-alpha) * ewm_anterior
    # --------------------------------

    def atualizar_ewm(valor_antigo, valor_novo):
        return (
            alpha * valor_novo
            + (1 - alpha) * valor_antigo
        )

    # mandante
    estado_simulacao.at[idx_m, "pontos_ewm_atual"] = atualizar_ewm(
        estado_simulacao.at[idx_m, "pontos_ewm_atual"],
        pontos_m
    )

    estado_simulacao.at[idx_m, "gols_ewm_atual"] = atualizar_ewm(
        estado_simulacao.at[idx_m, "gols_ewm_atual"],
        gols_m
    )

    estado_simulacao.at[idx_m, "gols_sofridos_ewm_atual"] = atualizar_ewm(
        estado_simulacao.at[idx_m, "gols_sofridos_ewm_atual"],
        gols_v
    )

    estado_simulacao.at[idx_m, "vitoria_ewm_atual"] = atualizar_ewm(
        estado_simulacao.at[idx_m, "vitoria_ewm_atual"],
        vitoria_m
    )

    # visitante
    estado_simulacao.at[idx_v, "pontos_ewm_atual"] = atualizar_ewm(
        estado_simulacao.at[idx_v, "pontos_ewm_atual"],
        pontos_v
    )

    estado_simulacao.at[idx_v, "gols_ewm_atual"] = atualizar_ewm(
        estado_simulacao.at[idx_v, "gols_ewm_atual"],
        gols_v
    )

    estado_simulacao.at[idx_v, "gols_sofridos_ewm_atual"] = atualizar_ewm(
        estado_simulacao.at[idx_v, "gols_sofridos_ewm_atual"],
        gols_m
    )

    estado_simulacao.at[idx_v, "vitoria_ewm_atual"] = atualizar_ewm(
        estado_simulacao.at[idx_v, "vitoria_ewm_atual"],
        vitoria_v
    )


    # --------------------------------
    # 4. Atualizar EWM por mando
    # --------------------------------

    idx_mando_m = estado_mando_simulacao[
        (estado_mando_simulacao["time"] == mandante) &
        (estado_mando_simulacao["casa"] == 1)
    ].index[0]

    idx_mando_v = estado_mando_simulacao[
        (estado_mando_simulacao["time"] == visitante) &
        (estado_mando_simulacao["casa"] == 0)
    ].index[0]

    # mandante em casa
    estado_mando_simulacao.at[
        idx_mando_m, "pontos_ewm_mando_atual"
    ] = atualizar_ewm(
        estado_mando_simulacao.at[
            idx_mando_m, "pontos_ewm_mando_atual"
        ],
        pontos_m
    )

    estado_mando_simulacao.at[
        idx_mando_m, "gols_ewm_mando_atual"
    ] = atualizar_ewm(
        estado_mando_simulacao.at[
            idx_mando_m, "gols_ewm_mando_atual"
        ],
        gols_m
    )

    estado_mando_simulacao.at[
        idx_mando_m, "gols_sofridos_ewm_mando_atual"
    ] = atualizar_ewm(
        estado_mando_simulacao.at[
            idx_mando_m, "gols_sofridos_ewm_mando_atual"
        ],
        gols_v
    )

    estado_mando_simulacao.at[
        idx_mando_m, "vitoria_ewm_mando_atual"
    ] = atualizar_ewm(
        estado_mando_simulacao.at[
            idx_mando_m, "vitoria_ewm_mando_atual"
        ],
        vitoria_m
    )

    # visitante fora
    estado_mando_simulacao.at[
        idx_mando_v, "pontos_ewm_mando_atual"
    ] = atualizar_ewm(
        estado_mando_simulacao.at[
            idx_mando_v, "pontos_ewm_mando_atual"
        ],
        pontos_v
    )

    estado_mando_simulacao.at[
        idx_mando_v, "gols_ewm_mando_atual"
    ] = atualizar_ewm(
        estado_mando_simulacao.at[
            idx_mando_v, "gols_ewm_mando_atual"
        ],
        gols_v
    )

    estado_mando_simulacao.at[
        idx_mando_v, "gols_sofridos_ewm_mando_atual"
    ] = atualizar_ewm(
        estado_mando_simulacao.at[
            idx_mando_v, "gols_sofridos_ewm_mando_atual"
        ],
        gols_m
    )

    estado_mando_simulacao.at[
        idx_mando_v, "vitoria_ewm_mando_atual"
    ] = atualizar_ewm(
        estado_mando_simulacao.at[
            idx_mando_v, "vitoria_ewm_mando_atual"
        ],
        vitoria_v
    )


    # --------------------------------
    # 5. Atualizar Elo
    # --------------------------------

    elo_m = estado_simulacao.at[idx_m, "elo_atual"]
    elo_v = estado_simulacao.at[idx_v, "elo_atual"]

    esperado_m = 1 / (
        1 + 10 ** ((elo_v - elo_m) / 400)
    )

    esperado_v = 1 - esperado_m

    novo_elo_m = (
        elo_m
        + K * (resultado_elo_m - esperado_m)
    )

    novo_elo_v = (
        elo_v
        + K * (resultado_elo_v - esperado_v)
    )

    estado_simulacao.at[idx_m, "elo_atual"] = novo_elo_m
    estado_simulacao.at[idx_v, "elo_atual"] = novo_elo_v

    return estado_simulacao, estado_mando_simulacao

def simular_temporada(
    jogos_restantes,
    estado_inicial,
    estado_mando_inicial,
    modelo,
    rng
):

    # Cada temporada começa da realidade atual
    estado_simulacao = estado_inicial.copy()
    estado_mando_simulacao = estado_mando_inicial.copy()

    resultados_jogos = []

    # ordem dos jogos
    jogos = jogos_restantes.sort_values(
        ["data", "rodada"]
    ).copy()

    for _, jogo in jogos.iterrows():

        mandante = jogo["time_mandante"]
        visitante = jogo["time_visitante"]

        # 1. Montar as 27 features NO ESTADO ATUAL
        X_jogo = montar_features_jogo(
            mandante,
            visitante,
            estado_simulacao,
            estado_mando_simulacao
        )

        # 2. Probabilidades do modelo
        proba = modelo.predict_proba(X_jogo)[0]

        classes = modelo.named_steps[
            "logistica"
        ].classes_

        # 3. Sortear H / D / A
        resultado = sortear_resultado(
            classes,
            proba,
            rng
        )

        # 4. Calcular médias esperadas de gols
        lambda_m, lambda_v = calcular_lambdas(
            mandante,
            visitante,
            estado_simulacao,
            estado_mando_simulacao
        )

        # 5. Sortear placar compatível com H/D/A
        gols_m, gols_v = sortear_placar(
            resultado,
            lambda_m,
            lambda_v,
            rng
        )

        # 6. Guardar resultado
        resultados_jogos.append({
            "rodada": jogo["rodada"],
            "mandante": mandante,
            "visitante": visitante,
            "gols_mandante": gols_m,
            "gols_visitante": gols_v,
            "resultado": resultado
        })

        # 7. Atualizar o mundo simulado
        estado_simulacao, estado_mando_simulacao = (
            atualizar_estado_jogo(
                mandante,
                visitante,
                gols_m,
                gols_v,
                estado_simulacao,
                estado_mando_simulacao
            )
        )

    # classificação ao fim da temporada
    classificacao = estado_simulacao.sort_values(
        [
            "pontos",
            "vitorias",
            "saldo_gols",
            "gols_pro"
        ],
        ascending=False
    ).reset_index(drop=True)

    classificacao["posicao"] = (
        classificacao.index + 1
    )

    resultados_jogos = pd.DataFrame(
        resultados_jogos
    )

    return classificacao, resultados_jogos

def rodar_monte_carlo(df_jogos_restantes, estado_inicial, estado_ewm_mando_atual, modelo_final, n_simulacoes=1000, seed=42):
    if n_simulacoes < 1:
        raise ValueError("n_simulacoes deve ser positivo")

    resultados_mc = []

    rng = np.random.default_rng(seed)

    for simulacao in range(n_simulacoes):

        classificacao, _ = simular_temporada(
            df_jogos_restantes,
            estado_inicial,
            estado_ewm_mando_atual,
            modelo_final,
            rng
        )

        for _, linha in classificacao.iterrows():

            resultados_mc.append({
                "simulacao": simulacao,
                "time": linha["time"],
                "posicao": linha["posicao"],
                "pontos": linha["pontos"]
            })

        # só para acompanhar
        if (simulacao + 1) % 100 == 0:
            print(f"{simulacao + 1}/{n_simulacoes}")
    df_mc = pd.DataFrame(resultados_mc)
    return df_mc

def resumir_monte_carlo(df_mc, n_simulacoes):
    prob_titulo = (
        df_mc[df_mc["posicao"] == 1]
        .groupby("time")
        .size()
        .div(n_simulacoes)
        .mul(100)
        .sort_values(ascending=False)
    )

    prob_titulo
    prob_g4 = (
        df_mc[df_mc["posicao"] <= 4]
        .groupby("time")
        .size()
        .div(n_simulacoes)
        .mul(100)
        .sort_values(ascending=False)
    )

    prob_g4
    prob_g6 = (
        df_mc[df_mc["posicao"] <= 6]
        .groupby("time")
        .size()
        .div(n_simulacoes)
        .mul(100)
        .sort_values(ascending=False)
    )
    prob_g6
    prob_z4 = (
        df_mc[df_mc["posicao"] >= 17]
        .groupby("time")
        .size()
        .div(n_simulacoes)
        .mul(100)
        .sort_values(ascending=False)
    )

    prob_z4
    # Probabilidade de cada time terminar em cada posição
    prob_posicao = pd.crosstab(
        df_mc["time"],
        df_mc["posicao"]
    )

    prob_posicao = prob_posicao.reindex(
        columns=range(1, 21),
        fill_value=0
    )

    prob_posicao = (
        prob_posicao
        .div(n_simulacoes)
        .mul(100)
    )
    from scipy.optimize import linear_sum_assignment

    matriz = prob_posicao.to_numpy()

    # negativo porque a função minimiza;
    # queremos maximizar as probabilidades
    linhas, colunas = linear_sum_assignment(-matriz)

    tabela_projetada = pd.DataFrame({
        "time": prob_posicao.index[linhas],
        "posicao": prob_posicao.columns[colunas],
        "probabilidade_%": matriz[linhas, colunas]
    })

    tabela_projetada = (
        tabela_projetada
        .sort_values("posicao")
        .reset_index(drop=True)
    )

    tabela_projetada["probabilidade_%"] = (
        tabela_projetada["probabilidade_%"]
        .round(1)
    )

    tabela_projetada

    resumo_mc = pd.DataFrame({'prob_titulo': prob_titulo, 'prob_g4': prob_g4, 'prob_g6': prob_g6, 'prob_z4': prob_z4}).reindex(prob_posicao.index).fillna(0)
    resumo_mc['posicao_media'] = df_mc.groupby('time')['posicao'].mean()
    resumo_mc = resumo_mc.reset_index()
    return (resumo_mc, prob_posicao, tabela_projetada)
