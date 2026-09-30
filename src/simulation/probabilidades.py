import pandas as pd
from src.simulation.monte_carlo import montar_features_jogo

COLUNAS = ['id_api', 'rodada', 'data_jogo', 'mandante', 'visitante',
           'prob_h', 'prob_d', 'prob_a', 'odd_justa_h', 'odd_justa_d', 'odd_justa_a']

def probabilidades_jogos(df_jogos_restantes, estado_inicial, estado_mando, modelo_final):
    """Previsões com o estado real atual; probabilidades em porcentagem."""
    linhas = []
    for _, jogo in df_jogos_restantes.iterrows():
        m, v = jogo['time_mandante'], jogo['time_visitante']
        x = montar_features_jogo(m, v, estado_inicial, estado_mando)
        probs = dict(zip(modelo_final.classes_, modelo_final.predict_proba(x)[0]))
        linha = dict(id_api=jogo.get('id_api'), rodada=jogo['rodada'],
                     data_jogo=jogo['data'], mandante=m, visitante=v)
        for classe in 'HDA':
            p = float(probs[classe])
            linha['prob_' + classe.lower()] = p * 100
            linha['odd_justa_' + classe.lower()] = 1 / p if p > 0 else None
        linhas.append(linha)
    return pd.DataFrame(linhas, columns=COLUNAS)
