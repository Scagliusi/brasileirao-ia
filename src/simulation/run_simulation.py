"""Mantém o nome usado na organização do notebook."""
from src.simulation.monte_carlo import rodar_monte_carlo, resumir_monte_carlo

def rodar_simulacao(df_jogos_restantes, estado_inicial, estado_ewm_mando_atual,
                   modelo_final, n_simulacoes=1000, seed=42):
    df_mc = rodar_monte_carlo(df_jogos_restantes, estado_inicial,
                            estado_ewm_mando_atual, modelo_final, n_simulacoes, seed)
    resumo, prob_posicao, tabela_projetada = resumir_monte_carlo(df_mc, n_simulacoes)
    r = resumo.set_index('time')
    return (df_mc, r.prob_titulo, r.prob_g4, r.prob_g6, r.prob_z4, prob_posicao, tabela_projetada)
