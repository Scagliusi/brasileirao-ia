import pandas as pd
import numpy as np
from src.data.api_football import buscar_jogos_brasileirao

ANOS_API = (2024, 2025, 2026)
ANO_ATUAL = 2026
colunas_base = [
    "ano_campeonato", "data", "rodada", "time_mandante",
    "time_visitante", "gols_mandante", "gols_visitante"
]
mapa_times = {
    "Atlético-PR": "Athletico-PR", "Goiás EC": "Goiás", "Santos FC": "Santos",
    "Bahia": "EC Bahia", "Bragantino": "RB Bragantino",
    "Ceará": "Ceará SC", "Mineiro": "Atlético-MG",
    "Recife": "Sport Recife", "Vitória": "EC Vitória",
    "Coritiba": "Coritiba FC", "Paranaense": "Athletico-PR",
}
def padroniza_jogos(dados, *, origem):
    obrigatorias = colunas_base + (["status"] if origem == "api" else [])
    faltantes = sorted(set(obrigatorias) - set(dados.columns))
    if faltantes:
        raise ValueError(f"Colunas ausentes em {origem}: {faltantes}")
    jogos = dados.copy()
    for coluna in ["time_mandante", "time_visitante"]:
        jogos[coluna] = jogos[coluna].astype("string").str.strip().replace(mapa_times)
        jogos[coluna] = jogos[coluna].replace("", pd.NA)
    if origem == "api":
        jogos["data"] = (pd.to_datetime(jogos["data"], utc=True, errors="raise")
                         .dt.tz_convert("America/Sao_Paulo").dt.tz_localize(None))
        jogos["status"] = jogos["status"].astype("string").str.strip().str.upper()
        if jogos["status"].isna().any() or jogos["status"].eq("").any():
            raise ValueError("A API retornou jogos sem status.")
    else:
        jogos["data"] = pd.to_datetime(jogos["data"], errors="raise")
    for coluna in ["ano_campeonato", "rodada", "gols_mandante", "gols_visitante"]:
        jogos[coluna] = pd.to_numeric(jogos[coluna], errors="raise").astype("Int64")
    return jogos


def valida_realizados(jogos):
    if jogos[colunas_base].isna().any().any():
        raise ValueError("Há jogos encerrados com campos obrigatórios ausentes.")
    if (jogos[["gols_mandante", "gols_visitante"]] < 0).any().any():
        raise ValueError("Há placares negativos.")
    if jogos["time_mandante"].eq(jogos["time_visitante"]).any():
        raise ValueError("Um time aparece enfrentando a si mesmo.")
    # Identidade do registro histórico. Inconsistências de mando em anos antigos
    # são sinalizadas abaixo, sem apagar placares/datas distintos do CSV.
    chave = ["ano_campeonato", "data", "rodada", "time_mandante", "time_visitante"]
    duplicados = jogos.duplicated(chave, keep=False)
    if duplicados.any():
        raise ValueError(f"Partidas duplicadas: {jogos.loc[duplicados, chave].to_dict('records')}")

def atualizar_dados(caminho_csv, ano_atual=2026, buscar=buscar_jogos_brasileirao):
    if ano_atual < 2024:
        raise ValueError('Este fluxo usa CSV até 2023 e API a partir de 2024.')
    anos_api = tuple(range(2024, ano_atual + 1))
    df_clean = pd.read_csv(caminho_csv).dropna(subset=["gols_mandante", "gols_visitante"])
    df_historico_ate_2023 = padroniza_jogos(
        df_clean.loc[df_clean["ano_campeonato"] < min(anos_api), colunas_base],
        origem="csv"
    )
    valida_realizados(df_historico_ate_2023)

    api_por_ano = {}
    for ano in anos_api:
        jogos = padroniza_jogos(buscar(ano), origem="api")
        if jogos.empty:
            raise ValueError(f"A API retornou uma temporada vazia: {ano}.")
        if jogos["ano_campeonato"].isna().any() or not jogos["ano_campeonato"].eq(ano).all():
            raise ValueError(f"A resposta da API contém uma temporada diferente de {ano}.")
        if "id_api" in jogos and jogos["id_api"].dropna().duplicated().any():
            raise ValueError(f"A API retornou identificadores duplicados em {ano}.")
        api_por_ano[ano] = jogos

    realizados_por_ano = {}
    for ano, jogos in api_por_ano.items():
        realizados = jogos.loc[jogos["status"].eq("FINISHED"), colunas_base].copy()
        valida_realizados(realizados)
        realizados_por_ano[ano] = realizados

    # Mantém os nomes usados anteriormente no notebook.


    df_jogos_restantes = api_por_ano[ano_atual].loc[
        ~api_por_ano[ano_atual]["status"].eq("FINISHED")
    ].copy()
    chave = ['ano_campeonato', 'time_mandante', 'time_visitante']
    if api_por_ano[ano_atual].duplicated(chave).any():
        raise ValueError('Calendário com confrontos duplicados; confira a resposta da API.')
    df_completo = pd.concat(
        [df_historico_ate_2023] + list(realizados_por_ano.values()),
        ignore_index=True
    ).sort_values(["data", "ano_campeonato", "rodada"], kind="stable").reset_index(drop=True)
    valida_realizados(df_completo)
    chave_mando = ['ano_campeonato','time_mandante','time_visitante']
    anomalias = df_historico_ate_2023.loc[df_historico_ate_2023.duplicated(chave_mando, keep=False)]
    avisos = []
    if not anomalias.empty:
        anos = ', '.join(map(str, sorted(anomalias['ano_campeonato'].unique())))
        avisos.append(f'O CSV histórico contém {len(anomalias)} registros com confrontos de mesmo mando repetidos em {anos}, mas datas/rodadas distintas. Foram preservados como no notebook; a fonte histórica precisa de revisão.')
    df_completo.attrs['avisos'] = avisos
    return (df_completo, df_jogos_restantes)

# Compatibilidade com a organização simples do notebook.
preparar_dados = atualizar_dados
