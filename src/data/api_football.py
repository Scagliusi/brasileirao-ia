import os
import requests
from dotenv import load_dotenv
import pandas as pd
import json

from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[2] / ".env")




def buscar_jogos_brasileirao(ano):
    api_key = os.getenv('FOOTBALL_DATA_KEY')
    if not api_key:
        raise ValueError('Configure FOOTBALL_DATA_KEY no .env da raiz do projeto.')
    url = "https://api.football-data.org/v4/competitions/BSA/matches"

    headers = {
        "X-Auth-Token": api_key
    }

    params = {
        "season": ano
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    dados = response.json()
    # Mantém a resposta recebida para conferência; não inclui o cabeçalho de autenticação.
    pasta = Path(__file__).resolve().parents[2] / 'artifacts' / 'api'
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / f'partidas_{ano}.json').write_text(json.dumps(dados, ensure_ascii=False), encoding='utf-8')

    jogos = []

    for jogo in dados["matches"]:
        jogos.append({
            "id_api": jogo["id"],
            "ano_campeonato": ano,
            "data": jogo["utcDate"],
            "rodada": jogo["matchday"],
            "time_mandante": jogo["homeTeam"]["shortName"],
            "time_visitante": jogo["awayTeam"]["shortName"],
            "gols_mandante": jogo["score"]["fullTime"]["home"],
            "gols_visitante": jogo["score"]["fullTime"]["away"],
            "status": jogo["status"]
        })

    return pd.DataFrame(jogos)




