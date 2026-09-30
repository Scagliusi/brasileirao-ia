"""Nomes usados no notebook, agora também compatíveis com imports src.data."""
from functools import lru_cache
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sqlalchemy import text
from src.data.models import Base
from src.data.persistencia import (obter_engine, inicializar_controle, salvar_resultados,
                                  ler_resultados, listar_execucoes,
                                  criar_execucao as _criar_execucao,
                                  salvar_classificacao_atual as _salvar_classificacao)

@lru_cache(maxsize=1)
def _engine():
    return obter_engine()

def __getattr__(nome):
    if nome == 'engine':
        return _engine()
    raise AttributeError(nome)

def create_tables():
    Base.metadata.create_all(_engine())
    inicializar_controle(_engine())

def testar_conexao():
    with _engine().connect() as conexao:
        nome = conexao.execute(text('SELECT current_database()')).scalar_one()
    print('Conectado ao banco:', nome)
    return nome

def criar_execucao(temporada, num_simulacoes, modelo, log_loss_validacao=None):
    with _engine().begin() as conexao:
        return _criar_execucao(temporada, num_simulacoes, modelo, log_loss_validacao, conexao=conexao)

def salvar_classificacao_atual(df, execucao_id):
    with _engine().begin() as conexao:
        return _salvar_classificacao(df, execucao_id, conexao=conexao)

if __name__ == '__main__':
    testar_conexao()
