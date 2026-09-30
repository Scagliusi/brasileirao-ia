"""Persistência transacional nas tabelas já existentes do projeto."""
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from src.config import RAIZ

TABELAS = ('classificacao_atual', 'probabilidades_times',
           'probabilidades_posicao', 'probabilidades_jogos')


def obter_engine():
    url = os.getenv('DATABASE_URL')
    if not url:
        raise ValueError(f'Configure DATABASE_URL no arquivo {RAIZ / ".env"}.')
    return create_engine(url, pool_pre_ping=True)


def inicializar_controle(engine):
    """Tabela auxiliar: não altera nem apaga execuções anteriores."""
    from src.data.schema import metadata
    metadata.create_all(engine)
    with engine.begin() as con:
        if 'metadados' not in {c['name'] for c in inspect(con).get_columns('pipeline_publicacoes')}:
            con.execute(text('ALTER TABLE pipeline_publicacoes ADD COLUMN metadados TEXT'))


def validar_resultado(r):
    """Não publica saídas incompletas ou probabilidades inconsistentes."""
    times = set(r['tabela_real']['time'])
    if not times or r['tabela_real']['time'].duplicated().any():
        raise ValueError('Classificação vazia ou com times duplicados.')
    if r['n_simulacoes'] < 1 or len(r['df_mc']) != len(times) * r['n_simulacoes']:
        raise ValueError('Quantidade de resultados do Monte Carlo inconsistente.')
    if set(r['resumo_mc']['time']) != times or set(r['prob_posicao'].index) != times:
        raise ValueError('Times ausentes nos resumos.')
    if r['resumo_mc']['time'].duplicated().any() or r['prob_posicao'].index.duplicated().any():
        raise ValueError('Times duplicados nos resumos.')
    for df in [r['resumo_mc'][['prob_titulo','prob_g4','prob_g6','prob_z4']], r['prob_posicao'],
               r['probabilidades_jogos'][['prob_h','prob_d','prob_a']]]:
        valores = df.to_numpy(dtype=float)
        if not np.isfinite(valores).all() or ((valores < 0) | (valores > 100)).any():
            raise ValueError('Probabilidades precisam estar entre 0 e 100 e ser finitas.')
    if not np.allclose(r['prob_posicao'].sum(axis=1), 100):
        raise ValueError('Probabilidades por posição não somam 100%.')
    if not np.isclose(r['resumo_mc']['prob_titulo'].sum(),100):
        raise ValueError('Probabilidades de título não somam 100%.')
    jogos = r['probabilidades_jogos']
    if not np.allclose(jogos[['prob_h','prob_d','prob_a']].sum(axis=1), 100):
        raise ValueError('Probabilidades por jogo não somam 100%.')
    if jogos.duplicated(['mandante','visitante']).any():
        raise ValueError('Jogos duplicados nas previsões.')
    if not (set(jogos['mandante']) | set(jogos['visitante'])) <= times:
        raise ValueError('Previsões com times fora da classificação.')


def criar_execucao(temporada, num_simulacoes, modelo, log_loss_validacao=None, *, conexao):
    return conexao.execute(text('''INSERT INTO execucoes_modelo
        (temporada, num_simulacoes, modelo, log_loss_validacao)
        VALUES (:temporada, :num_simulacoes, :modelo, :log_loss_validacao)
        RETURNING id'''), dict(temporada=temporada, num_simulacoes=num_simulacoes,
                              modelo=modelo, log_loss_validacao=log_loss_validacao)).scalar_one()


def salvar_classificacao_atual(df, execucao_id, *, conexao):
    colunas = ['posicao', 'time', 'jogos', 'pontos', 'vitorias',
               'gols_pro', 'gols_contra', 'saldo_gols']
    df[colunas].assign(execucao_id=execucao_id).to_sql(
        'classificacao_atual', conexao, if_exists='append', index=False)


def salvar_resultados(resultado, *, engine=None):
    validar_resultado(resultado)
    engine = engine if engine is not None else obter_engine()
    chave = resultado['assinatura']
    with engine.begin() as con:
        # Serializa publicações iguais inclusive entre processos diferentes.
        if con.dialect.name == 'postgresql':
            con.execute(text('SELECT pg_advisory_xact_lock(:chave)'),
                        {'chave': int(chave[:15], 16)})
        existente = con.execute(text('SELECT execucao_id FROM pipeline_publicacoes '
                                     'WHERE assinatura=:chave'), {'chave': chave}).scalar()
        if existente is not None:
            return existente
        execucao_id = criar_execucao(resultado['temporada'], resultado['n_simulacoes'],
                                    'Logistic Regression B', conexao=con)
        salvar_classificacao_atual(resultado['tabela_real'], execucao_id, conexao=con)
        posicoes = resultado['prob_posicao'].rename_axis(index='time', columns='posicao')
        posicoes = posicoes.stack().rename('probabilidade').reset_index()
        for tabela, df in [('probabilidades_times', resultado['resumo_mc']),
                          ('probabilidades_posicao', posicoes),
                          ('probabilidades_jogos', resultado['probabilidades_jogos'])]:
            df.assign(execucao_id=execucao_id).to_sql(tabela, con, if_exists='append', index=False)
        meta = {k: resultado.get(k) for k in ['seed','ultima_partida','gerado_em','avisos','jogos_realizados','jogos_restantes','dados_assinatura','codigo']}
        con.execute(text('INSERT INTO pipeline_publicacoes (assinatura, execucao_id, metadados) '
                         'VALUES (:chave, :id, :meta)'), {'chave': chave, 'id': execucao_id,
                                                       'meta': json.dumps(meta, ensure_ascii=False, default=str)})
    return execucao_id


def listar_execucoes(*, engine=None):
    engine = engine if engine is not None else obter_engine()
    with engine.connect() as con:
        return pd.read_sql(text('''SELECT e.*, p.metadados FROM execucoes_modelo e
            JOIN pipeline_publicacoes p ON p.execucao_id=e.id ORDER BY e.id DESC'''), con)


def ler_resultados(execucao_id=None, *, engine=None):
    engine = engine if engine is not None else obter_engine()
    with engine.connect() as con:
        if execucao_id is None:
            execucao_id = con.execute(text('SELECT MAX(execucao_id) FROM pipeline_publicacoes')).scalar()
        else:
            execucao_id = con.execute(text('SELECT execucao_id FROM pipeline_publicacoes WHERE execucao_id=:id'),
                                      {'id': int(execucao_id)}).scalar()
        if execucao_id is None:
            return None, {}
        tabelas = {t: pd.read_sql(text(f'SELECT * FROM {t} WHERE execucao_id=:id'), con,
                                 params={'id': execucao_id}) for t in TABELAS}
        return execucao_id, tabelas
