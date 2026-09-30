"""Tabelas de resultados: criação aditiva, sem apagar o histórico existente."""
from sqlalchemy import (MetaData, Table, Column, Integer, String, Numeric,
                        DateTime, BigInteger, ForeignKey, func)

metadata = MetaData()
execucoes = Table('execucoes_modelo', metadata,
    Column('id', Integer, primary_key=True),
    Column('data_execucao', DateTime, server_default=func.now()),
    Column('temporada', Integer, nullable=False), Column('num_simulacoes', Integer),
    Column('modelo', String(100)), Column('log_loss_validacao', Numeric))

def filha(nome, colunas):
    return Table(nome, metadata, Column('id', Integer, primary_key=True),
                 Column('execucao_id', Integer, ForeignKey('execucoes_modelo.id')), *colunas)

filha('classificacao_atual', [Column('posicao', Integer), Column('time', String(100))] +
      [Column(c, Integer) for c in ['jogos','pontos','vitorias','gols_pro','gols_contra','saldo_gols']])
filha('probabilidades_times', [Column('time', String(100))] +
      [Column(c, Numeric) for c in ['prob_titulo','prob_g4','prob_g6','prob_z4','posicao_media']])
filha('probabilidades_posicao', [Column('time', String(100)), Column('posicao', Integer), Column('probabilidade', Numeric)])
filha('probabilidades_jogos', [Column('id_api', BigInteger), Column('rodada', Integer),
      Column('data_jogo', DateTime), Column('mandante', String(100)), Column('visitante', String(100))] +
      [Column(c, Numeric) for c in ['prob_h','prob_d','prob_a','odd_justa_h','odd_justa_d','odd_justa_a']])
Table('pipeline_publicacoes', metadata, Column('assinatura', String(40), primary_key=True),
      Column('execucao_id', Integer, ForeignKey('execucoes_modelo.id'), nullable=False),
      Column('metadados', String))
