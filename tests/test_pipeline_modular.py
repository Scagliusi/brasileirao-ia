import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from src.data.preparar_dados import atualizar_dados
from src.features.build_features import construir_features, features_modelo_B
from src.simulation.estados import preparar_estados
from src.simulation.monte_carlo import sortear_placar
from src.data.persistencia import inicializar_controle, salvar_resultados
from src.cache import salvar, assinatura
from pipeline import treinar, simular


def jogos(ano):
    times = ['Bahia', 'Flamengo', 'Santos', 'Goiás']
    linhas = []
    for i, (m, v) in enumerate((m, v) for m in times for v in times if m != v):
        linhas.append(dict(id_api=ano*100+i, ano_campeonato=ano,
                           data=f'{ano}-05-{i+1:02d}T20:00:00Z', rodada=i+1,
                           time_mandante=m, time_visitante=v, gols_mandante=i%3,
                           gols_visitante=(i//3)%3, status='FINISHED'))
    return pd.DataFrame(linhas)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pasta = Path(self.tmp.name)
        csv = self.pasta/'historico.csv'
        historico = pd.concat([jogos(a) for a in range(2012, 2024)])
        historico['data'] = pd.to_datetime(historico['data'], utc=True).dt.tz_localize(None)
        historico.to_csv(csv, index=False)
        def api(ano):
            df = jogos(ano)
            if ano == 2026:
                df.loc[df.index[-2:], ['gols_mandante', 'gols_visitante']] = np.nan
                df.loc[df.index[-2:], 'status'] = 'TIMED'
            return df
        completo, restantes = atualizar_dados(csv, buscar=api)
        times, modelo = construir_features(completo)
        tabela, estado, mando = preparar_estados(completo, times)
        self.dados = dict(temporada=2026, df_completo=completo, df_modelo_atual=modelo,
                          df_jogos_restantes=restantes, estado_tabela=tabela,
                          estado_inicial=estado, estado_ewm_mando_atual=mando)
        self.dados['assinatura'] = assinatura(self.dados)
        salvar('dados', self.dados, self.pasta)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fluxo_cache_e_probabilidades(self):
        self.assertEqual(len(features_modelo_B), 27)
        self.assertIn('EC Bahia', set(self.dados['estado_inicial']['time']))
        antes = self.dados['estado_inicial'].copy(deep=True)
        treinar(self.pasta)
        resultado = simular(3, 42, self.pasta)
        self.assertEqual(len(resultado['df_mc']), 12)
        self.assertTrue(np.allclose(resultado['prob_posicao'].sum(axis=1), 100))
        self.assertTrue(np.allclose(resultado['probabilidades_jogos'][['prob_h','prob_d','prob_a']].sum(axis=1), 100))
        pd.testing.assert_frame_equal(antes, self.dados['estado_inicial'])
        with patch('src.simulation.lote.simular_lote', side_effect=AssertionError('Recalculou')):
            outro = simular(3, 42, self.pasta)
        pd.testing.assert_frame_equal(resultado['df_mc'], outro['df_mc'])
        self.dados['assinatura'] = 'novos-dados'
        salvar('dados', self.dados, self.pasta)
        with self.assertRaisesRegex(ValueError, 'mudaram'):
            simular(3, 42, self.pasta)

    def test_sem_vazamento(self):
        original = self.dados['df_completo'].copy()
        _, a = construir_features(original)
        original.loc[original.index[-1], 'gols_mandante'] = 30
        _, b = construir_features(original)
        pd.testing.assert_frame_equal(a[features_modelo_B], b[features_modelo_B])

    def test_placar_impossivel(self):
        with self.assertRaises(ValueError):
            sortear_placar('H', 0, 1, np.random.default_rng(42))

    def test_lote_preserva_uma_trajetoria_original(self):
        from src.simulation.lote import simular_lote
        from src.simulation.monte_carlo import simular_temporada
        modelo=treinar(self.pasta)['modelo_final']
        args=(self.dados['df_jogos_restantes'],self.dados['estado_inicial'],
              self.dados['estado_ewm_mando_atual'],modelo)
        for seed in [1,42,137]:
            original,_=simular_temporada(*args,np.random.default_rng(seed))
            lote=simular_lote(*args,n_simulacoes=1,seed=seed)
            pd.testing.assert_frame_equal(original[['time','posicao','pontos']].reset_index(drop=True),
                                          lote[['time','posicao','pontos']].reset_index(drop=True),check_dtype=False)

    def test_publicacao_rejeita_probabilidades_invalidas(self):
        from src.data.persistencia import validar_resultado
        treinar(self.pasta)
        r=simular(2,42,self.pasta)
        r['prob_posicao'].iloc[0,0]=101
        with self.assertRaises(ValueError):
            validar_resultado(r)

    def test_persistencia_atomica_e_repeticao(self):
        treinar(self.pasta)
        resultado = simular(1, 42, self.pasta)
        engine = create_engine('sqlite://')
        with engine.begin() as con:
            con.execute(text('CREATE TABLE execucoes_modelo (id INTEGER PRIMARY KEY, '
                             'temporada INTEGER, num_simulacoes INTEGER, modelo TEXT, log_loss_validacao REAL)'))
        inicializar_controle(engine)
        primeiro = salvar_resultados(resultado, engine=engine)
        self.assertEqual(primeiro, salvar_resultados(resultado, engine=engine))
        with engine.connect() as con:
            self.assertEqual(con.execute(text('SELECT COUNT(*) FROM execucoes_modelo')).scalar(), 1)
            self.assertEqual(con.execute(text('SELECT COUNT(*) FROM classificacao_atual')).scalar(), 4)
        outro = dict(resultado, assinatura='a'*40)
        with patch.object(pd.DataFrame, 'to_sql', side_effect=RuntimeError('falha simulada')):
            with self.assertRaises(RuntimeError):
                salvar_resultados(outro, engine=engine)
        with engine.connect() as con:
            self.assertEqual(con.execute(text('SELECT COUNT(*) FROM execucoes_modelo')).scalar(), 1)


if __name__ == '__main__':
    unittest.main()
