import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'app/streamlit_app.py'

def fixture():
    real=pd.DataFrame({'id':[1,2], 'execucao_id':[2,2], 'posicao':[1,2], 'time':['A','B'],
        'jogos':[1,1],'pontos':[3,0],'vitorias':[1,0],'gols_pro':[1,0],'gols_contra':[0,1],'saldo_gols':[1,-1]})
    resumo=pd.DataFrame({'id':[1,2],'execucao_id':[2,2],'time':['A','B'],'prob_titulo':[80.,20.],
        'prob_g4':[100.,100.],'prob_g6':[100.,100.],'prob_z4':[0.,0.],'posicao_media':[1.2,1.8]})
    pos=pd.DataFrame({'id':[1,2,3,4],'execucao_id':[2]*4,'time':['A','A','B','B'],
        'posicao':[1,2,1,2],'probabilidade':[80.,20.,20.,80.]})
    jogos=pd.DataFrame({'id':[1],'execucao_id':[2],'id_api':[123],'rodada':[2],
        'data_jogo':[pd.Timestamp('2026-10-01')],'mandante':['B'],'visitante':['A'],
        'prob_h':[30.],'prob_d':[20.],'prob_a':[50.],'odd_justa_h':[3.333],'odd_justa_d':[5.],'odd_justa_a':[2.]})
    hist=pd.DataFrame([dict(id=2,temporada=2026,num_simulacoes=1000,metadados=json.dumps({'seed':42,'avisos':[]}))])
    return hist,{'classificacao_atual':real,'probabilidades_times':resumo,'probabilidades_posicao':pos,'probabilidades_jogos':jogos}

class AppTests(TestCase):
    def setUp(self):
        import streamlit as st
        st.cache_data.clear()

    def test_painel_e_filtros(self):
        hist,tabelas=fixture()
        with patch('src.data.persistencia.listar_execucoes',return_value=hist), patch('src.data.persistencia.ler_resultados',return_value=(2,tabelas)):
            app=AppTest.from_file(str(APP),default_timeout=20).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.tabs),5)
            self.assertEqual(app.metric[0].value,'2')
            next(s for s in app.selectbox if s.label=='Escolha seu time').select('B')
            app.run()
            self.assertFalse(app.exception)
            next(s for s in app.selectbox if s.label=='Filtrar time').select('A')
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get('download_button')),4)

    def test_sem_publicacao(self):
        with patch('src.data.persistencia.listar_execucoes',return_value=pd.DataFrame()):
            app=AppTest.from_file(str(APP),default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.info)

    def test_banco_indisponivel(self):
        with patch('src.data.persistencia.listar_execucoes',side_effect=RuntimeError('teste')):
            app=AppTest.from_file(str(APP),default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
