"""Painel de consulta; nunca executa treinamento ou simulação ao abrir."""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import altair as alt
import streamlit as st
from scipy.optimize import linear_sum_assignment
from src.data.persistencia import listar_execucoes, ler_resultados

st.set_page_config(page_title='Brasileirão IA', page_icon='⚽', layout='wide')
st.title('Brasileirão IA')
st.caption('Classificação real e projeções com base nos resultados disponíveis.')

@st.cache_data(ttl=60)
def execucoes():
    return listar_execucoes()

@st.cache_data(ttl=60)
def consultar(identificador):
    return ler_resultados(identificador)

def baixar(df, nome):
    st.download_button('Baixar tabela (CSV)', df.to_csv(index=False).encode('utf-8-sig'),
                       nome+'.csv', 'text/csv', key=nome)

with st.sidebar:
    st.header('Consulta')
    if st.button('Recarregar resultados', width='stretch'):
        st.cache_data.clear()
    st.caption('Consulta com cache de até 60 segundos. O botão força uma nova leitura.')
try:
    historico=execucoes()
except Exception:
    st.error('Não foi possível consultar o PostgreSQL. Confira se o serviço está ligado e a configuração DATABASE_URL no .env.')
    st.info('Na primeira utilização, execute atualizar_resultados.cmd na pasta do projeto.')
    st.stop()
if historico.empty:
    st.info('Ainda não há uma execução completa publicada. Execute atualizar_resultados.cmd.')
    st.caption('Execuções antigas incompletas permanecem no banco, mas não aparecem como uma projeção completa.')
    st.stop()

def rotulo(identificador):
    r=historico.loc[historico['id'].eq(identificador)].iloc[0]
    return f"Execução {int(identificador)} · {int(r['temporada'])} · {int(r['num_simulacoes'])} simulações"

with st.sidebar:
    escolhido=st.selectbox('Execução publicada',historico['id'].astype(int).tolist(),format_func=rotulo)
try:
    execucao_id,tabelas=consultar(escolhido)
except Exception:
    st.error('Falha ao ler as tabelas desta execução. Recarregue ou verifique a conexão.')
    st.stop()
if execucao_id is None:
    st.warning('Execução indisponível. Recarregue a consulta.')
    st.stop()
registro=historico.loc[historico['id'].eq(escolhido)].iloc[0]
try:
    meta=json.loads(registro['metadados'] or '{}')
except (TypeError,ValueError):
    meta={}
real=tabelas['classificacao_atual'].sort_values('posicao').drop(columns=['id','execucao_id'])
resumo=tabelas['probabilidades_times'].drop(columns=['id','execucao_id'])
posicoes=tabelas['probabilidades_posicao'].drop(columns=['id','execucao_id'])
jogos=tabelas['probabilidades_jogos'].drop(columns=['id','execucao_id'])
if real.empty or resumo.empty or posicoes.empty:
    st.error('Execução incompleta. Publique novamente pelo pipeline.')
    st.stop()
st.caption(f"Temporada {int(registro['temporada'])} · execução {execucao_id} · última partida na base: {meta.get('ultima_partida') or 'não informada'}")
for aviso in meta.get('avisos') or []:
    st.warning(aviso)
a,b,c,d=st.columns(4)
a.metric('Times',len(real))
b.metric('Jogos realizados',meta.get('jogos_realizados') or int(real['jogos'].sum()/2))
c.metric('Jogos restantes',len(jogos))
d.metric('Temporadas simuladas',str(int(registro['num_simulacoes'])))
visao,clubes,distribuicao,confrontos,metodo=st.tabs(['Campeonato','Meu time','Posições','Próximos jogos','Como foi calculado'])
with visao:
    left,right=st.columns(2)
    with left:
        st.subheader('Classificação atual')
        st.dataframe(real,hide_index=True,width='stretch')
        baixar(real,'classificacao_'+str(execucao_id))
    with right:
        st.subheader('Chance de título')
        st.altair_chart(alt.Chart(resumo).mark_bar(color='#168567').encode(
            x=alt.X('prob_titulo:Q',title='Probabilidade (%)',scale=alt.Scale(domain=[0,100])),
            y=alt.Y('time:N',sort='-x',title=None),
            tooltip=['time',alt.Tooltip('prob_titulo:Q',format='.1f')]),width='stretch')
    st.subheader('Probabilidades por time')
    st.caption('Percentuais de 0 a 100. G4 e G6 são posições na tabela, não garantias de vagas em competições continentais.')
    st.dataframe(resumo.sort_values('prob_titulo',ascending=False),hide_index=True,width='stretch',
        column_config={c:st.column_config.NumberColumn(t,format='%.1f%%') for c,t in
                       [('prob_titulo','Título'),('prob_g4','G4'),('prob_g6','G6'),('prob_z4','Z4')]})
    baixar(resumo,'times_'+str(execucao_id))
with clubes:
    time=st.selectbox('Escolha seu time',sorted(real['time'].tolist()))
    r=resumo.loc[resumo['time'].eq(time)].iloc[0]
    cols=st.columns(4)
    for col,k,t in zip(cols,['prob_titulo','prob_g4','prob_g6','prob_z4'],['Título','G4','G6','Z4']):
        col.metric(t,f'{float(r[k]):.1f}%')
    st.caption(f"Posição média simulada: {float(r['posicao_media']):.1f}. É uma média, não uma posição garantida.")
    dist=posicoes.loc[posicoes['time'].eq(time)].sort_values('posicao')
    st.altair_chart(alt.Chart(dist).mark_bar(color='#168567').encode(
        x=alt.X('posicao:O',title='Posição final'),y=alt.Y('probabilidade:Q',title='Probabilidade (%)'),
        tooltip=['posicao',alt.Tooltip('probabilidade:Q',format='.1f')]),width='stretch')
    st.subheader('Partidas restantes do time')
    st.dataframe(jogos.loc[jogos['mandante'].eq(time)|jogos['visitante'].eq(time)].sort_values(['rodada','data_jogo']),hide_index=True,width='stretch')
with distribuicao:
    st.subheader('Chance em cada posição')
    matriz=posicoes.pivot(index='time',columns='posicao',values='probabilidade').sort_index(axis=1)
    st.altair_chart(alt.Chart(posicoes).mark_rect().encode(
        x=alt.X('posicao:O',title='Posição'),y=alt.Y('time:N',title=None),
        color=alt.Color('probabilidade:Q',scale=alt.Scale(scheme='greens',domain=[0,100]),title='%'),
        tooltip=['time','posicao',alt.Tooltip('probabilidade:Q',format='.1f')]).properties(height=520),width='stretch')
    st.dataframe(matriz.round(1),width='stretch')
    baixar(posicoes,'posicoes_'+str(execucao_id))
    st.subheader('Projeção com posições distintas')
    st.caption('Maximiza a soma das probabilidades individuais ao atribuir uma posição a cada time. Não é uma probabilidade conjunta da tabela inteira.')
    linhas,colunas=linear_sum_assignment(-matriz.to_numpy(dtype=float))
    proj=pd.DataFrame({'Posição':matriz.columns[colunas],'Time':matriz.index[linhas],
                      'Chance nessa posição (%)':matriz.to_numpy()[linhas,colunas]}).sort_values('Posição')
    st.dataframe(proj,hide_index=True,width='stretch')
with confrontos:
    st.subheader('Previsões dos jogos restantes')
    st.caption('Usam o estado real atual, sem atualizar times entre estes confrontos. Datas da API podem mudar, especialmente em partidas adiadas.')
    if jogos.empty:
        st.info('Não há jogos restantes nesta execução.')
    else:
        filtros=st.columns(2)
        clube=filtros[0].selectbox('Filtrar time',['Todos']+sorted(real['time'].tolist()))
        rodada=filtros[1].selectbox('Filtrar rodada',['Todas']+sorted(jogos['rodada'].dropna().astype(int).unique().tolist()))
        vista=jogos.copy()
        if clube!='Todos':vista=vista.loc[vista['mandante'].eq(clube)|vista['visitante'].eq(clube)]
        if rodada!='Todas':vista=vista.loc[vista['rodada'].eq(rodada)]
        vista=vista.sort_values(['rodada','data_jogo'])
        st.dataframe(vista,hide_index=True,width='stretch',column_config={
            'prob_h':st.column_config.NumberColumn('Mandante (%)',format='%.1f'),
            'prob_d':st.column_config.NumberColumn('Empate (%)',format='%.1f'),
            'prob_a':st.column_config.NumberColumn('Visitante (%)',format='%.1f')})
        baixar(vista,'jogos_'+str(execucao_id))
with metodo:
    st.markdown('''**Modelo:** regressão logística B, 27 features; EWM com span=5; Elo com K=18 e regressão sazonal de 0,75.

**Simulação:** cada temporada começa no estado real. Após cada placar, pontos, médias e Elo são atualizados. Desempates usam pontos, vitórias, saldo e gols pró. Critérios disciplinares e outros desempates oficiais não são modelados.

**Publicação:** os resultados são gravados juntos. Repetir a mesma publicação reutiliza seu ID, sem duplicação.

**Atualização:** execute `atualizar_resultados.cmd` na pasta do projeto. Depois clique em **Recarregar resultados**. Abrir o painel não executa o Monte Carlo.

**Odds justas:** 1 dividido pela probabilidade, sem margem. Nenhum agente está implementado.''')
    st.write('Seed:',meta.get('seed'))
    st.write('Calculado em (UTC):',meta.get('gerado_em'))
    st.caption('A métrica de validação não foi recalculada nesta publicação. Fica vazia em vez de reutilizar um resultado antigo.')

