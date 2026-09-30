# Brasileirão IA

Aplicação local completa para atualizar dados, treinar a Logistic B, simular o campeonato, publicar no PostgreSQL e consultar resultados no Streamlit. **Não contém agente.**

## Abrir e atualizar

- Abra `abrir_app.cmd` com duplo clique. O painel usa `http://127.0.0.1:8501`. Mantenha o processo aberto enquanto consultar.
- Para uma rodada nova, execute `atualizar_resultados.cmd`. Ele consulta a API, recalcula estados/features, treina ou reutiliza o modelo, simula 1.000 temporadas e publica tudo junto. Depois, clique em **Recarregar resultados** no painel.
- Abrir a tela não recalcula previsões. É somente uma consulta ao PostgreSQL.
- Repetir uma publicação idêntica devolve o mesmo ID, sem duplicar linhas. As execuções anteriores não são apagadas.

O PostgreSQL precisa estar em execução. O `.env` da raiz deve definir `DATABASE_URL` (driver postgresql+psycopg) e `FOOTBALL_DATA_KEY`. As credenciais existentes foram preservadas.

## Comandos individuais

Execute na raiz do projeto, com `.venv` ativado:

```powershell
python pipeline.py atualizar
python pipeline.py treinar
python pipeline.py simular --simulacoes 1000 --seed 42
python pipeline.py salvar
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Ou execute todo o processamento e publicação:

```powershell
python pipeline.py completo --temporada 2026 --simulacoes 1000 --seed 42
```

`python pipeline.py preparar-banco` cria as tabelas de resultados ausentes sem apagar as já existentes. O comando `salvar` também prepara as tabelas. Não há renumeração de IDs. Lacunas de sequência no PostgreSQL são normais.

O modelo e os resultados ficam em `artifacts/`. A resposta da API fica em `artifacts/api/` para auditoria, sem a chave de autenticação. O arquivo `modelo_final.joblib` é um dicionário com o modelo e sua origem; carregue o modelo por `carregar('modelo_final')['modelo_final']`. Carregue somente arquivos joblib confiáveis.

## O que aparece no painel

- Classificação real, chances de título, G4, G6 e Z4.
- Detalhes por clube, posição média e distribuição de posições.
- Projeção com uma posição distinta para cada clube.
- Jogos futuros com filtros por time/rodada, probabilidades e odds justas.
- Histórico das execuções completas e download das tabelas em CSV.
- Data da última partida incluída, quantidade de simulações, seed e avisos de qualidade/calendário.

As probabilidades no banco e painel estão na escala **0 a 100**. Odds são calculadas com `1/p`, usando p na escala 0 a 1. Probabilidades dos jogos usam o estado real atual, não a frequência de placares do Monte Carlo.

## Arquivos principais

| Arquivo | Papel |
|---|---|
| `pipeline.py` | Comandos e ligação entre etapas |
| `src/data/preparar_dados.py` | CSV + API com nomes e datas padronizados |
| `src/features/build_features.py` | 27 features pré-jogo e Elo |
| `src/models/train_model.py` | Treinamento da Logistic B |
| `src/simulation/estados.py` | Situação real antes das simulações |
| `src/simulation/monte_carlo.py` | Simulação e resumos |
| `src/simulation/lote.py` | Mesmas transições com estados NumPy e previsão em lote para acelerar o fluxo completo |
| `src/simulation/run_simulation.py` | Função compatível com a organização do notebook |
| `src/simulation/probabilidades.py` | Previsões dos jogos futuros |
| `src/data/persistencia.py` | Validação, transação, deduplicação e consultas |
| `src/data/schema.py` | Estrutura das tabelas de resultados |
| `src/data/database.py` | Nomes de funções já utilizados pelo notebook |
| `src/cache.py` | Gravação atômica e leitura dos artefatos |
| `app/streamlit_app.py` | Painel somente de leitura |

São publicadas `execucoes_modelo`, `classificacao_atual`, `probabilidades_times`, `probabilidades_posicao` e `probabilidades_jogos`. `pipeline_publicacoes` identifica execuções completas e guarda sua origem. As tabelas antigas `teams` e `matches` são preservadas; não são necessárias para este painel.

O notebook original é mantido intacto. Chamadas antigas diretas a `criar_execucao`/`salvar_classificacao_atual` continuam possíveis, mas não têm a deduplicação do pacote completo: prefira `pipeline.py salvar`. O painel apresenta apenas execuções completas registradas pelo novo fluxo; não considera uma execução pai antiga vazia como resultado pronto.

## Hipóteses e limites preservados

- Treino desde 2012; Logistic Regression com C=0.01 e max_iter=1000. EWM span=5; Elo K=18 e regressão sazonal alpha=0.75.
- O CSV anterior a 2024 participa da construção do histórico. Foram encontrados registros inconsistentes de mesmo mando em 2006 com datas/rodadas diferentes; foram preservados e sinalizados. O cruzamento das features agora inclui o adversário para não multiplicar partidas.
- Quando há jogo adiado ou sem data confirmada, a cópia do calendário usada na simulação segue a ordem de rodadas e datas. A data oficial não é alterada no banco. A ordem real futura pode mudar as projeções; o painel avisa quando essa hipótese foi usada.
- Jogos em andamento, suspensos ou cancelados bloqueiam uma nova publicação até serem concluídos ou corrigidos na fonte. Times sem estado geral ou por mando também exigem conferência.
- Desempates usam pontos, vitórias, saldo e gols pró. Não modelam cartões e outros critérios oficiais. G4/G6 são posições, não garantia de vagas em torneios.
- A classificação projetada por atribuição maximiza a soma de probabilidades individuais; não é a probabilidade conjunta de toda a classificação.
- Não foi recalculada uma métrica de validação nesta publicação; `log_loss_validacao` fica nulo. Resultados de treinamento não são apresentados como desempenho de teste.
- O fluxo completo simula várias temporadas em lote, mantendo estados independentes e as fórmulas originais. Isso reduz filtros de DataFrame e chamadas ao modelo. A seed é reproduzível para a mesma quantidade de simulações; o sorteio em lote muda a sequência em relação ao loop antigo. A equivalência de uma temporada foi testada com três seeds. A versão individual do notebook permanece disponível para comparação.
- O cache só é concluído após a simulação terminar; não há retomada parcial de uma execução interrompida. Alterações no código ou dados exigem um modelo compatível antes de simular/publicar.

## Testes

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Os testes usam dados sintéticos e banco temporário, sem inserir previsões de teste no banco de produção. Também foi feita verificação com os dados reais antes da publicação.

## Instalação em outra máquina

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Configure o `.env`, disponibilize o CSV em `src/data/brasileirao_historico.csv` e o PostgreSQL. Depois execute `atualizar_resultados.cmd` e `abrir_app.cmd`.
