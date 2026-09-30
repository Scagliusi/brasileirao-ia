"""Etapas independentes: atualizar, treinar, simular, salvar e preparar-banco."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from src.config import RAIZ, ARTEFATOS
from src.cache import carregar, salvar, assinatura


def codigo_assinatura():
    return assinatura([(str(p.relative_to(RAIZ)), p.read_bytes())
                       for p in sorted((RAIZ / 'src').rglob('*.py'))] +
                      [('pipeline.py', Path(__file__).read_bytes())])


def atualizar(csv, temporada, pasta=ARTEFATOS):
    
    from src.data.preparar_dados import atualizar_dados
    from src.features.build_features import construir_features
    from src.simulation.estados import preparar_estados
    df_completo, df_jogos_restantes = atualizar_dados(csv, temporada)
    df_times_completo, df_modelo_atual = construir_features(df_completo)
    estado_tabela, estado_inicial, estado_mando = preparar_estados(
        df_completo, df_times_completo, temporada)
    dados = dict(temporada=temporada, df_completo=df_completo,
                 df_jogos_restantes=df_jogos_restantes, df_modelo_atual=df_modelo_atual,
                 estado_tabela=estado_tabela, estado_inicial=estado_inicial,
                 estado_ewm_mando_atual=estado_mando)
    dados['assinatura'] = assinatura(dados)
    salvar('dados', dados, pasta)
    print(f'Dados prontos: {len(df_completo)} jogos históricos, {len(df_jogos_restantes)} restantes.', flush=True)
    return dados


def treinar(pasta=ARTEFATOS):
    from src.models.train_model import treinar_modelo
    dados = carregar('dados', pasta)
    chave = assinatura((dados['assinatura'], codigo_assinatura()))
    if (Path(pasta) / 'modelo_final.joblib').exists():
        existente = carregar('modelo_final', pasta)
        if existente['assinatura'] == chave:
            print('Modelo já treinado para estes dados; carregado do disco.')
            return existente
    modelo = dict(assinatura=chave, dados_assinatura=dados['assinatura'],
                  codigo=codigo_assinatura(), modelo_final=treinar_modelo(dados['df_modelo_atual']))
    salvar('modelo_final', modelo, pasta)
    return modelo


def simular(n_simulacoes=1000, seed=42, pasta=ARTEFATOS):
    import numpy as np
    from src.simulation.monte_carlo import rodar_monte_carlo, resumir_monte_carlo
    from src.simulation.lote import simular_lote
    from src.simulation.probabilidades import probabilidades_jogos
    dados, modelo = carregar('dados', pasta), carregar('modelo_final', pasta)
    if modelo['dados_assinatura'] != dados['assinatura'] or modelo['codigo'] != codigo_assinatura():
        raise ValueError('Os dados ou o código mudaram. Execute treinar antes de simular.')
    if n_simulacoes < 1:
        raise ValueError('Use pelo menos uma simulação.')
    chave = assinatura((dados['assinatura'], modelo['assinatura'], n_simulacoes, seed))
    nome = 'simulacao_' + chave
    if (Path(pasta) / (nome + '.joblib')).exists():
        resultado = carregar(nome, pasta)
        salvar('resultado_atual', resultado, pasta)
        print('Monte Carlo já disponível; carregado do disco.')
        return resultado
    jogos, estado, mando = (dados['df_jogos_restantes'], dados['estado_inicial'],
                             dados['estado_ewm_mando_atual'])
    if estado.empty:
        raise ValueError('Temporada sem classificação atual; confira os jogos realizados.')
    times = set(jogos['time_mandante']) | set(jogos['time_visitante'])
    if times - set(estado['time']):

        raise ValueError('Há times futuros sem estado atual. Confira nomes e jogos realizados.')
    if jogos['rodada'].isna().any():

        raise ValueError('Há jogos sem rodada; confira o calendário na API.')
    if 'status' in jogos and not jogos['status'].isin(['SCHEDULED', 'TIMED', 'POSTPONED']).all():
        raise ValueError('Há jogos em andamento, suspensos ou cancelados. Atualize após a conclusão ou correção na API.')
    avisos = list(dados['df_completo'].attrs.get('avisos', []))
    jogos_simulacao = jogos.copy()
    if jogos['data'].isna().any() or ('status' in jogos and jogos['status'].eq('POSTPONED').any()):
        avisos.append('Há partida adiada ou sem data confirmada. A simulação usa a ordem das rodadas para os jogos restantes; a ordem real pode mudar as projeções.')
        import pandas as pd
        jogos_simulacao = jogos.sort_values(['rodada','data'], kind='stable', na_position='last').copy()
        jogos_simulacao['data'] = pd.date_range('2000-01-01', periods=len(jogos), freq='h')
        print(avisos[-1], flush=True)
    pares = set(zip(mando['time'], mando['casa']))
    if any((t, c) not in pares for t in times for c in (0, 1)):

        raise ValueError('Histórico de mando incompleto para os times da temporada.')
    campos = [c for c in estado if c.endswith('_atual')]
    campos_mando = [c for c in mando if c.endswith('_atual')]

    if not np.isfinite(estado[campos].to_numpy(dtype=float)).all() or not np.isfinite(mando[campos_mando].to_numpy(dtype=float)).all():

        raise ValueError('Estados com médias ausentes; confira o histórico.')
    
    df_mc = simular_lote(jogos_simulacao, estado, mando, modelo['modelo_final'], n_simulacoes, seed)

    resumo_mc, prob_posicao, tabela_projetada = resumir_monte_carlo(df_mc, n_simulacoes)

    tabela_real = dados['estado_tabela'].copy()

    tabela_real['posicao'] = range(1, len(tabela_real) + 1)

    resultado = dict(assinatura=chave, dados_assinatura=dados['assinatura'], codigo=modelo['codigo'],
                     gerado_em=datetime.now(timezone.utc).isoformat(), avisos=avisos,
                     ultima_partida=str(dados['df_completo']['data'].max()),
                     jogos_realizados=int(dados['df_completo']['ano_campeonato'].eq(dados['temporada']).sum()),
                     jogos_restantes=len(jogos),
                     temporada=dados['temporada'], n_simulacoes=n_simulacoes, seed=seed,
                     df_mc=df_mc, resumo_mc=resumo_mc, prob_posicao=prob_posicao,
                     tabela_projetada=tabela_projetada, tabela_real=tabela_real,
                     probabilidades_jogos=probabilidades_jogos(jogos, estado, mando, modelo['modelo_final']))
    
    salvar(nome, resultado, pasta)
    salvar('resultado_atual', resultado, pasta)
    return resultado


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('etapa', choices=['atualizar', 'treinar', 'simular', 'salvar', 'preparar-banco', 'completo'])
    parser.add_argument('--csv', type=Path, default=RAIZ / 'src/data/brasileirao_historico.csv')
    parser.add_argument('--temporada', type=int, default=2026)
    parser.add_argument('--simulacoes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.etapa == 'completo':
        from src.data.persistencia import obter_engine, inicializar_controle, salvar_resultados
        atualizar(args.csv, args.temporada)
        treinar()
        resultado = simular(args.simulacoes, args.seed)
        inicializar_controle(obter_engine())
        print('Execução publicada:', salvar_resultados(resultado))
    elif args.etapa == 'atualizar':
        atualizar(args.csv, args.temporada)
    elif args.etapa == 'treinar':
        treinar()
    elif args.etapa == 'simular':
        simular(args.simulacoes, args.seed)
    else:
        from src.data.persistencia import obter_engine, inicializar_controle, salvar_resultados
        if args.etapa == 'preparar-banco':
            inicializar_controle(obter_engine())
        else:
            resultado, dados = carregar('resultado_atual'), carregar('dados')
            if resultado['dados_assinatura'] != dados['assinatura']:
                raise ValueError('Resultado desatualizado. Execute treinar e simular.')
            if resultado.get('codigo') != codigo_assinatura():
                raise ValueError('Código alterado após a simulação. Execute treinar e simular antes de publicar.')
            inicializar_controle(obter_engine())
            print('Execução no PostgreSQL:', salvar_resultados(resultado))
    print('Etapa concluída:', args.etapa)


if __name__ == '__main__':
    main()
