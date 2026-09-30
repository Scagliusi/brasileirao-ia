"""Mesmas transições do notebook, com estados NumPy de várias temporadas.

Cada coluna de simulação tem seu próprio estado. O modelo prevê todas as
temporadas em uma chamada por partida; não filtramos DataFrames a cada time.
A seed continua reproduzível, mas mudar o tamanho do lote muda a sequência
dos sorteios. Com uma única temporada, o caminho coincide com a versão original.
"""
import numpy as np
import pandas as pd
from src.features.build_features import features_modelo_B

def simular_lote(jogos_restantes, estado_inicial, estado_mando_inicial, modelo,
                 n_simulacoes=1000, seed=42):
    if n_simulacoes < 1:
        raise ValueError('n_simulacoes deve ser positivo')
    rng=np.random.default_rng(seed)
    times=estado_inicial['time'].to_numpy()
    indices={t:i for i,t in enumerate(times)}
    n_times=len(times)
    campos=['jogos','pontos','vitorias','gols_pro','gols_contra','saldo_gols',
            'pontos_ewm_atual','gols_ewm_atual','gols_sofridos_ewm_atual','vitoria_ewm_atual','elo_atual']
    geral={c:np.broadcast_to(estado_inicial[c].to_numpy(dtype=float),(n_simulacoes,n_times)).copy() for c in campos}
    bases=['pontos','gols','gols_sofridos','vitoria']
    index_mando=pd.MultiIndex.from_product([times,[0,1]],names=['time','casa'])
    mando_df=estado_mando_inicial.set_index(['time','casa']).reindex(index_mando)
    mando={b:np.broadcast_to(mando_df[b+'_ewm_mando_atual'].to_numpy(dtype=float).reshape(n_times,2),
                             (n_simulacoes,n_times,2)).copy() for b in bases}
    if any(not np.isfinite(v).all() for v in [*geral.values(),*mando.values()]):
        raise ValueError('Estado incompleto ou não finito para a simulação.')
    classes=np.asarray(modelo.classes_)
    jogos=jogos_restantes.sort_values(['data','rodada'])
    alpha=2/(5+1)
    for numero,(_,jogo) in enumerate(jogos.iterrows(),1):
        m,v=indices[jogo['time_mandante']],indices[jogo['time_visitante']]
        em,ev=geral['elo_atual'][:,m].copy(),geral['elo_atual'][:,v].copy()
        features={'elo_mandante_sazonal':em,'elo_visitante_sazonal':ev,'elo_sazonal_diff':em-ev}
        for b in bases:
            gm,gv=geral[b+'_ewm_atual'][:,m],geral[b+'_ewm_atual'][:,v]
            mm,mv=mando[b][:,m,1],mando[b][:,v,0]
            features.update({b+'_ewm_mandante':gm,b+'_ewm_visitante':gv,b+'_ewm_diff':gm-gv,
                             b+'_ewm_mando_mandante':mm,b+'_ewm_mando_visitante':mv,b+'_ewm_mando_diff':mm-mv})
        probs=modelo.predict_proba(pd.DataFrame(features,columns=features_modelo_B))
        if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1),1):
            raise ValueError('Modelo retornou probabilidades inválidas.')
        sorteios=rng.random(n_simulacoes)
        classe=classes[np.minimum((sorteios[:,None]>np.cumsum(probs,axis=1)).sum(axis=1),len(classes)-1)]
        ataque_m=(geral['gols_ewm_atual'][:,m]+mando['gols'][:,m,1])/2
        ataque_v=(geral['gols_ewm_atual'][:,v]+mando['gols'][:,v,0])/2
        defesa_v=(geral['gols_sofridos_ewm_atual'][:,v]+mando['gols_sofridos'][:,v,0])/2
        defesa_m=(geral['gols_sofridos_ewm_atual'][:,m]+mando['gols_sofridos'][:,m,1])/2
        lm,lv=(ataque_m+defesa_v)/2,(ataque_v+defesa_m)/2
        if (lm<0).any() or (lv<0).any() or ((classe=='H')&(lm==0)).any() or ((classe=='A')&(lv==0)).any():
            raise ValueError('Sorteio de placar incompatível com médias de gols.')
        gols_m=np.empty(n_simulacoes,dtype=int);gols_v=np.empty(n_simulacoes,dtype=int)
        pendentes=np.arange(n_simulacoes)
        for tentativa in range(100000):
            pm=rng.poisson(lm[pendentes]);pv=rng.poisson(lv[pendentes]);c=classe[pendentes]
            aceitos=((c=='H')&(pm>pv))|((c=='D')&(pm==pv))|((c=='A')&(pm<pv))
            gols_m[pendentes[aceitos]]=pm[aceitos];gols_v[pendentes[aceitos]]=pv[aceitos]
            pendentes=pendentes[~aceitos]
            if not len(pendentes):break
        else:
            raise RuntimeError('Limite de tentativas para sortear placar compatível.')
        vm=(gols_m>gols_v).astype(float);vv=(gols_v>gols_m).astype(float);emp=(gols_m==gols_v).astype(float)
        pontos_m=3*vm+emp;pontos_v=3*vv+emp
        for i,casa,gp,gc,pontos,vit in [(m,1,gols_m,gols_v,pontos_m,vm),(v,0,gols_v,gols_m,pontos_v,vv)]:
            geral['jogos'][:,i]+=1;geral['pontos'][:,i]+=pontos;geral['vitorias'][:,i]+=vit
            geral['gols_pro'][:,i]+=gp;geral['gols_contra'][:,i]+=gc
            geral['saldo_gols'][:,i]=geral['gols_pro'][:,i]-geral['gols_contra'][:,i]
            for b,novo in zip(bases,[pontos,gp,gc,vit]):
                geral[b+'_ewm_atual'][:,i]=alpha*novo+(1-alpha)*geral[b+'_ewm_atual'][:,i]
                mando[b][:,i,casa]=alpha*novo+(1-alpha)*mando[b][:,i,casa]
        esperado_m=1/(1+10**((ev-em)/400))
        geral['elo_atual'][:,m]=em+18*(vm+0.5*emp-esperado_m)
        geral['elo_atual'][:,v]=ev+18*(vv+0.5*emp-(1-esperado_m))
        if numero%20==0 or numero==len(jogos):
            print(f'Monte Carlo: {numero}/{len(jogos)} jogos processados nas {n_simulacoes} temporadas.',flush=True)
    ordem=np.lexsort((-geral['gols_pro'],-geral['saldo_gols'],-geral['vitorias'],-geral['pontos']),axis=1)
    return pd.DataFrame({'simulacao':np.repeat(np.arange(n_simulacoes),n_times),
                         'time':times[ordem].ravel(),'posicao':np.tile(np.arange(1,n_times+1),n_simulacoes),
                         'pontos':np.take_along_axis(geral['pontos'],ordem,axis=1).ravel()})
