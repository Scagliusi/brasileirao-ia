"""Artefatos locais versionados pelo conteúdo e gravados atomicamente."""
import os
from pathlib import Path
from uuid import uuid4
import joblib
from src.config import ARTEFATOS


def salvar(nome, objeto, pasta=ARTEFATOS):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / (nome + '.joblib')
    temporario = pasta / (nome + '.' + uuid4().hex + '.tmp')
    try:
        joblib.dump(objeto, temporario)
        os.replace(temporario, destino)
    finally:
        temporario.unlink(missing_ok=True)


def carregar(nome, pasta=ARTEFATOS):
    caminho = Path(pasta) / (nome + '.joblib')
    if not caminho.exists():
        raise ValueError(f'Artefato {nome} ausente. Execute a etapa anterior.')
    return joblib.load(caminho)


def assinatura(objeto):
    return joblib.hash(objeto, hash_name='sha1')
