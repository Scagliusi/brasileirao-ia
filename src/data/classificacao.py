# Código extraído de src/data/tratamento.ipynb; células numeradas a partir de 1.

def montar_tabela_real(estado_tabela):
    # Célula original 326
    tabela_real = (
        estado_tabela
        .sort_values(
            ["pontos", "vitorias", "saldo_gols", "gols_pro"],
            ascending=False
        )
        .reset_index(drop=True)
    )

    tabela_real["posicao"] = tabela_real.index + 1

    return tabela_real
