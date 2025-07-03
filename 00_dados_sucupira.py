import pandas as pd
from pathlib import Path

ROOT = Path('../')
DATA_DIR   = ROOT / 'data' / 'raw' / 'files_from_sucupira_plataform'
CO_PROFIAP = '53045009001P3'

def carregar_e_filtrar_csvs(data_dir, co_programa):
    """
    Lê todos os arquivos .csv em `data_dir`, concatena
    em um único DataFrame e filtra onde CD_PROGRAMA == programa.
    """
    # Lista todos os CSVs no diretório
    csv_paths = list(data_dir.glob('*.csv'))
    if not csv_paths:
        raise FileNotFoundError(f"Não foi encontrado nenhum CSV em {data_dir}")

    # Lê cada CSV em um DataFrame e guarda numa lista
    dfs = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"Lido {path.name}: {len(df)} linhas")
        except Exception as e:
            print(f"Erro ao ler {path.name}: {e}")

    # Concatena todos
    todos = pd.concat(dfs, ignore_index=True)
    print(f"Total concatenado: {len(todos)} linhas")

    # Filtra
    filtrado = todos[todos['CD_PROGRAMA'] == co_programa]
    print(f"Linhas com CD_PROGRAMA == {co_programa}: {len(filtrado)}")

    return filtrado

if __name__ == "__main__":
    df_resultado = carregar_e_filtrar_csvs(DATA_DIR, CO_PROFIAP)
    df_resultado.to_csv(ROOT/'data'/'raw'/'dissertacoes_profiap_14_23.csv')