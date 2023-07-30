def abrindo_decomposicao(nome = 'dados.txt'):
    import os
    import pandas as pd
    
    path = '/Users/rafaelvieira/Desktop/Tratamento Dados - CHEIC'
    nome = nome
    

    # Nome do arquivo que contém os dados
#     nome_arquivo = 'dados.txt'
    nome_arquivo = os.path.join(path, nome)

    # Lista para armazenar os dados
    data = []

    # Leitura dos dados do arquivo
    with open(nome_arquivo, 'r') as file:
        lines = file.readlines()

    # Loop para processar as linhas e extrair os dados
    for line in lines:
        # Verifica se a linha não está vazia e não contém o título das colunas
        if line.strip() and 'Residue' not in line:
            # Divide a linha em elementos separados por vírgula
            elements = line.strip().split(',')

            # Remove espaços em branco dos elementos
            elements = [e.strip() for e in elements]

            # Extrai os dados relevantes
            residue = elements[0]
            internal = float(elements[1]) if elements[1] and elements[1] != 'Avg.' else None
            vdw = float(elements[4]) if elements[4] and elements[4] != 'Avg.' else None
            electrostatic = float(elements[7]) if elements[7] and elements[7] != 'Avg.' else None
            polar_solvation = float(elements[10]) if elements[10] and elements[10] != 'Avg.' else None
            non_polar_solvation = float(elements[13]) if elements[13] and elements[13] != 'Avg.' else None
            total = float(elements[16]) if elements[16] and elements[16] != 'Avg.' else None

            # Adiciona os dados à lista
            data.append([residue, internal, vdw, electrostatic, polar_solvation, non_polar_solvation, total])

    # Cria o DataFrame Pandas
    columns = ['Residue', 'Internal', 'van der Waals', 'Electrostatic', 'Polar Solvation', 'Non-Polar Solvation', 'Total']
    df = pd.DataFrame(data, columns=columns)

    # Remove linhas com valores ausentes (NaN)
    df = df.dropna()
    
    return df

