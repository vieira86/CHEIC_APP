#!/usr/bin/env python
# coding: utf-8
# ######################
# Import libraries
######################
# 
import streamlit as st
import streamlit.components.v1 as stc
import numpy as np
# File Processing Pkgs
import pandas as pd
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
# import matchms
import os
# from matchms.importing import load_from_msp
# from matchms.filtering import default_filters
# from matchms.filtering import repair_inchi_inchikey_smiles
# from matchms.filtering import derive_inchikey_from_inchi
# from matchms.filtering import derive_smiles_from_inchi
# from matchms.filtering import derive_inchi_from_smiles
# from matchms.filtering import harmonize_undefined_inchi
# from matchms.filtering import harmonize_undefined_inchikey
# from matchms.filtering import harmonize_undefined_smiles
# from matchms.filtering import default_filters
# from matchms.filtering import normalize_intensities
# from matchms.filtering import select_by_intensity
# from matchms.filtering import select_by_mz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
import mols2grid
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# st.sidebar.image(image, use_column_width=True)
# st.write("""
#         #            LUMIOS
#         ###### Label Using Machine In Organic Samples
#         ***
#         """)
st.markdown("<h1 style='text-align: center; color: black;'>CHEIC - CHEMICAL IMAGE CLASSIFICATOR</h1>", unsafe_allow_html=True)
# st.markdown("<h5 style='text-align: center; color: gray;'>Classification using rede neural</h5>", unsafe_allow_html=True)



# image = Image.open('lumios.jpg')

# st.image(image, use_column_width=True)


def main():
    # st.title("Molecular Desreplication - APP")
    # menu = ["Home", "Deep Learning", "Classificação", "Molecular Docking", "About"]
    menu = ["Home", "Deep Learning",  "Docking", "Data Molecular Dynamics", "About"]
    import streamlit as st
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        from PIL import Image
        image = Image.open('capa_app.png')
        st.image(image, use_column_width=True)
        image = Image.open('cheic-aba.png')
        st.sidebar.image(image, use_column_width=True)

    elif choice == ("Deep Learning"):
        # from PIL import Image
        # image1 = Image.open('smiles_menor.png')
        # st.sidebar.image(image1)

        # st.subheader("sobre o processamento dos dados")

        title = st.sidebar.text_input('Insert SMILES here: ')
        data_file = st.sidebar.file_uploader("Insert a list of SMILES here",type=['xlsx'], accept_multiple_files=False)

        if len(title)==0:
            st.markdown('#### Waiting')
            # from PIL import Image
            # image = Image.open('desenho.png')
            # st.image(image)
            # st.write('Process your data!')

        # data_file = st.file_uploader("Upload CSV",type=['npy'], accept_multiple_files=False)

        else:
            # if st.button("Process"):
            if title is not None:
                st.sidebar.write('Seu smile é:', title)

                import pandas as pd
                lista =[]
                lista.append(title)
                molecula = pd.DataFrame(lista)
                molecula.rename(columns = {0: 'SMILES'}, inplace=True)

                # anotacao_final.rename(columns={'smiles': 'SMILES'}, inplace=True)
                raw_html = mols2grid.display(molecula,  subset=["SMILES", 'img'])._repr_html_()
                # components.html(raw_html, width=900, height=900, scrolling=True)
                components.html(raw_html)

                # st.write("#### Codificando sua estrutura para input da rede neural")

                st.dataframe(molecula)

                from PIL import Image 
                import os   
                import pandas as pd
                from rdkit import Chem
                from rdkit.Chem import Draw

                # dir = r'C:\Users\PC\Desktop\Projeto-Streamlit-MVOCS\teste_moleculas2'
                # if not os.path.exists(dir):
                #     os.makedirs(dir)

                import tempfile
                import os

                # Cria uma pasta temporária
                dir_temporaria = tempfile.mkdtemp()

                # Use a pasta temporária conforme necessário
                # print("Pasta temporária criada em:", dir_temporaria)

                # Faça o que precisa fazer com a pasta temporária aqui...

                # Lembre-se de remover a pasta temporária quando não precisar mais dela
                # os.rmdir(dir_temporaria)
                # print("Pasta temporária removida.")

                # Salvar as imagens das moléculas na pasta temporária
                # Salvar as imagens das moléculas na pasta temporária
                # import streamlit as st

                # Assuming dir_temporaria is a list of image file paths
                # for img_path in dir_temporaria:
                #     st.image(img_path, caption="Imagem salva na pasta temporária")

                # for i in range(len(molecula)):
                mol = Chem.MolFromSmiles(molecula['SMILES'][0]) ### é sempre zero pq eh individual
                img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(100, 100), returnPNG=False)
                nome_imagem = f"mol_{0}.png"  # Nome único para cada imagem
                endereco = os.path.join(dir_temporaria, nome_imagem)
                st.write(endereco)
                img.save(endereco)

                # st.markdown("Imagens salvas na pasta temporária:", dir_temporaria)

                st.markdown('#### Converting SMILES in Image with dimension: 100x100')

                # from PIL import Image
                # image = Image.open(r'C:\Users\PC\Desktop\Projeto-Streamlit-MVOCS\teste_moleculas2\teste_smiles.png')
                # st.image(image, use_column_width=False)

            
                import os

                dataset_folder = dir_temporaria

                class_img_filenames = os.listdir(dataset_folder)

                endereco_moleculas = []
                nome_molecula =[]
                for i in class_img_filenames:
                    full_img_path = os.path.join(dataset_folder, i)

                    endereco_moleculas.append(full_img_path)
                    nome_molecula.append(i)

                #Lembre-se de remover a pasta temporária quando não precisar mais dela
                import shutil

                # Remova o diretório inteiro, incluindo arquivos e subdiretórios
                shutil.rmtree(dir_temporaria)
                st.write("Pasta temporária removida.")
                

                import pandas as pd

                dataset_df = pd.DataFrame({
                    'image_pathname': endereco_moleculas,
                    'class': nome_molecula
                })

                import cv2
                imagens_train = []
                for i in dataset_df['image_pathname']:
                    endereco =  i
                    print(endereco)
                    img = cv2.imread(endereco)
                    imagens_train.append(img)

                import numpy as np
                x_novo_mvocs = np.zeros((len(imagens_train), 100, 100, 3)) # (n_amostras, (dimensao))

                defeito = []
                for i in range(len(x_novo_mvocs)):
                    try:
                        x_novo_mvocs[i] = imagens_train[i]
                    except:
                        print(i)
                        defeito.append(i)

                for i in range(x_novo_mvocs.shape[0]):
                    x_novo_mvocs[i] = x_novo_mvocs[i]/250

                from tensorflow import keras
                model = keras.models.load_model(r'C:\Users\Rafa Vieira\Desktop\New_CHEIC\modelo_keras')

                molecula_test_proba = model.predict(x_novo_mvocs)

                y_test_pred = []
                for i in molecula_test_proba:
                    if i > 0.5:
                        a = 1
                    else:
                        a=0
                    y_test_pred.append(a)

                results = []
                if y_test_pred[0] == 1:
                    a = 'Respiratory/Drug'
                    results.append(a)
                else:
                    a = 'mVOCS'
                    results.append(a)

                molecula['results'] = results

                st.write('##### Results: ', title)
                st.dataframe(molecula)

        if data_file is None:
            
            from PIL import Image
            image = Image.open('desenho.png')
            st.image(image)

        else:
            # if st.button("Process"):
            if data_file is not None:
                st.sidebar.write('Processing')
                # st.dataframe(data_file)

                import pandas as pd
                # lista =[]
                # lista.append(title)
                molecula = pd.read_excel(data_file)
                molecula.rename(columns = {'smiles': 'SMILES'}, inplace=True)

                # anotacao_final.rename(columns={'smiles': 'SMILES'}, inplace=True)
                raw_html = mols2grid.display(molecula,  subset=["SMILES", 'img'])._repr_html_()
                components.html(raw_html, width=900, height=300, scrolling=True)
                # components.html(raw_html)

                # st.write("#### Codificando sua estrutura para input da rede neural")

                st.dataframe(molecula)

                from PIL import Image 
                import os   
                import pandas as pd
                from rdkit import Chem
                from rdkit.Chem import Draw

                # dir = r'C:\Users\PC\Desktop\Projeto-Streamlit-MVOCS\teste_moleculas2'
                # if not os.path.exists(dir):
                #     os.makedirs(dir)
                import tempfile
                import os

                # Cria uma pasta temporária
                dir_temporaria = tempfile.mkdtemp()

                from PIL import Image 
                import os   
                import pandas as pd
                from rdkit import Chem
                from rdkit.Chem import Draw

                for i in range(len(molecula)):
                    mol = Chem.MolFromSmiles(molecula['SMILES'][i]) ### é sempre zero pq eh individual
                    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(100, 100), returnPNG=False)
                    nome_imagem = f"mol_{i}.png"  # Nome único para cada imagem
                    endereco = os.path.join(dir_temporaria, nome_imagem)
                    st.write(endereco)
                    img.save(endereco)

                # st.write('imagem foi processada no formato 100x100')

                # from PIL import Image
                # image = Image.open(r'C:\Users\PC\Desktop\Projeto-Streamlit-MVOCS\teste_moleculas2\teste_smiles.png')
                # st.image(image, use_column_width=False)

                import os

                dataset_folder = dir_temporaria

                class_img_filenames = os.listdir(dataset_folder)

                endereco_moleculas = []
                nome_molecula =[]
                for i in class_img_filenames:
                    full_img_path = os.path.join(dataset_folder, i)

                    endereco_moleculas.append(full_img_path)
                    nome_molecula.append(i)
                

                import pandas as pd

                dataset_df = pd.DataFrame({
                    'image_pathname': endereco_moleculas,
                    'class': nome_molecula
                })

                # Remova o diretório inteiro, incluindo arquivos e subdiretórios
                import shutil
                shutil.rmtree(dir_temporaria)
                st.write("Pasta temporária removida.")

                import cv2
                imagens_train = []
                for i in dataset_df['image_pathname']:
                    endereco =  i
                    print(endereco)
                    img = cv2.imread(endereco)
                    imagens_train.append(img)

                import numpy as np
                x_novo_mvocs = np.zeros((len(imagens_train), 100, 100, 3)) # (n_amostras, (dimensao))

                defeito = []
                for i in range(len(x_novo_mvocs)):
                    try:
                        x_novo_mvocs[i] = imagens_train[i]
                    except:
                        print(i)
                        defeito.append(i)

                for i in range(x_novo_mvocs.shape[0]):
                    x_novo_mvocs[i] = x_novo_mvocs[i]/250

                from tensorflow import keras
                model = keras.models.load_model(r'c:\Users\Rafa Vieira\Desktop\New_CHEIC\modelo_keras')

                molecula_test_proba = model.predict(x_novo_mvocs)

                y_test_pred = []
                for i in molecula_test_proba:
                    if i > 0.6:
                        a = 1
                    else:
                        a=0
                    y_test_pred.append(a)

                results = []
                for i in y_test_pred:
                    if i == 1:
                        a = 'Respiratory/Drug'
                        results.append(a)
                    else:
                        a = 'mVOCS'
                        results.append(a)

                molecula['results'] = results

                st.markdown('### mVOCS Classification by Convolutional Neural Network')
                st.dataframe(molecula)

                @st.cache_data
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(molecula)

                st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='amostras_deep_learning_for_docking.csv',
                mime='text/csv',
                )

        

    elif choice == ("Docking"):
        data_file = st.sidebar.file_uploader("Insert the output from Deep Learning classification",type=['csv'], accept_multiple_files=False)
 

        if st.button("DOCKING"):
                if data_file is not None:
                    st.write('Verificando moleculas')
        

        # st.subheader("sobre o processamento dos dados")

        # title = st.sidebar.text_input('Insert SMILES here: ')
                    import pandas as pd
                    molecula = pd.read_csv(data_file)
                    try:
                        molecula.drop('Unnamed: 0',axis=1, inplace=True)
                    except:
                        print('')
                    molecula.rename(columns = {'smiles': 'SMILES'}, inplace=True)

                    # anotacao_final.rename(columns={'smiles': 'SMILES'}, inplace=True)
                    raw_html = mols2grid.display(molecula,  subset=["SMILES", 'img'])._repr_html_()
                    components.html(raw_html, width=900, height=300, scrolling=True)
                    # components.html(raw_html)

                    # molecula = molecula.loc[molecula['results']=='Respiratory/Drug']

                    df = molecula

                    # df.rename(columns = {'index': 'index_base1'}, inplace=True)
                    df.rename(columns = {'SMILES': 'smiles'}, inplace=True)
                    st.dataframe(df)

                    df.reset_index(inplace=True)
                    df.rename(columns = {'index': 'id'}, inplace=True)
                    st.dataframe(df)

                    df.rename(columns = {'index': 'index_base'}, inplace=True)
                    # df.rename(columns = {'Unnamed: 0', 'index'}, inplace=True)
                    # df.reset_index(inplace=True)

                    st.markdown ('## ESTE É O DATAFRAME')

                    st.dataframe(df)

                    import os
                    import git
                    import tempfile

                    import os

                    # Obtenha o caminho para a área de trabalho
                    desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

                    # Nome da nova pasta a ser criada
                    nova_pasta_nome = "CHEIC"

                    # Caminho completo da nova pasta
                    nova_pasta_path = os.path.join(desktop_path, nova_pasta_nome)

                    # Verifica se a pasta já existe antes de criar
                    if not os.path.exists(nova_pasta_path):
                        os.makedirs(nova_pasta_path)
                        st.write(f"A pasta '{nova_pasta_nome}' foi criada na área de trabalho.")
                    else:
                        st.write(f"A pasta '{nova_pasta_nome}' já existe na área de trabalho.")
                        
                        
                    try:
                        import os
                        import git
                        import tempfile
                        # Clona o repositório para a pasta temporária
                        repo_url = "https://github.com/vieira86/NEW_CHEIC.git"
                        git.Repo.clone_from(repo_url, nova_pasta_path)
                        st.write(f"Clone do repositório concluído. Pasta temporária: {nova_pasta_nome}")
                    except git.exc.GitCommandError as e:
                        st.write(f"Erro ao clonar o repositório: {e}")


                    # Cria uma pasta temporária
                    dir_temporaria = nova_pasta_path

                    print(f"Pasta temporária criada em: {dir_temporaria}")

                    # Cria uma pasta dentro da pasta temporária
                    nova_pasta = "runs2"
                    nova_pasta_path = os.path.join(dir_temporaria, nova_pasta)
                    os.makedirs(nova_pasta_path)

                    # print(f"Pasta '{nova_pasta}' criada dentro da pasta temporária.")
                    # print(f"Caminho completo da nova pasta: {nova_pasta_path}")


                    # Cria uma pasta dentro da pasta temporária
                    nova_pasta = "teste_teste"
                    nova_pasta_path = os.path.join(dir_temporaria, nova_pasta)
                    os.makedirs(nova_pasta_path)

                    # print(f"Pasta '{nova_pasta}' criada dentro da pasta temporária.")
                    # print(f"Caminho completo da nova pasta: {nova_pasta_path}")


                    # try:
                    #     molecula.drop('Unnamed: 0',axis=1, inplace=True)
                    # except:
                    #     print('')
                    # molecula.rename(columns = {'smiles': 'SMILES'}, inplace=True)

                    # df = molecula

                    # df.rename(columns = {'index': 'index_base1'}, inplace=True)
                    df.rename(columns = {'SMILES': 'smiles'}, inplace=True)
                    st.markdown ('# renomeado')
                    st.dataframe(df)

                    # df.reset_index(inplace=True)
                    # df.rename(columns = {'index': 'id'}, inplace=True)

                    st.markdown('# este funcina?')
                    st.dataframe(df)

                    df.rename(columns = {'index': 'index_base'}, inplace=True)

                    ids = []
                    for i in df['id']:
                        a = str(i)
                        ids.append(a)

                    df['id'] = ids

                    st.markdown ('# antes do mols')

                    st.dataframe(df)

                    mols = []
                    from rdkit import Chem
                    from rdkit.Chem import AllChem

                    for _, row in df.iterrows():
                        print(row)
                        try:
                            m = Chem.MolFromSmiles(row.smiles)

                            m = Chem.AddHs(m)

                            AllChem.EmbedMolecule(m, AllChem.ETKDG())
                            minimize_status = AllChem.UFFOptimizeMolecule(m, 2000)

                            if not minimize_status == 0:
                                print(f"Failed to minimize_compound'{row['name']}")

                            AllChem.ComputeGasteigerCharges(m)

                            mols.append(m)  
                        except:
                            mols.append(0)   
                            
                    import pandas as pd
                    from rdkit import Chem
                    from rdkit.Chem import AllChem

                    try:
                        df.drop('Unnamed: 0', axis=1, inplace=True)
                    except:
                        print('Nao ha a coluna')

                    # df['index'] = df['index'].astype(str)
                    df['id'] = df['id'].astype(str)

                    st.write(mols)

                    import pandas as pd
                    from rdkit import Chem
                    from rdkit.Chem import AllChem

                    try:
                        df.drop('Unnamed: 0', axis=1, inplace=True)
                    except:
                        print('Nao ha a coluna')

                    # df['index'] = df['index'].astype(str)
                    df['id'] = df['id'].astype(str)

                    st.markdown('DATAFRAME antes do SDF')
                    st.write(df)

                    sdf_file_path = os.path.join(dir_temporaria, 'lig_dataset_novo.sdf')

                    # Cria o escritor SDF
                    sdf_writer = Chem.SDWriter(sdf_file_path)

                    import pandas as pd
                    from rdkit import Chem
                    from rdkit.Chem import AllChem

                    import numpy as np
                    import os
    
                    pdbqt_folder = os.path.join(dir_temporaria, "teste_teste_NEW_CHEIC")
                    if not os.path.exists(pdbqt_folder):
                        os.makedirs(pdbqt_folder)
                    # sdf_writer = Chem.SDWriter('lig_dataset_novo.sdf')

                    lig_properties = df.columns.to_list()

                    st.markdown('lig properties')
                    st.write(lig_properties)

                    for i, mol in enumerate(mols):
                        data_ref = df.iloc[i]
                        mol.SetProp('index', '%s' % i)
                        mol.SetProp('_Name', str(data_ref['id']))  # Convert to string using str()
                        for p in lig_properties:
                            mol.SetProp(f"+{p}", str(data_ref[p]))  # Convert to string using str()
                        sdf_writer.write(mol)
                    sdf_writer.close()

                    from openbabel import pybel
                    import os
                    # os.makedirs(out_dir_lig, exist_ok = True)
                    import tempfile
                    # tempfile.mkdtemp('teste_teste')
                    # print(tempfile.mkdtemp('teste_teste'))
                    # pdbqt_folder = os.path.join(dir_temporaria, "teste_teste_NEW_CHEIC")

                    # Verifica se a pasta de destino para os arquivos PDBQT existe e cria se não existir
                    # if not os.path.exists(pdbqt_folder):
                    #     os.makedirs(pdbqt_folder)
                    # # import os
                    # import shutil

                    # path = './teste_teste'
                    # if not os.path.exists(path):
                    #     os.makedirs(path)
                    # else:
                    #     shutil.rmtree(path)           # Removes all the subdirectories!
                    #     os.makedirs(path)

                    for mol in pybel.readfile('sdf', 'lig_dataset_novo.sdf'):
                        mol.write('pdbqt', 'teste_teste_NEW_CHEIC/%s.pdbqt' %mol.data['index'], overwrite=True)

                    # lig_properties = df.columns.to_list()
                    # for i, mol in enumerate(mols):
                    #     data_ref = df.iloc[i]
                    #     mol.SetProp('index', '%s' % i)
                    #     mol.SetProp('_Name', str(data_ref['id']))
                    #     for p in lig_properties:
                    #         mol.SetProp(f"+{p}", str(data_ref[p]))
                    #     sdf_writer.write(mol)

                    # sdf_writer.close()

                    # from openbabel import pybel
                    # import os
                    # # os.makedirs(out_dir_lig, exist_ok = True)
                    # import tempfile
                    # # tempfile.mkdtemp('teste_teste')
                    # # print(tempfile.mkdtemp('teste_teste'))
                    # # os.mkdir('./teste_teste_CHEIC_todas')

                    # import tempfile
                    # import os

                    # # Cria uma pasta temporária
                    # # dir_temporaria2 = tempfile.mkdtemp()
                    # pasta = os.path.join(dir_temporaria, 'pasta')

                    # # pasta = './moleculas_teste/'

                    #     # Verificar se a pasta já existe
                    # if not os.path.exists(pasta):
                    #     os.makedirs(pasta)


                    # for mol in pybel.readfile('sdf', 'lig_dataset_novo.sdf'):
                    #     mol.write('pdbqt', 'pasta/%s.pdbqt' %mol.data['index'], overwrite=True)

                    # import os
                    # from openbabel import pybel

                    # Caminho para a pasta temporária contendo o arquivo SDF
                    # dir_temporaria = "C:/Users/RAFAVI~1/AppData/Local/Temp/tmpil4bk3oo/"

                    # Caminho para a pasta onde você deseja salvar os arquivos PDBQT (dentro da pasta temporária)
                    # pdbqt_folder = os.path.join(dir_temporaria, "teste_teste")

                    # Verifica se a pasta de destino para os arquivos PDBQT existe e cria se não existir
                    # if not os.path.exists(pdbqt_folder):
                    #     os.makedirs(pdbqt_folder)

                    # Caminho completo do arquivo SDF na pasta temporária
                    sdf_file_path = os.path.join(dir_temporaria, 'lig_dataset_novo.sdf')
                    print(sdf_file_path)

                    # Lê os dados do arquivo SDF usando Pybel e escreve em formato PDBQT
                    for mol in pybel.readfile('sdf', sdf_file_path):
                        pdbqt_file_path = os.path.join(pdbqt_folder, '%s.pdbqt' % mol.data['index'])
                        print(pdbqt_file_path)
                        mol.write('pdbqt', pdbqt_file_path, overwrite=True)
                        
                    RECEPTOR = ['7P2G', '4DD8', '1NC6', '6VVU']

                    import os
                    df_docagem =[]
                    for i in  RECEPTOR:
                        WORK_DIR = dir_temporaria
                        st.markdown('dir temporaria')
                        st.write(dir_temporaria)

                        # LIG_DIR = os.path.join(WORK_DIR, 'teste_teste') #que será temporária
                        LIG_DIR = os.path.join(WORK_DIR, 'teste_teste_NEW_CHEIC')
                        RECEPTOR_DIR = os.path.join(WORK_DIR, 'receptors', i)
                        OUT_DIR = os.path.join(WORK_DIR, "runs2", i)

                        # Verifica se o diretório 'runs2' já existe, caso contrário, cria-o
                        pdbqt_folder = os.path.join(dir_temporaria, "runs2")

                        # Verifica se a pasta de destino para os arquivos PDBQT existe e cria se não existir
                        if not os.path.exists(pdbqt_folder):
                            os.makedirs(pdbqt_folder)
                        
                        
                    #     if not os.path.exists('./runs2/'):
                    #         os.mkdir('./runs2/')

                        # Verifica se o diretório específico para o receptor já existe, caso contrário, cria-o
                        if not os.path.exists(OUT_DIR):
                            os.mkdir(OUT_DIR)

                        ligands = []

                        # Loop para encontrar os arquivos de ligantes
                        for file in os.listdir(LIG_DIR):
                            if os.path.isfile(os.path.join(LIG_DIR, file)) and file.endswith('.pdbqt'):
                                ligands.append(file)
                        
                        
                        
                        prepared_lig_dirs = []
                        from shutil import copy2
                        # Loop para copiar arquivos para o diretório do receptor
                        for lig in ligands:
                            lig_filename = os.path.splitext(lig)[0]
                            out_dir_lig = os.path.join(OUT_DIR, lig_filename)
                            os.makedirs(out_dir_lig, exist_ok=True)

                            copy2(os.path.join(RECEPTOR_DIR, f"{i}final.pdbqt"), out_dir_lig)
                            copy2(os.path.join(RECEPTOR_DIR, f"{i}config.txt"), out_dir_lig)
                            copy2(os.path.join(LIG_DIR, lig), out_dir_lig)

                            prepared_lig_dirs.append(out_dir_lig)
                            
                        import shlex, subprocess
                        from datetime import datetime

                        st.write('PREPARED LIG DIRS')
                        st.write(prepared_lig_dirs)

                        # output_logs = ""

                        import subprocess
                        from datetime import datetime
                        import os
                        import streamlit as st

                        # Assuming you have the list prepared_lig_dirs

                        output_logs = ""

                        for j in prepared_lig_dirs:
                            st.write(f"\n[STARTRUN] {datetime.now()} OUTDIR {j}\n[STARTLOG]\n")

                            ligand = f"{os.path.basename(j)}.pdbqt"
                            st.write(ligand)

                            vina_exe_path = os.path.join(dir_temporaria, "Vina", "vina.exe")
                            st.write('vina')
                            st.write(vina_exe_path)

                            args = [vina_exe_path, '--receptor', f"{i}final.pdbqt", '--config', 
                                    f"{i}config.txt", '--ligand', ligand, '--log', 'results.txt']
                            st.write('args da docagem')
                            st.write(args)

                            process = subprocess.Popen(
                                args,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                cwd=os.path.join(j)
                            )

                            # Display the output dynamically
                            st_output = st.empty()
                            while process.poll() is None:
                                output = process.stdout.readline().decode("utf-8")
                                st_output.write(output)

                            output, error = process.communicate()

                            if error:
                                st.write("Error:", error.decode("utf-8"))
                                output_logs = output_logs + error.decode("utf-8")
                            else:
                                st.write("Output:", output.decode("utf-8"))
                                output_logs = output_logs + output.decode("utf-8")

                            output_logs = output_logs + f"\n[ENDLOG]\n[ENDRUN] {datetime.now()}\n+++++++++++++\n"

                            # print(i)
                            # print(len(prepared_lig_dirs))
                        # for j in prepared_lig_dirs:

                        #     output_logs = output_logs + f"\n[STARTRUN] {datetime.now()} OUTDIR {j}\n[STARTLOG]\n"

                        #     ligand = f"{os.path.basename(j)}.pdbqt"

                        #     st.markdown('VENDO OS LIGAND IN OS.PATH.BASENAME')
                        #     st.write(ligand)
                        #     #         vina = r'C:\Users\Rafa Vieira\Desktop\New_CHEIC\Vina\vina.exe'

                        #     #args = shlex.split(f"{vina} --config_7P2G.txt --ligand {ligand}")
                        #     # pdbqt = i+'final.pdbqt'
                        #     # config = i+'config.txt'

                        #     vina_exe_path = os.path.join(dir_temporaria, "Vina", "vina.exe")
                        #     st.markdown('vina')
                        #     st.write(vina_exe_path)
                        #     print(f"{i}final.pdbqt")
                        #     args = [vina_exe_path, '--receptor', f"{i}final.pdbqt", '--config', 
                        #             f"{i}config.txt", '--ligand', ligand, '--log', 'results.txt']
                        #     st.markdown('args da docagem')
                        #     st.write(args)
                        #     #args[0] = 'C:\Program Files (x86)\The Scripps Research Institute\Vina\vina.exe'
                        #     process = subprocess.Popen(
                        #         args,
                        #         stdout=subprocess.PIPE,
                        #         stderr=subprocess.PIPE,
                        #         cwd=os.path.join(j)
                        #     )
                        #     output, error = process.communicate()
                        #     print("Output:", output.decode("utf-8"))
                        #     print("Error:", error.decode("utf-8"))
                        #     # process = subprocess.Popen(
                        #     # args,
                        #     # stdout = subprocess.PIPE,
                        #     # stderr = subprocess.PIPE,
                        #     # cwd = os.path.join(j)
                        #     # )
                        #     # output, error = process.communicate()

                        #     # if error:
                        #     #     print("Error: ", error.decode("utf-8"))
                        #     #     output_logs = output_logs + error.decode("utf-8")


                        #     # else:
                        #     #     print(output.decode("utf-8"))
                        #     #     output_logs = output_logs + output.decode("utf-8")
                            
                        #     output_logs = output_logs + f"\n[ENDLOG]\n[ENDRUN] {datetime.now()}\n+++++++++++++\n"

                        
                        WORK_DIR = os.path.join(dir_temporaria, 'runs2')
                        WORK_DIR = os.path.join(WORK_DIR, i)
                        ligands = []
                        for file in os.listdir(WORK_DIR):
                            ligands.append(file)
                        #         print('Este sao os ligantes')
                        #         print(ligands)
                        #         print('+++++++++++++++++++++')

                        prepared_lig_dirs = []
                        for b in ligands:
                            #lig_filename = os.path.splitext(lig)[0]
                            out_dir_lig = os.path.join(WORK_DIR, b)
                            prepared_lig_dirs.append(out_dir_lig)
                        #         print('Este sao os prepared_lig_dirs')
                        #        0 print(prepared_lig_dirs)
                        #         print('+++++++++++++++++++++')

                        endereco = []
                        for q in prepared_lig_dirs:
                            a = q + '\\results.txt'
                        #             print(a)
                            endereco.append(a)
                            
                        df_docagem1 = []

                        import pandas as pd
                        for t in endereco:

                            data = pd.read_csv(t, header = None, on_bad_lines='skip')

                            if len(data) == 12:
                                df_docagem1.append(0)

                            else:

                                best_pose = pd.DataFrame(data[0][20].split('      ')).T
                                try:
                                    a=best_pose.iat[0, 1].strip(" ")
                                except:
                                    a=0
                        #             a=best_pose.iat[0, 1].strip(" ")


                                df_docagem1.append(a)
                        df_docagem.append(df_docagem1)



                        # import os
                        # # Diretório base
                        # base_path = dir_temporaria

                        # # Loop para renomear a pasta de acordo com cada nome

                        # old_path = os.path.join(base_path, 'runs2')
                        # new_path = os.path.join(base_path, i)
                        # os.rename(old_path, new_path)
                                                
                        
                    docking = pd.DataFrame(df_docagem).T

                    for i, j in enumerate(RECEPTOR):
                                # print(i, j)
                                #a = j
                        docking.rename(columns = {i: j}, inplace=True)

                    # docking.reset_index(inplace=True)
                    # st.dataframe(docking)

                    docking['index'] = ligands
                    docking['index'] = docking['index'].astype(int)
                                                
                    df.reset_index(inplace=True)
                    anotacoes_com_docagem = df.merge(docking, how='inner', on='index')
                    anotacoes_com_docagem

                    import shutil
                    try:
                        shutil.rmtree(dir_temporaria)
                        print("Pasta temporária removida.")
                    except:
                    #     shutil.rmtree(dir_temporaria)
                        print("Pasta nao removida.")

                st.write('Results with poses:')
                st.dataframe(anotacoes_com_docagem)


                @st.cache_data
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df(anotacoes_com_docagem)

                st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='amostras_machine_learning_docking.csv',
                mime='text/csv',
                )


                import shutil
                # shutil.rmtree(r'C:\Users\PC\Desktop\Projeto-Streamlit-MVOCS\\teste_teste')
                # shutil.rmtree(r'C:\Users\PC\Desktop\Projeto_final\runs2')
                # st.write('RESULTS', i)


                st.markdown('#### Docking results - 7P2G - SARS-Cov-2')
                df = anotacoes_com_docagem

                from plotly.subplots import make_subplots
                import plotly.graph_objects as go
                df = df[['id', '7P2G', '4DD8', '1NC6', '6VVU']]

                import plotly.graph_objects as go
                df = df[['id', '7P2G', '4DD8', '1NC6', '6VVU']]
                # df = df[['compound_name', '7P2G']]
                colors = ['lightslategray',] * len(df)
                
                novo = df.loc[df['7P2G'].astype(float)<=-6.8]
                indices = novo.index
                valores = indices.values
                for i in valores:
                    colors[i] = 'crimson'
                    
                fig = make_subplots(rows=1, cols=1, shared_yaxes=True, subplot_titles=("7P2G (SARS-CoV-2)"))

                fig = go.Figure(data=[go.Bar(
                    x=df['id'],
                    y=df['7P2G'],
                    marker_color=colors # marker color can be a single color value or an iterable
                )])

                fig.add_hline(y=-6.8, line_width=3, line_dash = 'dash', line_color = 'black', annotation_text = "-6.8 baseline 7P2G")
                fig.update_layout(title_text='Protein 7P2G - Results Docking')
                st.plotly_chart(fig)

                st.markdown('#### Docking results - 4DD8 - Asthma')
                from plotly.subplots import make_subplots
                import plotly.graph_objects as go
                # df = df[['compound_name', '7P2G', '4DD8', '1NC6', '6VVU']]
                import plotly.graph_objects as go
                df = df[['id', '7P2G', '4DD8', '1NC6', '6VVU']]
                colors = ['lightslategray',] * len(df)
                novo = df.loc[df['4DD8'].astype(float)<=-7.0]

                indices = novo.index
                valores = indices.values
                for i in valores:
                    colors[i] = 'sienna'
                    
                fig = make_subplots(rows=1, cols=1, shared_yaxes=True, subplot_titles=("4DD8 (Asthma)"))

                fig = go.Figure(data=[go.Bar(
                    x=df['id'],
                    y=df['4DD8'],
                    marker_color=colors # marker color can be a single color value or an iterable
                )])

                fig.add_hline(y=-7.0, line_width=3, line_dash = 'dash', line_color = 'black', annotation_text = "-7.0 baseline 4DD8")
                fig.update_layout(title_text='Protein 4DD8 - Results Docking')
                st.plotly_chart(fig)

                st.markdown('#### Docking results - 1NC6 - Asthma')
                from plotly.subplots import make_subplots
                import plotly.graph_objects as go
                # df = df[['compound_name', '7P2G', '4DD8', '1NC6', '6VVU']]
                import plotly.graph_objects as go
                df = df[['id', '7P2G', '4DD8', '1NC6', '6VVU']]
                colors = ['lightslategray',] * len(df)
                novo = df.loc[df['1NC6'].astype(float)<=-6.5]
                indices = novo.index
                valores = indices.values
                for i in valores:
                    colors[i] = 'mediumaquamarine'
                    
                fig = make_subplots(rows=1, cols=1, shared_yaxes=True, subplot_titles=("1NC6 (Asthma)"))

                fig = go.Figure(data=[go.Bar(
                    x=df['id'],
                    y=df['1NC6'],
                    marker_color=colors # marker color can be a single color value or an iterable
                )])

                fig.add_hline(y=-6.5, line_width=3, line_dash = 'dash', line_color = 'black', annotation_text = "-6.5 baseline 1NC6")
                fig.update_layout(title_text='Protein 1NC6 - Results Docking')
                st.plotly_chart(fig)

                st.markdown('#### Docking results - 6VVU - Asthma')
                from plotly.subplots import make_subplots
                import plotly.graph_objects as go
                # df = df[['compound_name', '7P2G', '4DD8', '1NC6', '6VVU']]
                import plotly.graph_objects as go
                df = df[['id', '7P2G', '4DD8', '1NC6', '6VVU']]
                colors = ['lightslategray',] * len(df)
                novo = df.loc[df['6VVU'].astype(float)<=-4.8]
                indices = novo.index
                valores = indices.values
                for i in valores:
                    colors[i] = 'steelblue'
                    
                fig = make_subplots(rows=1, cols=1, shared_yaxes=True, subplot_titles=("6VVU (Asthma)"))

                fig = go.Figure(data=[go.Bar(
                    x=df['id'],
                    y=df['6VVU'],
                    marker_color=colors # marker color can be a single color value or an iterable
                )])

                fig.add_hline(y=-4.8, line_width=3, line_dash = 'dash', line_color = 'black', annotation_text = "-4.8 baseline 6VVU")
                fig.update_layout(title_text='Protein 6VVU - Results Docking')
                st.plotly_chart(fig)

        else:
            st.write('carregue seus dados')

    elif choice == ("Data Molecular Dynamics"):
        data_file_Delta = st.file_uploader("prot-lig_Normal_GB_Delta.csv",type=['csv'], accept_multiple_files=False)
        data_file_Delta_padrao = st.file_uploader("prot-lig_Normal_GB_Delta. (PATTERN)",type=['csv'], accept_multiple_files=False)
        data_file_Total = st.file_uploader("prot-lig_Normal_GB_Delta_TOTAL.csv",type=['csv'], accept_multiple_files=False)
        data_file_Total_padrao = st.file_uploader("prot-lig_Normal_GB_Delta_TOTAL.csv (PATTERN)",type=['csv'], accept_multiple_files=False) 
        data_file_Hydrog = st.file_uploader("hydrogen2-bond-protein-ligand.xvg",type=['xvg'], accept_multiple_files=False)
        data_file_Hydrog_padrao = st.file_uploader("hydrogen2-bond-protein-ligand.xvg (PATTERN)",type=['xvg'], accept_multiple_files=False)
        data_file_RMSD = st.file_uploader("RMSD.xvg",type=['xvg'], accept_multiple_files=False)
        data_file_RMSD_padrao = st.file_uploader("RMSD.xvg (PATTERN)",type=['xvg'], accept_multiple_files=False)
        # data_file_Temp = st.file_uploader("temperature.xvg",type=['xvg'], accept_multiple_files=False)
        # data_file_Temp_padrao = st.file_uploader("temperature.xvg (PATTERN)",type=['xvg'], accept_multiple_files=False)

        
        if st.button("Exploratory Analysis"):
                if data_file_Delta is not None:
                    st.write('Creating graphs')

                    #mvocs
                    import pandas as pd
                    mVOC_delta = pd.read_csv(data_file_Delta)
                    mVOC_Total = pd.read_csv(data_file_Total)
                    mVOC_H = pd.read_csv(data_file_Hydrog, sep='\s+', skiprows=25, header=None, names=['Time', 'Energy'])
                    # mVOC_T = pd.read_csv(data_file_Temp, sep='\s+', skiprows=25, header=None, names=['Time', 'Energy'])
                    mVOC_RMSD = pd.read_csv(data_file_RMSD, sep='\s+', skiprows=27, header=None, names=['Time', 'Energy'])
                    

                    #Patterns
                    padrao_delta = pd.read_csv(data_file_Delta_padrao)
                    padrao_Total = pd.read_csv(data_file_Total_padrao)
                    padrao_H = pd.read_csv(data_file_Hydrog_padrao, sep='\s+', skiprows=25, header=None, names=['Time', 'Energy'])
                    # padrao_T = pd.read_csv(data_file_Temp_padrao, sep='\s+', skiprows=25, header=None, names=['Time', 'Energy'])
                    padrao_RMSD = pd.read_csv(data_file_RMSD_padrao, sep='\s+', skiprows=27, header=None, names=['Time', 'Energy'])

                    ###IMPORT LIBRARIES
                    import pandas as pd
                    from comparando_decomposicao import comparando_decomposicao
                    from estatistica_comparacao import estatistica_comparacao
                    from grafico_radar import grafico_radar
                    from grafico_rmsd import grafico_rmsd
                    from grafico_temperatura import grafico_temperatura
                    from grafico_total import grafico_total
                    from ligacao_hidrogenio import ligacao_hidrogenio
                    from mapa_calor import mapa_calor
                    from comparation_statistics import perform_ttest, estatistica_comparacao
                    from abrindo_decomposicao import abrindo_decomposicao
                    from grafico_decomposicao import grafico_residuos
                    from preparando_decomposicao import preparando_decomposicao


                    grafico_radar(padrao_delta, mVOC_delta, 'Pattern - 1NC6', 'mVOC_21')
                    grafico_total (padrao_Total, mVOC_Total, 'Pattern - 1NC6', 'mVOC_21')
                    mapa_calor (padrao_Total, mVOC_Total, 'Pattern - 1NC6', 'mVOC_21') 
                    grafico_rmsd (padrao_RMSD, mVOC_RMSD, 'Pattern - 1NC6', 'mVOC_21')
                    # print('________________________')
                    # print('#########################')
                    # print('DECOMPOSIÇÃO DOS RESÍDUOS DE AMINOÁCIDOS - PADRAO')
                    # print('#########################')
                    # # print(comparando_decomposicao (padrao_decomposicao_TDS, mVOC_decomposicao_TDS))
                    # # df_decomposicao = abrindo_decomposicao(nome = 'decomp_padrao_1NC6.txt')
                    # grafico_residuos(padrao_decomposicao_TDS)
                    # # grafico_decomposicao (padrao_decomposicao_TDS)
                    # print('________________________')
                    # print('#########################')

                    # print('________________________')
                    # print('#########################')
                    # print('DECOMPOSIÇÃO DOS RESÍDUOS DE AMINOÁCIDOS - mVOC')
                    # print('#########################')
                    # # print(comparando_decomposicao (padrao_decomposicao_TDS, mVOC_decomposicao_TDS))
                    # # grafico_decomposicao (mVOC_decomposicao_TDS)
                    # # df_decomposicao = abrindo_decomposicao(nome = 'mVOC21_1NC6.txt')
                    # grafico_residuos(mVOC_decomposicao_TDS)

                    # print('________________________')
                    # print('#########################')

                    # print('Comparacao H-BONDS')
                    # print('-------------------')
                    # # estatistica_comparacao (padrao_T,mVOC_T, 'temperature')
                    # ligacao_hidrogenio (padrao_H, mVOC_H, 'Pattern - 1NC6', 'mVOC_21')

                    # print('Comparacao RMSD')
                    # # print('-------------------')
                    # # estatistica_comparacao (padrao_RMSD, mVOC_RMSD, 'RMSD')

                    grafico_rmsd (padrao_RMSD, mVOC_RMSD, 'Pattern - 1NC6', 'mVOC_21')








    else:
        st.subheader("About")
        # st.dataframe(amostras)
        
        image = Image.open('rafael.png')

        st.image(image, use_column_width=True)
        # st.info("Built with Streamlit")
        st.info("Rafael Vieira")
        



if __name__ == '__main__':
    main()
