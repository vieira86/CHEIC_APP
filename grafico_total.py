def grafico_total (df_padrao_total, df_mVOC_total, pattern = 'Pattern', mVOC = 'mVOC'):
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd

    import streamlit as st

    # Dados para o primeiro gráfico de linhas - PADRÃO (TOTAL)
    x1 = df_padrao_total['Frames']
    y1 = df_padrao_total['TOTAL']
    desvio_padrao1 = round(df_padrao_total['TOTAL'].std(),2)
    media_padrao = round(df_padrao_total['TOTAL'].mean(),2)
#     media_padrao = round(media_padrao, 2)

    # Dados para o segundo gráfico de linhas - catequina
    x2 = df_mVOC_total['Frames']
    y2 = df_mVOC_total['TOTAL']
    desvio_padrao2 = round(df_mVOC_total['TOTAL'].std(),2)
    media_mVOC = round(df_mVOC_total['TOTAL'].mean(),2)
#     media_mVOC = round(media_mVOC, 2)


    # Criar as figuras para os gráficos
    fig = go.Figure()

    # Adicionar o primeiro gráfico de linhas com a faixa de erro vertical
    fig.add_trace(go.Scatter(
        x=x1, y=y1,
        mode='lines',
        
        name= pattern + '->'+ ' ' + str(media_padrao) + ' +/- ' + str(desvio_padrao1),
        error_y=dict(
            type='data',
            array=np.full(len(y1), desvio_padrao1),
            visible=False,
            color='rgba(0,176,246,0.7)',
            thickness=1,
            width=5
        )
    ))

    # Adicionar o segundo gráfico de linhas com a faixa de erro vertical
    fig.add_trace(go.Scatter(
        x=x2, y=y2,
        mode='lines',
        name= mVOC + '->' + ' ' + str(media_padrao) + ' +/- ' + str(desvio_padrao2),
        error_y=dict(
            type='data',
            array=np.full(len(y2), desvio_padrao2),
            visible=True,
            color='Black',
            thickness=1,
            width=5
        )
    ))


    fig.update_layout(
        xaxis_title='FRAMES',
        yaxis_title='TOTAL ENERGY',
        legend=dict(
            orientation='h',  # Legenda na horizontal (rodapé)
            yanchor='bottom',  # Ancoragem no rodapé
            y=1.02,  # Posição vertical da legenda acima do gráfico
            font=dict(
                size=18  # Tamanho da fonte da legenda (altere o valor conforme necessário)
            )
        ),
        plot_bgcolor='white',  # Definindo o fundo do gráfico como branco
        width=1000,  # Largura do gráfico em pixels
        height=600  # Altura do gráfico em pixels
    )
    # Exibir o gráfico
    # fig.show()
    st.plotly_chart(fig)
