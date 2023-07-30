def grafico_radar (A, B, pattern_name = 'Pattern', mVOC_name = 'mVOC'):
    import streamlit as st
    import pandas as pd
    lista = [A,B]
    
    mVOC_name = mVOC_name
    pattern_name = pattern_name
    dataframe = []
    for i in lista:
    #     print(i)
        a = i.T[[0,1]]
    #     a['label'] = 
        a = pd.DataFrame(a)
        a.reset_index(inplace=True)
        a.rename(columns = {'index': 'metric', 0:'value', 1: 'stand_desv'}, inplace=True)

        a = a.iloc[1:,:]

        dataframe.append(a)

    nomes = [pattern_name, mVOC_name]

    for df, nome in zip(dataframe, nomes):
        # Adicione a coluna "nome" com o valor do nome em todas as linhas
        df['molecule'] = nome

    df = pd.DataFrame()
    for i in dataframe:
        df = df.append(i)
        
    import plotly.express as px

    colors = ['blue', 'red']

    fig = px.line_polar(df, r='value', theta="metric", color="molecule", line_close=True,
                        color_discrete_sequence=colors)
    fig.show()
    
    fig = px.bar(df, x="metric", y='value', color="molecule", title="MMPBSA",  barmode='group', text_auto='.3s', error_y='stand_desv')

    # fig.update_traces(textposition='inside')

    # Definindo a cor do fundo do gráfico
    fig.update_layout(
        plot_bgcolor='rgb(250, 250, 250)'
    )

    # Layout do gráfico
    fig.update_layout(
        title='',
        xaxis_title='METRICS',
        yaxis_title='TOTAL ENERGY',
        legend=dict(
            orientation='h',  # Legenda na horizontal (rodapé)
            yanchor='bottom',  # Ancoragem no rodapé
            y=1.02,  # Posição vertical da legenda acima do gráfico
            font=dict(
                size=20  # Tamanho da fonte da legenda (altere o valor conforme necessário)
            )
        ),
        plot_bgcolor='white',  # Definindo o fundo do gráfico como branco
        width=800,  # Largura do gráfico em pixels
        height=800  # Altura do gráfico em pixels
    )

    # Aumentando o tamanho dos valores dentro do gráfico
    fig.update_traces( textfont_size=16)

    # Aumentando o tamanho dos eixos
    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=18),
            title_font=dict(size=18)
        ),
        yaxis=dict(
            tickfont=dict(size=18),
            title_font=dict(size=18)
        )
    )


    fig.update_layout(
        shapes=[
            dict(
                type="line",
                xref="paper",  # Referência do eixo X é o papel (eixo X relativo)
                x0=0,
                x1=1,
                yref="y",  # Referência do eixo Y é o próprio eixo Y
                y0=0,
                y1=0,
                line=dict(
                    color="black",  # Cor da linha (aqui, definida como vermelha)
                    width=2,  # Largura da linha
                    dash="dash"  # Estilo da linha (aqui, definido como tracejado)
                )
            )
        ]
    )

    # fig.show()
    st.plotly_chart(fig)

    # return df