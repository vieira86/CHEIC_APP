def mapa_calor(df1_padrao, df2, PATTERN = 'Pattern', mVOC = 'mVOC'):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import streamlit as st
    import pandas as pd
    # Criar subplots com uma única barra de legenda
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=['Energy '+ PATTERN, 'Energy ' + mVOC])

    # Adicionar o primeiro heatmap ao subplot superior
    fig.add_trace(go.Heatmap(
        z=df1_padrao['TOTAL'],
        x=df1_padrao.index,
        y=['TOTAL'] * len(df1_padrao),
        colorscale='RdBu',
        showscale=False  # Desabilitar a barra de legenda para o primeiro heatmap
    ), row=1, col=1)

    # Remover tick labels do eixo X no subplot superior
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    # Configurações do layout do subplot superior
    fig.update_layout(
        yaxis=dict(anchor='x')
    )

    # Adicionar o segundo heatmap ao subplot inferior
    fig.add_trace(go.Heatmap(
        z=df2['TOTAL'],
        x=df2.index,
        y=['TOTAL'] * len(df2),
        colorscale='RdBu',
        colorbar=dict(x=1.0, y=0.5)  # Posicionar a barra de legenda no canto direito do subplot inferior
    ), row=2, col=1)

    # Configurações do layout do subplot inferior
    fig.update_layout(
        yaxis=dict(anchor='x')
    )

    # fig.show()
    st.plotly_chart(fig)

