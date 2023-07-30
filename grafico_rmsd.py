
def grafico_rmsd(padrao_RMSD, mVOC_RMSD, PATTERN='Pattern', mVOC='mVOC'):
    import plotly.graph_objects as go
    import pandas as pd
    import streamlit as st
    
    PATTERN = PATTERN
    mVOC = mVOC

    # Dados de exemplo (substitua pelos seus dados)
    x_values = padrao_RMSD['Time']
    y_values1 = padrao_RMSD['Energy']
    y_values2 = mVOC_RMSD['Energy']

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_values, y=y_values1, fill='tozeroy', mode='lines', name=PATTERN))
    fig.add_trace(go.Scatter(x=x_values, y=y_values2, fill='tozeroy', mode='lines', name=mVOC))

    fig.update_layout(title='RMSD', xaxis_title='Time (ns)', yaxis_title='RMSD (nm)',
                      plot_bgcolor='white', paper_bgcolor='white')

    # fig.show()
    st.plotly_chart(fig)
