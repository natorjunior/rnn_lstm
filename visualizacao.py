import streamlit as st
import numpy as np
import plotly.graph_objects as go

from LSTM.lstm_explicacao import *
from RNN.rnn_explicacao import *
# Verifique qual página deve ser exibida com base no estado da sessão
if "page_num" not in st.session_state:
    st.session_state.page_num = 1

st.sidebar.title('RNN - LSTM')
side_selecao1 = st.sidebar.selectbox('Selecione uma rede: ',['RNN','LSTM'])
if side_selecao1 == 'LSTM':
    explicacao_lstm(st)
elif side_selecao1 == 'RNN':
    explicacao_rnn(st)
