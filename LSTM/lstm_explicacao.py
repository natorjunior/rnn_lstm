from LSTM.passo_a_passo import *
import numpy as np
import plotly.graph_objects as go

def sigmoid_view(x):
    # Calculando os valores da função sigmoide
    sigmoid_values = 1 / (1 + np.exp(-x))
    return sigmoid_values

def sigmoid_derivative(x):
    # Calculando os valores da derivada da função sigmoide
    sigmoid_values = sigmoid_view(x)
    sigmoid_derivative_values = sigmoid_values * (1 - sigmoid_values)
    return sigmoid_derivative_values

def than_view(x):
    # Calculando os valores da função tangente hiperbólica
    tanh_values = np.tanh(x)
    return tanh_values

def than_derivative(x):
    # Calculando os valores da derivada da função tangente hiperbólica
    tanh_values = than_view(x)
    than_derivative_values = 1 - np.square(tanh_values)
    return than_derivative_values
# Criando valores para o eixo x
x = np.linspace(-7, 7, 400)

def explicacao_lstm(st):
    side_selecao = st.sidebar.selectbox('Selecione uma visualizacao: ',['Explicacao','Aplicacao'])
    if side_selecao == 'Explicacao':
        sidebar_op1 = st.sidebar.selectbox('Selecione uma parte:',['Passo a passo','funcoes de ativacao e derivadas'])
        selecao_radio_estrutura_basica = st.radio(
                "",
                ["Teoria", "Algoritmo"],
                horizontal=True
            )
        if selecao_radio_estrutura_basica == 'Teoria':
            if sidebar_op1 == 'Passo a passo':
                if st.session_state.page_num == 1:
                    page_one(st)
                elif st.session_state.page_num == 2:
                    page_two(st)
                elif st.session_state.page_num == 3:
                    page_tres(st)
                elif st.session_state.page_num == 4:
                    page_4(st)
                elif st.session_state.page_num == 5:
                    page_5(st)
                elif st.session_state.page_num == 6:
                    page_6(st)
                elif st.session_state.page_num == 7:
                    page_7(st)
        elif selecao_radio_estrutura_basica == 'Algoritmo':
            st.code('''
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle 

# Carrega os dados IMDB
max_features = 10000  # Considera apenas as 10,000 palavras mais frequentes
maxlen = 100  # Limita as avaliações a 100 palavras
batch_size = 32

print('Carregando dados...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'sequências de treino')
print(len(input_test), 'sequências de teste')

# Preprocessa os dados
print('Padronizando sequências...')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# Construindo o modelo LSTM
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compilando o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando o modelo
print('Treinando o modelo...')
model.fit(input_train, y_train, epochs=5, batch_size=batch_size, validation_split=0.2)

# Avaliando o modelo
score, accuracy = model.evaluate(input_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', accuracy)

            ''',language='python'
            )
        
        if sidebar_op1 == 'funcoes de ativacao e derivadas':
            sidebar_op2 = st.selectbox('Selecione uma funcao de ativacao:',['Sigmoid','than'])
            if sidebar_op2 == 'Sigmoid':
                
                # Calculando os valores das funções e de suas derivadas
                sigmoid_values = sigmoid_view(x)
                sigmoid_derivative_values = sigmoid_derivative(x)
                
                # Criando o gráfico usando Plotly
                fig = go.Figure()
                # Adicionando a curva da sigmoide
                fig.add_trace(go.Scatter(x=x, y=sigmoid_values, mode='lines', name='Sigmoid'))
                fig.add_trace(go.Scatter(x=x, y=sigmoid_derivative_values, mode='lines', name='Sigmoid Derivada'))

                # Personalizando o layout do gráfico
                fig.update_layout(title='Funções Sigmóide',
                                xaxis_title='x',
                                yaxis_title='Valor da Função',
                                legend=dict(x=0, y=1, traceorder='normal'))
                st.plotly_chart(fig,use_container_width=False)
            if sidebar_op2 == 'than':
                tanh_values = than_view(x)
                tanh_derivative_values = than_derivative(x)
                fig = go.Figure()
                # Adicionando a curva da tangente hiperbólica
                fig.add_trace(go.Scatter(x=x, y=tanh_values, mode='lines', name='Tanh'))
                fig.add_trace(go.Scatter(x=x, y=tanh_derivative_values, mode='lines', name='Tanh Derivada'))


                # Personalizando o layout do gráfico
                fig.update_layout(title='Tangente Hiperbólica',
                                xaxis_title='x',
                                yaxis_title='Valor da Função',
                                legend=dict(x=0, y=1, traceorder='normal'))
                st.plotly_chart(fig,use_container_width=False)
