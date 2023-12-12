import plotly.figure_factory as ff
import numpy as np
from numpy.random import randn
import pickle 
from keras.models import load_model


# Função para análise de sentimento de um texto usando o modelo treinado
def analyze_sentiment(text):
    loaded_model = load_model('modelo_2.h5')

    
    # Tokeniza e padroniza o texto de entrada
    words = imdb.get_word_index()
    words = {k:(v+3) for k,v in words.items()}
    words["<PAD>"] = 0
    words["<START>"] = 1
    words["<UNK>"] = 2
    words["<UNUSED>"] = 3
    
    input_text = [words.get(word, 2) for word in text.split()]
    input_text = sequence.pad_sequences([input_text], maxlen=maxlen)
    
    # Realiza a previsão de sentimento
    prediction = loaded_model.predict(input_text)
    
    # Retorna a previsão (0 para negativo, 1 para positivo)
    return "Positivo" if prediction[0][0] > 0.5 else "Negativo"

def saidas_rnn(x):
    if x == 'h':
        h = [[ 0.96821177],[-0.99998743],[-1.        ],[ 0.99999778]]
        return h
    elif x == 'y':
        y = [[ 0.90821177],[0.10298743]]
        return y


def saida_dos_pesos(input_size,hidden_size,output_size,tipo='whh'):
    np.random.seed(42)
    # Pesos
    Whh = randn(hidden_size, hidden_size) 
    Wxh = randn(hidden_size, input_size) 
    Why = randn(output_size, hidden_size)
    matriz_pra_ser_mostrada = Whh
    if tipo == 'Whh':
        matriz_pra_ser_mostrada = Whh
        xaxis = 'camada estado'
        yaxis = 'camada escondida'
        # Nomes das camadas (para o eixo x e y)
        nome_entrada = [f"neuron_{i+1+matriz_pra_ser_mostrada.shape[1]}" for i in range(matriz_pra_ser_mostrada.shape[1])]
        nome_sainda = [f"neuron_{i+1}" for i in range(matriz_pra_ser_mostrada.shape[0])]
    elif tipo == 'Wxh':
        matriz_pra_ser_mostrada = Wxh
        xaxis = 'Entradas'
        yaxis = 'Pesos'
        # Nomes das camadas (para o eixo x e y)
        nome_entrada = [f"x_{i+1}" for i in range(matriz_pra_ser_mostrada.shape[1])]
        nome_sainda = [f"neuron_{i+1}" for i in range(matriz_pra_ser_mostrada.shape[0])]
    elif tipo == 'Why':
        matriz_pra_ser_mostrada = Why
        xaxis = 'Pesos de saída h_t'
        yaxis = 'Saída'
        # Nomes das camadas (para o eixo x e y)
        nome_entrada = [f"neuron_{i+1}" for i in range(matriz_pra_ser_mostrada.shape[1])]
        nome_sainda = [f"neuron_{i+2+matriz_pra_ser_mostrada.shape[1]}" for i in range(matriz_pra_ser_mostrada.shape[0])]


    # Criar um heatmap com anotações
    fig = ff.create_annotated_heatmap(z=matriz_pra_ser_mostrada, x=nome_entrada, y=nome_sainda, colorscale='Viridis')

    # Personalizar o layout do gráfico
    fig.update_layout(
        title='Pesos da RNN',
        xaxis=dict(title=xaxis),
        yaxis=dict(title=yaxis),
        margin=dict(l=50, r=50, b=50, t=100),
    )
    return fig

code = '''
import numpy as np
from numpy.random import randn

class RNN_VANILA:
    # Vanilla RNN.
    def __init__(self, tamanho_da_entrada, tamanho_da_saida, tamaho_da_camada_oculta=64):
        # Pesos
        self.Whh = randn(tamaho_da_camada_oculta, tamaho_da_camada_oculta)
        self.Wxh = randn(tamaho_da_camada_oculta, tamanho_da_entrada)
        self.Why = randn(tamanho_da_saida, tamaho_da_camada_oculta)

        # Biases
        self.bh = np.zeros((tamaho_da_camada_oculta, 1))
        self.by = np.zeros((tamanho_da_saida, 1))
        
    def softmax(self,xs):
        return np.exp(xs) / sum(np.exp(xs))

    def predict(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))

        self.ultima_entrada = inputs
        self.ultima_hs = { 0: h }
        #passa letra por letra
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.ultima_hs[i + 1] = h

        # calculando a saida
        y = self.softmax(self.Why @ h + self.by)
        return y

    def fit(self, x_data, y_data):
        for epoch in range(500):
            loss = 0
            num_correct = 0

            for inputs, target in zip(x_data, y_data):
                # Forward
                out = self.predict(inputs)
                probs = out
                #print(out)
                # Calculate loss / accuracy
                loss -= np.log(probs[target])
                num_correct += int(np.argmax(probs) == target)
                d_L_d_y = probs
                d_L_d_y[target] -= 1

                # Backward
                self.backprop(d_L_d_y)
            erro = loss / len(x_data)
            acc = num_correct / len(x_data)
            if epoch % 50 == 0:
                print('--- Epoch %d' % (epoch + 1))
                print('Train:\tLoss %.3f | Accuracy: %.3f' % (erro, acc))
            #print(erro,acc)

    def backprop(self, d_y, learn_rate=2e-2):
        n = len(self.ultima_entrada)

        # Calculando dL/dWhy e dL/dby.
        d_Why = d_y @ self.ultima_hs[n].T
        d_by = d_y

        # Inicializando os vetores dL/dWhh, dL/dWxh, and dL/dbh com zero.
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # calculando dL/dh para o ultimo  h.
        # dL/dh = dL/dy * dy/dh
        d_h = self.Why.T @ d_y

        # retropropagando pra cada t
        for t in reversed(range(n)):
            # ...: dL/dh * (1 - h^2)
            temp = ((1 - self.ultima_hs[t + 1] ** 2) * d_h)

            # dL/db = dL/dh * (1 - h^2)
            d_bh += temp

            # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            d_Whh += temp @ self.ultima_hs[t].T

            # dL/dWxh = dL/dh * (1 - h^2) * x
            d_Wxh += temp @ self.ultima_entrada[t].T

            # Next dL/dh = dL/dh * (1 - h^2) * Whh
            d_h = self.Whh @ temp

        # cortando para evitar explosao do gradiente 
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        # Atualizando os pesos usando o gradiente descente.
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by
'''


#st.latex(r'''
#h_{t} = tanh(W_{hh}*h_{t-1} + W_{xh}*x_{t})\\
#''')

def explicacao_rnn(st):
    side_selecao = st.sidebar.selectbox('Selecione uma visualizacao: ',['Explicacao','Aplicacao'])
    if side_selecao == 'Explicacao':
        selecao_radio = st.radio(
            "Selecione uma opcao 👇",
            ["Teoria", "Código", "Resumo"],
            key="visibility",horizontal=True
        )

        if selecao_radio == 'Teoria':
            st.markdown('''
            # Redes Neurais Recorrentes (RNNs)
            As RNNs são um tipo de arquitetura de rede neural que é especialmente adequada para lidar com dados sequenciais ou temporais, como texto, áudio, séries temporais etc. As RNNs possuem conexões retroativas, permitindo que informações sejam mantidas e processadas ao longo do tempo.
            
            # Estrutura Básica
            Uma RNN consiste em unidades (neurônios) interconectadas, onde cada unidade possui uma memória interna (estado oculto). A entrada em uma RNN é não apenas a entrada atual, mas também o estado oculto da iteração anterior. Isso permite que as RNNs capturem padrões temporais e dependências de longo prazo nos dados de entrada.
            ''')
            selecao_radio_estrutura_basica = st.radio(
                "",
                ["Imagem", "Algoritmo"],
                horizontal=True
            )
            if selecao_radio_estrutura_basica == 'Imagem':
                selecao_imagem= st.selectbox(
                "",
                ["V1", "V2"]
                )
                if selecao_imagem == 'V1':
                    st.image('fig5.png')
                elif selecao_imagem == 'V2':
                    st.image('fig6.png')
                
            elif selecao_radio_estrutura_basica == 'Algoritmo':
                st.code('''
                    # Inicialização aleatória dos pesos com média zero
                    np.random.seed(42)
                    tamanho_da_entrada,tamaho_da_camada_oculta,tamanho_da_saida = 3,4,1
                    # Pesos
                    self.Whh = randn(tamaho_da_camada_oculta, tamaho_da_camada_oculta)
                    self.Wxh = randn(tamaho_da_camada_oculta, tamanho_da_entrada)
                    self.Why = randn(tamanho_da_saida, tamaho_da_camada_oculta)

                    # Biases
                    self.bh = np.zeros((tamaho_da_camada_oculta, 1))
                    self.by = np.zeros((tamanho_da_saida, 1))
                ''',language='python'
                )
                selecao_pesos = st.selectbox('selecione um peso:',['Wxh','Whh','Why'])
                input_size,hidden_size,output_size = 3,4,1
                st.write(saida_dos_pesos(input_size,hidden_size,output_size,selecao_pesos))
                st.markdown('''
                    Os pesos sao inicializados de forma aleatória. Treinamento da RNN por uma única iteração (um passo de tempo)
                ''')
                st.code('''
                    x = input_seq[:, 0][:, None]  # Entrada no primeiro passo de tempo
                    h_prev = np.zeros((hidden_size, 1))  # Estado oculto anterior
                ''',language='python'
                )
                selecao_inputs = st.selectbox('selecione um valor:',['x','h_prev'])
                if selecao_inputs == 'x':
                    inputs = np.array([list(map(int,map(abs,np.random.randn(input_size, 1)*10))) for _ in range(2)])
                    st.write(list(inputs))
                elif selecao_inputs == 'h_prev':
                    h_prev = np.zeros((hidden_size, 1)) 
                    st.write(list(h_prev))

                st.markdown('''
                A propagação para frente (forward propagation) em uma Rede Neural Recorrente (RNN) refere-se ao processo pelo qual a RNN calcula suas previsões para uma determinada entrada ou sequência de entrada
                ''')
                st.latex(''' 
                h_{t} = tanh( W_{xh}*x_{t} + W_{hh}*h_{t-1} )
                ''')

                st.code('''
                    # Forward propagation
                    h = tanh(np.dot(Wxh, x) + np.dot(Whh, h_prev) + bh)
                    y = softmax(np.dot(Why, h)) + by # dependendo da implementacao é opcionall incluir a softmax aqui 
                ''',language='python'
                )
                selecao_saidas = st.selectbox('selecione uma saida:',[
                    'h','y','np.dot(Wxh, x)','np.dot(Whh, h_prev)','bh','tanh(np.dot(Wxh, x) + np.dot(Whh, h_prev) + bh)'
                    ]
                    )
                
                st.write(saidas_rnn(selecao_saidas))
                


        elif selecao_radio == 'Código':
            st.write('---')
            st.markdown(''' # Tratamento de dados''')
            st.code('''
                import numpy as np
                def criando_as_entradas(text):
                    """ 
                        Criando uma matriz one-hot
                    """
                    vocabulario = []
                    for i in train_data.keys():
                        vocabulario += i.split()
                    vocabulario = list(np.unique(vocabulario))
                    vocab_size = len(vocabulario)

                    word_to_idx = { w: i for i, w in enumerate(vocabulario) }
                    idx_to_word = { i: w for i, w in enumerate(vocabulario) }

                    inputs = []
                    for w in text.split(' '):
                        v = np.zeros((vocab_size, 1))
                        v[word_to_idx[w]] = 1
                        inputs.append(v)
                    return inputs
            ''', language='python')
            st.write('---')
            st.markdown(''' # Modelo da RNN VANILA''')
            st.code(code, language='python')
            st.markdown(''' # Importando os dados e a RNN''')
            st.code('''
            import numpy as np
            import random
            from RNN_class import RNN_VANILA
            from dados import train_data, test_data,criando_as_entradas
            items = list(train_data.items())
            random.shuffle(items)
            x_train, y_train = [],[]
            #ajustando os dados
            for x, y in items:
                inputs = criando_as_entradas(x)
                target = int(y)
                x_train.append(inputs)
                y_train.append(target)
            tamanho_da_entrada, tamanho_da_saida, tamaho_da_camada_oculta=21,2,64
            model = RNN_VANILA(tamanho_da_entrada, tamanho_da_saida, tamaho_da_camada_oculta)
            model.fit(x_train,y_train)
            
            ''', language='python')
            st.code('''
            tamanho_da_entrada, tamanho_da_saida, tamaho_da_camada_oculta=21,2,64
            model = RNN_VANILA(tamanho_da_entrada, tamanho_da_saida, tamaho_da_camada_oculta)
            model.fit(x_train,y_train)
            
            y_preds = []
            y_true = []
            for x,y in zip(x_test,y_test):
                y_pred = model.predict(x)
                y_preds.append(np.argmax(y_pred))
                y_true.append(y)
            ''', language='python')
        elif selecao_radio == 'Resumo':
            st.write('---')
            st.markdown('''
                        Podemos definir (de forma simplificada) a RNN como três matrizes: $W_{xh}$, $W_{hh}$, $W_{hy}$. que definem os pesos da RNN;
                        - $W_{xh}$ são os pesos da entrada para os neuronios da primeira camada;
                        - $W_{hh}$ é os pesos da camada de estado, que recebe a saída da camada escondida;
                        - $W_{hy}$ é os pesos da saída, que saí dos neuronios da camada oculta e vai para a saída.

                        Na equacao a seguir, podemos calcular o $h_{t}$ que é o estado atual e vai servir para calcular o y estimado.
                        ''')
            
            st.latex(r'''
            h_{t} = tanh(W_{hh}*h_{t-1} + W_{xh}*x_{t} + bias)\\
            ''')
            st.markdown('''
                        Em que 
                        - $h_{t-1}$ é o estado anterior;
                        - $x_{t}$ são os dados de entrada;

                        Para calcular o $y$, podemos seguir a seguinte equacao:
                          
                        ''')
            st.latex(r'''
                y_{t} = W_{hy}*h_{t}+bias\\
                ''')
            st.markdown('''
                        Até esse ponto, temos a representacao da rede, apenas para frente, ou seja, o processo de treinamento ainda nao foi ilustrado até aqui.  
                        Para fazer o ajuste dos pesos, temos que calcular a perda, usando uma funcao de perda. 
                        
                        ''')
            #st.latex(r'''
            #            \frac{{\partial E}}{{\partial Whh}} = \sum_t \left(\frac{{\partial E}}{{\partial h_t}} \cdot \frac{{\partial h_t}}{{\partial Whh}}\right)
            #            ''')
            st.markdown('''
                        -  Derivada em relação ao peso da camada oculta Whh:
                        - Derivada em relação ao peso da camada de entrada Wxh:
                        - Derivada em relação ao bias da camada oculta bh:
                        - Derivada em relação ao estado oculto h_t:
                        - Derivada em relação à entrada x:

                        Essas derivadas são calculadas ao longo do tempo, retrocedendo na sequência de passos da RNN.
                        ''')
            
            st.write('---')
    elif side_selecao == 'Aplicacao':
        st.write('---')
        st.write('# APLICACAO')
        selecao_radio_aplicacao = st.radio(
            "Selecione uma opcao 👇",
            ["APP", "Código"],
            key="visibility",horizontal=True
        )

        if selecao_radio_aplicacao == 'APP':
            texto_box = st.text_input('Digite sua frase: ')
            # Exemplo de uso da função para análise de sentimento
            #texto_usuario = input("Digite um texto para análise de sentimento: ")
            if texto_box:
                st.write(texto_box)
                sentimento = analyze_sentiment(texto_box)
                print("Sentimento do texto:", sentimento)
        elif selecao_radio_aplicacao == 'Código':
            st.write('### Ajustando os dados')
            st.code('''
            import numpy as np
            from tensorflow.keras.datasets import imdb
            from tensorflow.keras.preprocessing import sequence
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

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


            ''', language='python')
            st.write('### Criando o modelo')
            st.code('''

                    # Constrói o modelo RNN muitos-para-um para análise de sentimento
                    model = Sequential()
                    model.add(Embedding(max_features, 32))  # Projeta as palavras em vetores de 32 dimensões
                    model.add(SimpleRNN(32))  # Camada SimpleRNN com 32 unidades
                    model.add(Dense(1, activation='sigmoid'))  # Camada de saída com uma única unidade e função de ativação sigmoid para classificação binária

                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                    # Treina o modelo
                    model.fit(input_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.2)

                    # Avalia o modelo no conjunto de teste
                    loss, accuracy = model.evaluate(input_test, y_test)
                    print("Loss:", loss)
                    print("Accuracy:", accuracy)

                    ''', language='python')
    st.write('---')
    st.write('')
    
    st.markdown('''
    - https://towardsdatascience.com/implementing-recurrent-neural-network-using-numpy-c359a0a68a67
    - https://www.perplexity.ai/
    - https://github.com/vzhou842/rnn-from-scratch
    - https://victorzhou.com/blog/intro-to-rnns/#7-the-backward-phase
    - https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    - https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/
    - https://imasters.com.br/data/um-mergulho-profundo-nas-redes-neurais-recorrentes
    - https://acervolima.com/como-implementar-uma-descida-gradiente-em-python-para-encontrar-um-minimo-local/
    - https://chat.openai.com/
    ''')

