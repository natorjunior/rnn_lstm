import numpy as np
from numpy.random import randn

class RNN_VANILA:
    # Vanilla RNN.
    def __init__(self, tamanho_da_entrada, tamanho_da_saida, tamaho_da_camada_oculta=64):
        # Pesos
        self.Whh = randn(tamaho_da_camada_oculta, tamaho_da_camada_oculta)/1000
        self.Wxh = randn(tamaho_da_camada_oculta, tamanho_da_entrada)/1000
        self.Why = randn(tamanho_da_saida, tamaho_da_camada_oculta)/1000

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
        d_h = self.Why.T @ d_y

        # retropropagando pra cada t
        for t in reversed(range(n)):
            temp = ((1 - self.ultima_hs[t + 1] ** 2) * d_h)
            d_bh += temp
            d_Whh += temp @ self.ultima_hs[t].T
            d_Wxh += temp @ self.ultima_entrada[t].T
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