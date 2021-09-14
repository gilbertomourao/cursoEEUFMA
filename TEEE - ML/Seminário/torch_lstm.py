"""
RNN com LSTM

Autor original: LeanManager
Ver: https://github.com/LeanManager/NLP-PyTorch/blob/master/LSTM%20Speech%20Tagging%20with%20PyTorch.ipynb

Modificações por: Gilberto José Guimarães de Sousa Mourão

O objetivo deste script é auxiliar uma apresentação da disciplina Aprendizado de Máquina na Universidade 
Federal do Maranhão (UFMA). Portanto, ele será utilizado somente no mundo acadêmico, sem qualquer intenção 
comercial envolvida. 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import ast
import warnings

class LSTMTagger(nn.Module):

    def __init__(self, device, embedding_dim, hidden_dim, training_data_file):
        ''' Incializa as camadas do modelo'''
        super(LSTMTagger, self).__init__()

        # verifica se device está ok
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Verifica se encontrou uma gpu disponível
            if self.device == "cpu":
                warnings.warn('ATENÇÃO: Nenhuma gpu foi encontrada!')

        elif device == "cpu":
            self.device = "cpu"
        else:
            raise ValueError('Nenhum dispositivo com essa nomenclatura pode ser reconhecido pelo modelo.')

        # A inicialização do modelo já armazena os dados de treinamento nele. 
        self.training_data, self.word2idx, self.tag2idx = self.read_training_data(training_data_file)

        # Obtém o tamanho do vocabulário, ou seja, a quantidade de palavras 
        # contempladas pelo modelo.
        vocab_size = len(self.word2idx)

        # Obtém o tamanho do conjunto de tags, ou seja, a quantidade de tags 
        # contempladas pelo modelo. 
        tagset_size = len(self.tag2idx)
        
        # Inicializa a dimensão do hidden layer
        self.hidden_dim = hidden_dim

        # Camada de associação que obtém vetores de um tamanho específico a partir
        # de uma palavra
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim).to(self.device)

        # A LSTM recebe os vetores da camada de associação como entrada e sai 
        # com os hidden states de tamanho hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim).to(self.device)

        # Camada linear que mapeia a dimensão do hidden state para o número 
        # de tags que se deseja obter na saída, ou seja, tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size).to(self.device)
        
        # Inicializa o hidden state
        self.hidden = self.init_hidden()

        
    def init_hidden(self):
        '''
        No início do treinamento, o modelo precisa inicializar o hidden state. 
        Como ele é formado basicamente por dados anteriores (h[t-1]), no início 
        não haverá qualquer hidden state. Esta função define um hidden state 
        inicial como um tensor de zeros.
        '''
        # As dimensões dos tensores são (n_layers, batch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim).to(self.device),
                torch.zeros(1, 1, self.hidden_dim).to(self.device))

    def forward(self, sentence):
        ''' Define o comportamento feedfoward do modelo '''
        # cria vetores para cada palavra na sentença através da camada de associação (embedding)
        embeds = self.word_embeddings(sentence)
        
        # obtém a saída e o hidden state aplicando os vetores das embedded words 
        # e o hidden state anterior.
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        
        # Obtém a pontuação para a tag mais provável utilizando a função softmax
        tag_outputs = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_outputs, dim=1)
        
        return tag_scores

    def train(self, n_epochs):
        """
        Treina o modelo
        """

        # Define a função de perda e o otimizador.
        # Nesse caso, são utilizadas as funções negative log likelihood e 
        # stochastic gradient descent.
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.1)

        for epoch in range(n_epochs):
    
            epoch_loss = 0.0
            
            # Obtém todas as sentenças e tags correspondentes nos dados de treinamento
            for sentence, tags in self.training_data:
                
                # zera os gradientes
                self.zero_grad()

                # zera o hidden state da LSTM, ou seja, reseta a história da mesma
                self.hidden = self.init_hidden()

                # prepara as entradas para o processamento pela rede, 
                # transforma todas as sentenças e targets em tensores numéricos
                sentence_in = self.prepare_sequence(sentence, self.word2idx)
                targets = self.prepare_sequence(tags, self.tag2idx)

                # forward pass para obter a pontuação das tags
                tag_scores = self(sentence_in)

                # calcula o loss e o gradiente
                loss = loss_function(tag_scores, targets)
                epoch_loss += loss.item()
                loss.backward()
                
                # atualiza os parâmetros do modelo
                optimizer.step()
                
            # Imprime a loss/perda média a cada 20 épocas.
            if(epoch%20 == 19):
                print("Epoch: %d, loss: %1.5f" % (epoch+1, epoch_loss/len(self.training_data)))

    def read_training_data(self, file):
        """
        Função para ler os dados para treinamento. As frases obtidas são divididas 
        em uma lista de palavras, onde cada palavra possui uma tag já associada. 
        """
        f = open(file)
        first_line = f.readline()

        tag2idx = ast.literal_eval(first_line)

        second_line = f.readline()
        lines = f.readlines()

        training_data = []

        for line in lines:
            line = line.split(', [')
            words = line[0].split()
            tags = ast.literal_eval('[' + line[1])
            training_data.append((words, tags))

        word2idx = {}

        for sent, tags in training_data:
            for word in sent:
                if word not in word2idx:
                    word2idx[word] = len(word2idx)

        return training_data, word2idx, tag2idx

    def prepare_sequence(self, seq, to_idx):
        """
        Esta função recebe uma sequência de palavras e retorna um tensor 
        correspondente de valores numéricos (índices de cada palavra).
        """
        
        idxs = [to_idx[w] for w in seq]
        idxs = np.array(idxs)
        
        tensor = torch.from_numpy(idxs).long()

        tensor = tensor.to(self.device)

        return tensor

    def predict_tags(self, sentence):
        """
        Prediz as tags de cada palavra de uma sentença.
        """

        seq = sentence.lower().split()

        inputs = self.prepare_sequence(seq, self.word2idx)
        tag_scores = self(inputs)
        print(tag_scores)

        # Obtém os índices com maior pontuação
        _, predicted_tags = torch.max(tag_scores, 1)

        # Converte os números em tags reais para melhor visualização

        tags = []

        for tag_id in predicted_tags:
            key = list(self.tag2idx.keys())[tag_id]
            tags.append(key)

        print('\n')
        print('Predicted tags: \n',tags)

