from torch_lstm import LSTMTagger

# EMBEDDING_DIM define o tamanho do vetor da palavra.
# Neste exemplo, com vocabulário simples e um conjunto de treinamento muito limitado, 
# resolvemos manter os valores das dimensões pequenos. 
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# Instancia o modelo
model = LSTMTagger("gpu", EMBEDDING_DIM, HIDDEN_DIM, 'frases.txt')

# Obtém os dados e imprime
training_data, word2idx, tag2idx = model.read_training_data('frases.txt')
print(training_data,'\n---\n',word2idx,'\n---\n',tag2idx)

# verifica o que a função prepare_sequence faz com uma das sentenças que serão 
# utilizadas no treinamento
example_input = model.prepare_sequence("O elefante atende o telefone".lower().split(), model.word2idx)

print(example_input)

# verifica como o modelo se sai antes do treinamento
model.predict_tags("O queijo ama o elefante")

# Treina o modelo
n_epochs = 300
model.train(n_epochs)

# verifica como o modelo se sai depois de treinado
model.predict_tags("O queijo ama o elefante")

# Realiza um novo teste
model.predict_tags("O garoto falou com uma pessoa")