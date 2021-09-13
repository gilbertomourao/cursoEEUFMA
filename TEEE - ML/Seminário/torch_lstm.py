import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Define o dispositivo que será utilizado (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setenças para treinamento e as classificações de suas palavras
# ART = artigo
# NN = nome
# V = verbo
training_data = [
	("O gato comeu o queijo".lower().split(), ["ART", "NN", "V", "ART", "NN"]),
    ("Ela leu aquele livro".lower().split(), ["NN", "V", "ART", "NN"]),
    ("O cachorro ama arte".lower().split(), ["ART", "NN", "V", "NN"]),
    ("O elefante atende o telefone".lower().split(), ["ART", "NN", "V", "ART", "NN"])
]

# cria um dicionário para representação numérica das palavras
# palavras -> índices
word2idx = {}

for sent, tags in training_data:
    for word in sent:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

print(word2idx)

# cria um dicionário para representação numérica da classificação das palavras
tag2idx = {"ART": 0, "NN": 1, "V": 2}

print(tag2idx)

# Função que converte uma sequência de palavras em um tensor de valores numéricos
# Será usada para treinamento
def prepare_sequence(seq, to_idx):
    """
    Esta função recebe uma sequência de palavras e retorna um tensor 
    correspondente de valores numéricos (índices de cada palavra).
    """
    
    idxs = [to_idx[w] for w in seq]
    idxs = np.array(idxs)
    
    tensor = torch.from_numpy(idxs).long()

    tensor = tensor.to(device)

    return tensor

# verifica o que a função prepare_sequence faz com uma das sentenças que serão 
# utilizadas no treinamento
example_input = prepare_sequence("O elefante atende o telefone".lower().split(), word2idx)

print(example_input)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        ''' Initialize the layers of this model.'''
        super(LSTMTagger, self).__init__()
        
        self.hidden_dim = hidden_dim

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim).to(device)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim).to(device)

        # the linear layer that maps the hidden state output dimension 
        # to the number of tags we want as output, tagset_size (in this case this is 3 tags)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size).to(device)
        
        # initialize the hidden state (see code below)
        self.hidden = self.init_hidden()

        
    def init_hidden(self):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim).to(device),
                torch.zeros(1, 1, self.hidden_dim).to(device))

    def forward(self, sentence):
        ''' Define the feedforward behavior of the model.'''
        # create embedded word vectors for each word in a sentence
        embeds = self.word_embeddings(sentence)
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        
        # get the scores for the most likely tag for a word
        tag_outputs = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_outputs, dim=1)
        
        return tag_scores

# the embedding dimension defines the size of our word vectors
# for our simple vocabulary and training set, we will keep these small
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# instantiate our model
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2idx), len(tag2idx))

test_sentence = "O queijo ama o elefante".lower().split()

# see what the scores are before training
# element [i,j] of the output is the *score* for tag j for word i.
# to check the initial accuracy of our model, we don't need to train, so we use model.eval()
inputs = prepare_sequence(test_sentence, word2idx)
tag_scores = model(inputs)
print(tag_scores)

# tag_scores outputs a vector of tag scores for each word in an inpit sentence
# to get the most likely tag index, we grab the index with the maximum score!
# recall that these numbers correspond to tag2idx = {"ART": 0, "NN": 1, "V": 2}
_, predicted_tags = torch.max(tag_scores, 1)
print('\n')
print('Predicted tags: \n',predicted_tags)

# define our loss and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# normally these epochs take a lot longer 
# but with our toy data (only 3 sentences), we can do many epochs in a short time
n_epochs = 300

for epoch in range(n_epochs):
    
    epoch_loss = 0.0
    
    # get all sentences and corresponding tags in the training data
    for sentence, tags in training_data:
        
        # zero the gradients
        model.zero_grad()

        # zero the hidden state of the LSTM, this ARTaches it from its history
        model.hidden = model.init_hidden()

        # prepare the inputs for processing by out network, 
        # turn all sentences and targets into Tensors of numerical indices
        sentence_in = prepare_sequence(sentence, word2idx)
        targets = prepare_sequence(tags, tag2idx)

        # forward pass to get tag scores
        tag_scores = model(sentence_in)

        # compute the loss, and gradients 
        loss = loss_function(tag_scores, targets)
        epoch_loss += loss.item()
        loss.backward()
        
        # update the model parameters with optimizer.step()
        optimizer.step()
        
    # print out avg loss per 20 epochs
    if(epoch%20 == 19):
        print("Epoch: %d, loss: %1.5f" % (epoch+1, epoch_loss/len(training_data)))

test_sentence = "O queijo ama o elefante".lower().split()

# see what the scores are after training
inputs = prepare_sequence(test_sentence, word2idx)
tag_scores = model(inputs)
print(tag_scores)

# print the most likely tag index, by grabbing the index with the maximum score!
# recall that these numbers correspond to tag2idx = {"ART": 0, "NN": 1, "V": 2}
_, predicted_tags = torch.max(tag_scores, 1)
print('\n')
print('Predicted tags: \n',predicted_tags)
