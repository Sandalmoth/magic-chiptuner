import torch
import random
import time


# I think I may change the data representation
# to be strings for chords
# hence each step along one dimension of the now 2d array
# would be a timestep
# this does necessitate an embedding step though
# but that should be fine (?)

# reading https://github.com/bentrevett/pytorch-seq2seq
# and my old code while implementing this


# let's make some test data
simple = [
    (64),
    (0),
    (0),
    (0),
    (68),
    (0),
    (0),
    (0),
    (64),
    (0),
    (0),
    (0),
    (68),
    (0),
    (0),
    (0),
    (66),
    (0),
    (0),
    (0),
    (66),
    (0),
    (0),
    (0),
    (64),
    (0),
    (0),
    (0),
    (64),
    (0),
    (0),
    (0),
]

full = [
    (64, 66, 68),
    (0),
    (66),
    (0),
    (68),
    (0),
    (0),
    (0),
    (64, 66, 68),
    (0),
    (66),
    (0),
    (68),
    (0),
    (0),
    (0),
    (66),
    (0),
    (66),
    (0),
    (66),
    (0),
    (66),
    (0),
    (64, 66, 68),
    (0),
    (0),
    (0),
    (64),
    (0),
    (0),
    (0),
]
# 10 magical nonsense points to anyone who can guess the song

SOS = -1
EOS = -2

simple.insert(0, (SOS))
simple.append((EOS))
full.insert(0, (SOS))
full.append((EOS))

# we need to convert to tensors in some way here

in_lang_forward = {(-2): 0, (-1): 1, (0): 2}
in_lang_backward = {0: (-2), 1: (-1), 2: (0)}
out_lang_forward = {(-2): 0, (-1): 1, (0): 2}
out_lang_backward = {0: (-2), 1: (-1), 2: (0)}

# build translation dicts
for token in simple:
    if token not in in_lang_forward:
        in_lang_forward[token] = len(in_lang_forward)
        in_lang_backward[len(in_lang_backward)] = token
for token in full:
    if token not in out_lang_forward:
        out_lang_forward[token] = len(out_lang_forward)
        out_lang_backward[len(out_lang_backward)] = token
print(in_lang_forward)
print(in_lang_backward)
print(out_lang_forward)
print(out_lang_backward)

device = torch.device('cpu')  # still dont have CUDA
# maybe i should switch to a ML package that supports OpenCL...

# translate and make tesors
src_tensor = torch.tensor(
    [in_lang_forward[x] for x in simple], dtype=torch.long, device=device
).view(-1, 1)
trg_tensor = torch.tensor(
    [out_lang_forward[x] for x in full], dtype=torch.long, device=device
).view(-1, 1)

data = [{'src': src_tensor, 'trg': trg_tensor}]

BATCH_SIZE = 1
# first of all, i dont wanna make more testa data
# second, for training on a cpu i dont believe it really matters?


class Encoder(torch.nn.Module):
    def __init__(self, inp_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = torch.nn.Embedding(inp_dim, emb_dim)
        self.rnn = torch.nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(torch.nn.Module):
    def __init__(self, out_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = torch.nn.Embedding(out_dim, emb_dim)
        self.rnn = torch.nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = torch.nn.Linear(hid_dim, out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inp, hidden, cell):
        inp = inp.unsqueeze(0)
        embedded = self.dropout(self.embedding(inp))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim
        assert encoder.n_layers == decoder.n_layers

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.out_dim

        outputs = torch.zeros(
            trg_len, batch_size, trg_vocab_size
        ).to(self.device)

        hidden, cell = self.encoder(src)
        inp = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(inp, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            inp = trg[t] if teacher_force else top1

        return outputs


INPUT_DIM = 6  # calculated by hand
OUTPUT_DIM = 7  # calculated by hand
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(count_parameters(model), "trainable parameters")
optimizer = torch.optim.Adam(model.parameters())
# gonna ignore the part about padding
# if i have padding, it'll mess up the rhythm really bad i think
criterion = torch.nn.CrossEntropyLoss()


def train(model, data, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, d in enumerate(data):
        src = d['src']
        trg = d['trg']

        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss/len(data)


def evaluate(model, data, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():

        for i, d in enumerate(data):
            src = d['src']
            trg = d['trg']

            output = model(src, trg, 0)  # no forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss/len(data)


N_EPOCHS = 100
CLIP = 1

best_valid_loss = float('inf')
print('epoch\ttrain_loss\tvalid_loss')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, data, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, data, criterion)
    end_time = time.time()

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'minimal.pt')

    print(epoch, train_loss, valid_loss, sep='\t')

# load best model and test
model.load_state_dict(torch.load('minimal.pt'))
test_loss = evaluate(model, data, criterion)
print('test_loss', test_loss)
