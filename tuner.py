import torch
import numpy as np
import click
import os
import random


MAX_LENGTH = 65536


# For now, I will implement seq2seq as in
# https://www.guru99.com/seq2seq-model.html
# and we'll see if/how well that works
# hence, for now, consider the Encoder, Decoder, Seq2Seq classes,
# clacModel, trainModel to
# have an ill defined license if you copy them
# as I cannot spot any particular details on the website

# TODO
# test LSTM instead of GRU?


@click.group()
def main():
    pass


def load(path, device):
    data = []
    files = sorted(os.listdir(path))
    # when sorted, the filenames will pair the simplification
    # and full song in order (correctly with transpositions)
    for ffull, fsimple in zip(files[::2], files[1::2]):
        print(fsimple, ffull)
        v_simple = np.load(path + fsimple)
        input_tensor = torch.tensor(v_simple, dtype=torch.uint8, device=device)
        v_full = np.load(path + ffull)
        target_tensor = torch.tensor(v_full, dtype=torch.uint8, device=device)
        data.append((input_tensor, target_tensor))
    random.shuffle(data)
    return data


INPUT_DIM = 256
OUTPUT_DIM = 256
HIDDEN_DIM = 512
EMBED_DIM = 256
NUM_LAYERS = 1


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.input_dim = INPUT_DIM
        self.hidden_dim = HIDDEN_DIM
        self.embed_dim = EMBED_DIM
        self.num_layers = NUM_LAYERS

        # self.embedding = torch.nn.Embedding(self.input_dim, self.embed_dim)
        self.gru = torch.nn.GRU(
            self.embed_dim, self.hidden_dim, num_layers=self.num_layers
        )

    def forward(self, src):
        embedded = src  # self.embedding(src).view(1, 1, -1)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.input_dim = INPUT_DIM
        self.output_dim = OUTPUT_DIM
        self.hidden_dim = HIDDEN_DIM
        self.embed_dim = EMBED_DIM
        self.num_layers = NUM_LAYERS

        # self.emebedding = torch.nn.Embedding(self.output_dim, self.embed_dim)
        self.gru = torch.nn.GRU(
            self.embed_dim, self.hidden_dim, num_layers=self.num_layers
        )
        self.out = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        inputs = inputs.view(1, -1)
        embedded = torch.nn.functional.relu(inputs)  # self.embedding(inputs))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))
        return prediction, hidden


class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        print(source.shape, target.shape)
        # input_length = source.size(0)
        batch_size = 1  # no batching in my data target.shape[1]
        target_length = target.shape[0]
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(
            target_length, batch_size, vocab_size
        ).to(self.device)
        # for i in range(input_length):
        #     enc_out, enc_hid = self.encoder(source[i])

        dec_hid = source  # enc_hid.to(self.device)
        # look into the 0 here
        dec_inp = torch.tensor([0], device=self.device)

        for t in range(target_length):
            dec_out, dec_hid = self.decoder(dec_inp, dec_hid)
            outputs[t] = dec_out
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = dec_out.topk(1)
            inp = (target[t] if teacher_force else topi)
            if teacher_force == False and inp.item() == 1:
                break

        return outputs


teacher_forcing_ratio = 0.5

def clacModel(model, input_tensor, target_tensor, model_optimizer, criterion):
    model_optimizer.zero_grad()
    # input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0
    output = model(input_tensor, target_tensor)
    num_iter = output.size(0)
    print(num_iter)

    for i in range(num_iter):
        loss += criterion(output[i], target_tensor[i])
    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter
    return epoch_loss


def trainModel(model, data, num_iters=1000):
    # data is the tensor pairs already
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # learning rate?
    criterion = torch.nn.NLLLoss()
    total_loss_iterations = 0

    for i in range(1, num_iters + 1):
        training_pair = data[(i - 1)%len(data)]
        in_tensor = training_pair[0]
        out_tensor = training_pair[1]

        loss = clacModel(model, in_tensor, out_tensor, optimizer, criterion)
        total_loss_iterations += loss

        if i % 5000 == 0:
            average_loss = total_loss_iterations / 5000
            total_loss_iterations = 0
            print(i, average_loss, sep='\t')

    torch.save(model.state_dict(), 'magic-chiptuner.pt')
    return model


NUM_ITERS = 1000

@main.command()
def train():
    device = torch.device('cpu')
    data = load('data/txt/', device)
    encoder = Encoder()
    decoder = Decoder()
    model = Seq2Seq(encoder, decoder, device)

    print(encoder)
    print(decoder)

    print(len(data))
    print(data[0])
    model = trainModel(model, data, NUM_ITERS)


if __name__ == '__main__':
    main()
