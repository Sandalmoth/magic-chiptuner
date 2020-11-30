import torch
import numpy as np
import click
import os
import random


MAX_LENGTH = 65536


# For now, I will implement seq2seq as in
# https://www.guru99.com/seq2seq-model.html
# and we'll see if/how well that works
# hence, for now, consider the Encoder, Decoder, Seq2Seq classes to
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


INPUT_DIM = 256
HIDDEN_DIM = 256
EMBED_DIM = 256
NUM_LAYERS = 4


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.input_dim = INPUT_DIM
        self.hidden_dim = HIDDEN_DIM
        self.embed_dim = EMBED_DIM
        self.num_layers = NUM_LAYERS

        self.emebedding = torch.nn.Embedding(self.input_dim, self.embed_dim)
        self.gru = torch.nn.GRU(
            self.embedded_dim, self.hidden_dim, num_layers=self.num_layers
        )

    def forward(self, src):
        embedded = self.embedding(src).view(1, 1, -1)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.input_dim = INPUT_DIM
        self.hidden_dim = HIDDEN_DIM
        self.embed_dim = EMBED_DIM
        self.num_layers = NUM_LAYERS

        self.emebedding = torch.nn.Embedding(self.output_dim, self.embed_dim)
        self.gru = torch.nn.GRU(
            self.embed_idm, self.hidden_dim, num_layers=self.num_layers
        )
        self.out = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = torch.nn.LogSoftMax(dim=1)

    def forward(self, inputs, hidden):
        inputs = inputs.view(1, -1)
        embedded = torch.nn.functional.relu(self.embedding(inputs))
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
        input_length = source.size(0)
        batch_size = target.shape[1]
        target_length = target.shape[0]
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(
            target_length, batch_size, vocab_size
        ).to(self.device)
        for i in range(input_length):
            enc_out, enc_hid = self.encoder(source[i])

        dec_hid = enc_hid.to(self.device)
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


@main.command()
def train():
    device = torch.device('cpu')
    data = load('data/txt/', device)
    encoder = Encoder()
    decoder = Decoder()
    s2s = Seq2Seq(encoder, decoder, device)


if __name__ == '__main__':
    main()
