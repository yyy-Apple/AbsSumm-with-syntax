import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from typing import Dict, List, Tuple
from itertools import chain
from torch.nn.utils import clip_grad_norm_
import fire
from tqdm import tqdm
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Vocab(object):
    def __init__(self, w2i: Dict):
        self.w2i = w2i
        self.i2w = {v: k for k, v in self.w2i.items()}

    def merge(self, vocab: 'Vocab'):
        for word in vocab.w2i:
            if word not in self.w2i:
                index = len(self.w2i)
                self.w2i[word] = index
                self.i2w[index] = word
        return self

    def add_token(self, token):
        index = len(self.w2i)
        self.w2i[token] = index
        self.i2w[index] = token

    @classmethod
    def from_list(cls, word_list: List[str]):
        w2i = {}
        index = 0
        for word in word_list:
            w2i[word] = index
            index += 1
        return cls(w2i)

    @classmethod
    def from_file(cls, filepath, threshold):
        word_counter = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                tokens = line.split(" ")
                for token in tokens:
                    if token not in word_counter:
                        word_counter[token] = 1
                    else:
                        word_counter[token] += 1
        word_list = []
        for k, v in word_counter.items():
            if v > threshold:
                word_list.append(k)
        return cls.from_list(word_list)

    def __len__(self):
        return len(self.w2i)

    def __str__(self):
        return "<Vocabulary> size=%d" % len(self)


LSTM_DIM = 1024
BERT_DIM = 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_oracle(fname):
    """ ignore the final data"""
    sent_idx = 1
    act_idx = 3
    with open(fname) as fh:
        sent_ctr = 0
        tree, sent, acts = "", [], []
        for line in fh:
            sent_ctr += 1
            line = line.strip()
            if line.startswith("# ("):
                sent_ctr = 0
                if tree:
                    yield sent
                tree, sent, acts = line, [], []
            if sent_ctr == sent_idx:
                sent = line.split() + ['<EOS>']
            if sent_ctr >= act_idx:
                if line:
                    acts.append(line)


def get_data(doc_path, target_path):
    """ return a list of form (sentence, tokenized summary)"""
    doc = []
    target = list(read_oracle(target_path))
    with open(doc_path, "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            doc.append(line)
    doc = doc[:-1]
    assert len(doc) == len(target)
    data = []
    for d, t in zip(doc, target):
        data.append((d, t))
    return doc, data


def create_vocab(all_terms):
    vocab = list(set(list(chain(*all_terms))))
    return Vocab.from_list(vocab)


class RNNBaseLine(nn.Module):

    def __init__(self, word_vocab: Vocab):
        super().__init__()
        self.word_vocab = word_vocab

        # use for the output buffer
        self.term_states = (torch.zeros((1, LSTM_DIM), dtype=torch.float32),
                            torch.zeros((1, LSTM_DIM), dtype=torch.float32))

        self.term_lstm = nn.LSTMCell(input_size=LSTM_DIM,
                                     hidden_size=LSTM_DIM)

        self.state2word = nn.Sequential(nn.Linear(in_features=LSTM_DIM + BERT_DIM,
                                                  out_features=LSTM_DIM),
                                        nn.ReLU(),
                                        nn.Linear(in_features=LSTM_DIM,
                                                  out_features=len(self.word_vocab)))

        # for attention
        self.query = nn.Linear(in_features=LSTM_DIM,
                               out_features=BERT_DIM)

        self.word_emb = nn.Embedding(num_embeddings=len(self.word_vocab),
                                     embedding_dim=LSTM_DIM)

        self.criterion = nn.CrossEntropyLoss()

        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.bert = BertModel.from_pretrained('bert-large-uncased')

        self.optimizer = optim.Adam(self.parameters(), lr=2e-5)

    def predict(self, memory: torch.Tensor, n_terms: int, train_sent=None):
        loss = None
        # current states
        h, c = self.term_states

        # query: 1 x BERT_DIM  key: 1 x len x BERT_DIM  value 1 x len x BERT_DIM
        query = self.query(h).unsqueeze(-1)
        attention = torch.matmul(memory, query)
        context = (attention * memory).sum(dim=-2)  # 1 x BERT_DIM

        final_states = torch.cat([h, context], dim=1)

        pred = self.state2word(final_states)
        if train_sent:
            try:
                gold_word = train_sent[n_terms]
            except IndexError:
                raise Exception("All terminals exhausted.")
            gold_word_index = 0
            if gold_word in self.word_vocab.w2i.keys():
                gold_word_index = self.word_vocab.w2i.get(gold_word)
            gold_word_index_tensor = torch.tensor(gold_word_index).unsqueeze(0).to(DEVICE)
            loss = self.criterion(pred, gold_word_index_tensor)
        else:
            # This is in the inference mode, use greedy decode
            gold_word_index_tensor = pred.argmax(dim=-1)
            gold_word_index = gold_word_index_tensor.squeeze().item()
            gold_word = self.word_vocab.i2w.get(gold_word_index)

        word_embedding = self.word_emb(gold_word_index_tensor)
        self.term_states = self.term_lstm(word_embedding, self.term_states)
        return gold_word, loss

    def generate(self, document: str, train_sent=None):
        """Jointly parsing and language modeling
           Here, document is str, train_sent is a list of str.
           In training mode, there is train_sent while in the inference time there is not
        """

        # Use BERT to encode document
        list_of_index = self.tokenizer.encode(document)
        tokens_tensor = torch.tensor(list_of_index).unsqueeze(0).to(DEVICE)
        memory = self.bert(tokens_tensor)[0]

        terms, word_losses = [], []
        while len(terms) < 2 or terms[-1] != '<EOS>':
            if len(terms) > 30:
                break
            gold_word, loss = self.predict(memory, len(terms), train_sent)
            if loss:
                word_losses.append(loss)
            terms.append(gold_word)

        return word_losses, terms

    def train(self, data_list: List[Tuple], epoch=10):
        """ Use the document and gold actions to train the model"""
        # in each tuple, the first is document(str), second is list of str (sentence)
        best_loss = float('inf')
        for i in range(epoch):
            # np.random.shuffle(data_list)
            running_loss = 0.
            for data in tqdm(data_list, desc=f'Training Epoch {i}'):
                # we need to clear states every time we parse a new sentence
                self.term_states = (torch.zeros((1, LSTM_DIM), dtype=torch.float32).to(DEVICE),
                                    torch.zeros((1, LSTM_DIM), dtype=torch.float32).to(DEVICE))

                if 'SEP' in data[0]:
                    # handle some parsing error
                    continue
                word_losses, terms = self.generate(data[0], data[1])
                print(" ".join(terms))
                if len(word_losses) > 0:
                    self.optimizer.zero_grad()
                    final_loss = sum(word_losses) / len(word_losses)
                    running_loss += final_loss.item()
                    final_loss.backward()
                    clip_grad_norm_(self.parameters(), 0.5)
                    self.optimizer.step()

            running_loss = running_loss / len(data_list)

            print(f'Epoch {i} Finished\n'
                  f'Total: {running_loss}')

            if running_loss < best_loss:
                best_loss = running_loss
                torch.save(self.state_dict(), 'rnn.pt')
                print('RNNBaseLine Model Updated.')

    def inference(self, doc_list):
        pred = []
        for doc in doc_list:
            _, terms = self.generate(doc)
            print(" ".join(terms))
            pred.append(" ".join(terms))
        return pred


def main(n_epochs=2):
    # Use for training
    _, train = get_data('train.article', 'train.oracle')
    # Use for inference
    doc, _ = get_data('test.article', 'test.oracle')

    word_vocab_valid_article = Vocab.from_file('data/sumdata/train/valid.article.txt', 1)
    word_vocab_valid_title = Vocab.from_file('data/sumdata/train/valid.title.txt', 1)
    word_vocab = word_vocab_valid_article.merge(word_vocab_valid_title)
    print("Vocabulary size: ", len(word_vocab))
    tp = RNNBaseLine(word_vocab).to(DEVICE)

    print("BEGIN TRAINING...")
    tp.train(train, epoch=n_epochs)
    print("END TRAINING...")

    # Write prediction to file
    pred = tp.inference(doc)
    with open("./predict_rnn.txt", "w", encoding="utf-8") as f:
        for p in pred:
            f.write(p)
            f.write("\n")


if __name__ == '__main__':
    fire.Fire(main)
