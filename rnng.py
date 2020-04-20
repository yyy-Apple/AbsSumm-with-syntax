import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from typing import Dict, List, Tuple
from itertools import chain
from torch.nn.utils import clip_grad_norm_
from tqdm import trange, tqdm
import fire


class Vocab(object):
    def __init__(self, w2i: Dict):
        self.w2i = w2i
        self.i2w = {v: k for k, v in self.w2i.items()}

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
                    yield tree, sent, acts
                tree, sent, acts = line, [], []
            if sent_ctr == sent_idx:
                sent = line.split()
            if sent_ctr >= act_idx:
                if line:
                    acts.append(line)


def get_data(doc_path, target_path):
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
        data.append((d, t[0], t[1], t[2]))
    return doc, data


def create_vocab(all_terms):
    vocab = list(set(list(chain(*all_terms))))
    return Vocab.from_list(vocab)


def get_NTs(actions):
    # Get all kinds of Non-terminals
    # because NT action is something like this: NT(S), NT(NP)
    NTs = []
    for act in actions:
        if act.startswith("NT"):
            NTs.append(act[3:-1])
    return NTs


class TransitionParser(nn.Module):

    def __init__(self, word_vocab: Vocab, act_vocab: Vocab, nt_vocab: Vocab):
        super().__init__()
        self.word_vocab = word_vocab
        self.act_vocab = act_vocab
        self.nt_vocab = nt_vocab
        self.act2nt = {v: self.nt_vocab.w2i[k[3:-1]] for k, v in self.act_vocab.w2i.items() if k.startswith("NT")}

        # use for the output buffer
        self.term_states = (torch.zeros((1, LSTM_DIM), dtype=torch.float32),
                            torch.zeros((1, LSTM_DIM), dtype=torch.float32))

        self.term_lstm = nn.LSTMCell(input_size=LSTM_DIM,
                                     hidden_size=LSTM_DIM)

        # use for the action
        self.act_states = (torch.zeros((1, LSTM_DIM), dtype=torch.float32),
                           torch.zeros((1, LSTM_DIM), dtype=torch.float32))

        self.act_lstm = nn.LSTMCell(input_size=LSTM_DIM,
                                    hidden_size=LSTM_DIM)

        # use for the partial tree
        self.stack = [((torch.zeros((1, LSTM_DIM), dtype=torch.float32),
                        torch.zeros((1, LSTM_DIM), dtype=torch.float32)),
                       "<ROOT>")]  # something like ((h_0, c_0), action_name)

        self.comp_lstm_fwd = nn.LSTMCell(input_size=LSTM_DIM,
                                         hidden_size=LSTM_DIM)
        self.comp_lstm_rev = nn.LSTMCell(input_size=LSTM_DIM,
                                         hidden_size=LSTM_DIM)

        self.state2act = nn.Sequential(nn.Linear(in_features=3 * LSTM_DIM + BERT_DIM,
                                                 out_features=LSTM_DIM),
                                       nn.ReLU(),
                                       nn.Linear(in_features=LSTM_DIM,
                                                 out_features=len(self.act_vocab)))

        self.state2word = nn.Sequential(nn.Linear(in_features=3 * LSTM_DIM + BERT_DIM,
                                                  out_features=LSTM_DIM),
                                        nn.ReLU(),
                                        nn.Linear(in_features=LSTM_DIM,
                                                  out_features=len(self.word_vocab)))
        self.comp_h = nn.Linear(2 * LSTM_DIM, LSTM_DIM)
        self.comp_c = nn.Linear(2 * LSTM_DIM, LSTM_DIM)

        # for attention
        self.query = nn.Linear(in_features=3 * LSTM_DIM,
                               out_features=BERT_DIM)

        self.word_emb = nn.Embedding(num_embeddings=len(self.word_vocab),
                                     embedding_dim=LSTM_DIM)
        self.nt_emb = nn.Embedding(num_embeddings=len(self.nt_vocab),
                                   embedding_dim=LSTM_DIM)
        self.act_emb = nn.Embedding(num_embeddings=len(self.act_vocab),
                                    embedding_dim=LSTM_DIM)
        self.criterion = nn.CrossEntropyLoss()

        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.bert = BertModel.from_pretrained('bert-large-uncased')

        self.optimizer = optim.Adam(self.parameters(), lr=2e-5)

    def get_valid_actions(self, open_nts: List[int], open_nt_ceil=100):
        """Return a list of valid action index"""

        valid_actions = []
        # The NT(X) operation can only be applied if B is not empty and n < 100
        if len(open_nts) < open_nt_ceil:
            valid_actions += [v for k, v in self.act_vocab.w2i.items() if k.startswith("NT")]
        # The SHIFT operation can only be applied if B is not empty and n >= 1
        if len(open_nts) >= 1:
            valid_actions += [self.act_vocab.w2i['SHIFT']]
        # The REDUCE operation can only be applied if the top of the stack is
        # not an open nonterminal symbol
        # the REDUCE operation can only be applied if
        # len(open_nts) >=2 or buffer is empty
        if len(open_nts) >= 1 and open_nts[-1] < (len(self.stack) - 1):
            valid_actions += [self.act_vocab.w2i['REDUCE']]
        return valid_actions

    def predict_action(self, valid_actions: List[int], memory: torch.Tensor):
        """ We need to both condition on the context from encoder, as well
            as the states in the decoder.
            Return a tensor
        """
        # states from the stack
        h, c = self.stack[-1][0]

        # concate output buffer states, action states and stack states together
        states = torch.cat([self.term_states[0], self.act_states[0], h], dim=1)

        # query: 1 x BERT_DIM   key: 1 x len x BERT_DIM   value: 1 x len x BERT_DIM
        query = self.query(states).unsqueeze(-1)
        attention = torch.matmul(memory, query)
        context = (attention * memory).sum(dim=-2)  # 1 x BERT_DIM

        final_states = torch.cat([states, context], dim=1)

        out = self.state2act(final_states)

        invalid_actions = [i for i in list(range(len(self.act_vocab))) if i not in valid_actions]
        for action_idx in invalid_actions:
            out[0][action_idx] = -9999999  # apply mask
        return out, final_states

    def get_action(self, valid_actions: List[int], n_actions: int, memory, train_acts=None):
        """ Return the gold action(int) with loss during training
            Return the predicted action(int) during inference
        """
        gold_action = valid_actions[0]
        pred, final_states = self.predict_action(valid_actions, memory)
        loss = None
        if len(valid_actions) > 1:
            if train_acts:
                # This is in the training mode
                try:
                    gold_action = self.act_vocab.w2i[train_acts[n_actions]]
                except IndexError:
                    raise Exception("All gold actions exhausted.")
                gold_action_tensor = torch.tensor(gold_action).unsqueeze(0).to(DEVICE)
                loss = self.criterion(pred, gold_action_tensor)
            else:
                # This is the inference mode, use greedy decode
                gold_action = pred.argmax(dim=-1).item()
        return gold_action, loss, final_states

    def do_action(self, action: int, open_nts: List[int], n_terms: int, final_states, train_sent=None):
        """ Update the term_states, act_states and stack """
        # here action is the index
        # to perform the action and update all the state information on the stack
        # There are 3 kinds of actions in total
        # NT, SHIFT, REDUCE

        loss = None  # associated with SHIFT
        open_nt_index = None  # associated with NT
        term = None  # associated with REDUCE

        act_tensor = torch.tensor(action).unsqueeze(0).to(DEVICE)
        act_embedding = self.act_emb(act_tensor)
        self.act_states = self.act_lstm(act_embedding, self.act_states)

        if self.act_vocab.i2w[action] == 'SHIFT':
            # if it is SHIFT, we have another additional loss which is the generated word loss
            pred = self.state2word(final_states)
            if train_sent:
                # This is in the training mode
                try:
                    gold_word = train_sent[n_terms]
                except IndexError:
                    raise Exception("All terminals exhausted.")
                gold_word_index = 0
                if gold_word in self.word_vocab.w2i.keys():
                    gold_word_index = self.word_vocab.w2i[gold_word]
                gold_word_tensor = torch.tensor(gold_word_index).unsqueeze(0).to(DEVICE)
                loss = self.criterion(pred, gold_word_tensor)
            else:
                # This is in the inference mode, use greedy decode
                gold_word_tensor = pred.argmax(dim=-1)
                gold_word_index = gold_word_tensor.squeeze().item()
                gold_word = self.word_vocab.i2w[gold_word_index]

            word_embedding = self.word_emb(gold_word_tensor)
            self.term_states = self.term_lstm(word_embedding, self.term_states)

            self.stack.append(((word_embedding, word_embedding), gold_word))

            term = gold_word

        elif self.act_vocab.i2w[action] == 'REDUCE':
            children = []
            last_nt_index = open_nts.pop()
            # we need to pop out the children on the stack
            while len(self.stack) - 1 > last_nt_index:
                children.append(self.stack.pop())  # it is in the reverse order
            parent = self.stack.pop()
            h_f, c_f = self.comp_lstm_fwd(parent[0][0])
            h_b, c_b = self.comp_lstm_rev(parent[0][0])
            for i in range(len(children)):
                h_f, c_f = self.comp_lstm_fwd(children[len(children) - 1 - i][0][0], (h_f, c_f))
                h_b, c_b = self.comp_lstm_rev(children[i][0][0], (h_b, c_b))
            cat_h = torch.cat([h_f, h_b], dim=1)
            cat_c = torch.cat([c_f, c_b], dim=1)
            new_h = self.comp_h(cat_h)
            new_c = self.comp_c(cat_c)
            comp_str = parent[1] + " " + " ".join([child[1] for child in children]) + ")"
            self.stack.append(((new_h, new_c), comp_str))

        else:  # action is NT
            nt_index = self.act2nt[action]
            nt_vector = torch.tensor(nt_index).unsqueeze(0).to(DEVICE)
            nt_embedding = self.nt_emb(nt_vector)

            # here we get a new open NT
            self.stack.append(((nt_embedding, nt_embedding), "(" + self.nt_vocab.i2w[nt_index]))
            open_nt_index = len(self.stack) - 1
        return loss, open_nt_index, term

    def generate(self, document: str, train_sent=None, train_acts=None):
        """Jointly parsing and language modeling
           Here, document is str, train_sent is a list of str.
           In training mode, there is train_sent while in the inference time there is not
        """

        # we need to clear states every time we parse a new sentence
        self.stack = [((torch.zeros((1, LSTM_DIM), dtype=torch.float32).to(DEVICE),
                        torch.zeros((1, LSTM_DIM), dtype=torch.float32).to(DEVICE)),
                       "<ROOT>")]  # something like ((h_0, c_0), action_name)
        self.term_states = (torch.zeros((1, LSTM_DIM), dtype=torch.float32).to(DEVICE),
                            torch.zeros((1, LSTM_DIM), dtype=torch.float32).to(DEVICE))

        self.act_states = (torch.zeros((1, LSTM_DIM), dtype=torch.float32).to(DEVICE),
                           torch.zeros((1, LSTM_DIM), dtype=torch.float32).to(DEVICE))

        # User BERT to encode document
        list_of_index = self.tokenizer.encode(document)
        tokens_tensor = torch.tensor(list_of_index).unsqueeze(0).to(DEVICE)
        memory = self.bert(tokens_tensor)[0]

        terms, actions, open_nts, losses = [], [], [], []
        while len(terms) < 2 or len(self.stack) > 2:  # we stop when we get a valid tree

            if len(terms) > 30:  # during inference
                break

            if train_sent:
                if len(terms) == len(train_sent):
                    break

            valid_actions = self.get_valid_actions(open_nts)

            action, loss, final_states = self.get_action(valid_actions, len(actions), memory, train_acts)
            if loss:
                losses.append(loss)
            actions.append(action)

            word_loss, open_nt_index, term = self.do_action(action, open_nts, len(terms), final_states, train_sent)
            if word_loss:
                losses.append(word_loss)
            if open_nt_index:
                open_nts.append(open_nt_index)
            if term:
                terms.append(term)
        # print(" ".join(terms))
        return losses, terms

    def train(self, data_list: List[Tuple], epoch=10):
        """ Use the document and gold actions to train the model"""
        # in each tuple, the first is document(str), second is tree, third is list of str (sentence), the forth is
        # list of str (action)
        for i in range(epoch):
            np.random.shuffle(data_list)
            running_loss = 0.0
            for data in tqdm(data_list, desc=f'Training Epoch {i}'):
                if 'SEP' in data[0]:
                    # handle some parsing error
                    continue
                losses, _ = self.generate(data[0], data[2], data[3])
                if len(losses) > 0:
                    self.optimizer.zero_grad()
                    final_loss = sum(losses)
                    running_loss += final_loss.item()
                    final_loss.backward()
                    clip_grad_norm_(self.parameters(), 0.5)
                    self.optimizer.step()
            running_loss = running_loss / len(data_list)
            print("On epoch %d, current loss is: %f" % (i, running_loss))

    def inference(self, doc_list):
        pred = []
        for doc in doc_list:
            _, terms = self.generate(doc)
            pred.append(" ".join(terms))
        return pred


def main(n_epochs=2):
    # Use for training
    _, train = get_data('train.article', 'train.oracle')
    # Use for inference
    doc, _ = get_data('dev.article', 'dev.oracle')

    word_vocab = Vocab.from_file('train.article', 5)
    act_vocab = create_vocab([x[3] for x in train])
    nt_vocab = Vocab.from_list(get_NTs(act_vocab.w2i.keys()))
    tp = TransitionParser(word_vocab, act_vocab, nt_vocab).to(DEVICE)
    print("BEGIN TRAINING...")
    tp.train(train, epoch=n_epochs)
    print("END TRAINING...")

    # Write prediction to file
    pred = tp.inference(doc)
    with open("./predict.txt", "w", encoding="utf-8") as f:
        for p in pred:
            f.write(p)
            f.write("\n")


if __name__ == '__main__':
    fire.Fire(main)
