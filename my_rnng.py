# %%
from itertools import chain

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

LSTM_DIM = 256
train_file = "data/train.oracle"
dev_file = "data/dev.oracle"
test_file = "data/test.oracle"
cluster_file = "data/bllip_clusters"


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
    def from_file(cls, file):
        # This is for the word vocab constructed from bllip_clusters
        word_list = []
        with open(file, "r", encoding="utf8") as f:
            for line in f.readlines():
                _, word, _ = line.split("\t")
                word_list.append(word)
        return cls.from_list(word_list)

    def __len__(self):
        return len(self.w2i)

    def __str__(self):
        return "<Vocabulary> size=%d" % len(self)


'''
Some notes here:
word, action, and non-terminal will all have embeddings here
'''


class TransitionParser(object):
    def __init__(self, word_vocab: Vocab, act_vocab: Vocab, nt_vocab: Vocab):
        self.word_vocab = word_vocab
        self.act_vocab = act_vocab
        self.nt_vocab = nt_vocab
        self.act2nt = {v: self.nt_vocab.w2i[k[3:-1]] for k, v in self.act_vocab.w2i.items() if k.startswith("NT")}
        # self.nt_vocab = nt_vocab
        self.stack = [((torch.zeros((1, LSTM_DIM), dtype=torch.float32),
                        torch.zeros((1, LSTM_DIM), dtype=torch.float32)),
                       "<ROOT>")]  # something like ((h_0, c_0), action_name)
        self.stack_lstm = nn.LSTMCell(input_size=LSTM_DIM, hidden_size=LSTM_DIM)
        self.comp_lstm_fwd = nn.LSTMCell(input_size=LSTM_DIM, hidden_size=LSTM_DIM)
        self.comp_lstm_rev = nn.LSTMCell(input_size=LSTM_DIM, hidden_size=LSTM_DIM)
        self.state2act = nn.Sequential(nn.Linear(in_features=LSTM_DIM,
                                                 out_features=LSTM_DIM),
                                       nn.ReLU(),
                                       nn.Linear(in_features=LSTM_DIM,
                                                 out_features=len(self.act_vocab)))

        self.state2word = nn.Sequential(nn.Linear(in_features=LSTM_DIM,
                                                  out_features=LSTM_DIM),
                                        nn.ReLU(),
                                        nn.Linear(in_features=LSTM_DIM,
                                                  out_features=len(self.word_vocab)))
        self.comp_h = nn.Linear(2 * LSTM_DIM, LSTM_DIM)
        self.comp_c = nn.Linear(2 * LSTM_DIM, LSTM_DIM)

        self.word_emb = nn.Embedding(num_embeddings=len(self.word_vocab),
                                     embedding_dim=LSTM_DIM)
        self.NT_emb = nn.Embedding(num_embeddings=len(self.nt_vocab),
                                   embedding_dim=LSTM_DIM)
        self.act_emb = None
        self.criterion = nn.CrossEntropyLoss()

        self.params = list(self.stack_lstm.parameters()) \
                      + list(self.comp_lstm_fwd.parameters()) \
                      + list(self.comp_lstm_rev.parameters()) \
                      + list(self.state2act.parameters()) \
                      + list(self.state2word.parameters()) \
                      + list(self.comp_h.parameters()) \
                      + list(self.comp_c.parameters()) \
                      + list(self.word_emb.parameters()) \
                      + list(self.NT_emb.parameters())

        self.optimizer = optim.Adam(self.params, lr=0.001)

    def get_valid_actions(self, open_nts: List[int], open_nt_ceil=100):
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

    def predict_action(self, valid_actions: List[int]):
        h, c = self.stack[-1][0]
        out = self.state2act(h)
        invalid_actions = [i for i in list(range(len(self.act_vocab))) if i not in valid_actions]
        for action_idx in invalid_actions:
            out[0][action_idx] = -9999999  # apply mask
        return out

    def get_action(self, valid_actions: List[int], n_actions, train_acts=None):
        gold_action = valid_actions[0]
        pred = self.predict_action(valid_actions)
        loss = None
        if len(valid_actions) > 1:
            if train_acts:
                try:
                    gold_action = self.act_vocab.w2i[train_acts[n_actions]]
                except IndexError:
                    raise Exception("All gold actions exhausted, return None")
                gold_action_tensor = torch.tensor(gold_action).unsqueeze(0)
                loss = self.criterion(pred, gold_action_tensor)
            else:
                # TODO: This is the inference part
                probs = F.softmax(pred, dim=1)
                gold_act_tensor = torch.multinomial(probs, 1)
                gold_action = gold_act_tensor.detach().numpy()[0][0]
                print(gold_action)
        return gold_action, loss

    # def do_action(self, stack, action, params, open_nts, n_terms, train_sent=None, dropout=None):
    def do_action(self, action: int, open_nts: List[int], n_terms, train_sent=None):
        # here action is the index
        # to perform the action and update all the state information on the stack
        # There are 3 kinds of actions in total
        # NT, SHIFT, REDUCE
        loss = None
        open_nt_index = None
        term = None
        print(self.act_vocab.i2w[action])
        if self.act_vocab.i2w[action] == 'SHIFT':
            # if it is SHIFT, we have another additional loss which is the generated word loss
            h, c = self.stack[-1][0]
            pred = self.state2word(h)
            if train_sent:
                try:
                    gold_word = train_sent[n_terms]
                except IndexError:
                    raise Exception("All terminals exhausted.")
                gold_word_index = 0
                if gold_word in self.word_vocab.w2i.keys():
                    gold_word_index = self.word_vocab.w2i[gold_word]
                gold_word_tensor = torch.tensor(gold_word_index).unsqueeze(0)
                loss = self.criterion(pred, gold_word_tensor)
            else:
                probs = F.softmax(pred, dim=1)
                gold_word_tensor = torch.multinomial(probs, 1).squeeze(0)
                gold_word_index = gold_word_tensor.detach().numpy()[0]
                gold_word = self.word_vocab.i2w[gold_word_index]
            word_embedding = self.word_emb(gold_word_tensor)
            h_t, c_t = self.stack_lstm(word_embedding, (h, c))
            term = gold_word
            self.stack.append(((h_t, c_t), gold_word))
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
            nt_vector = torch.tensor(nt_index).unsqueeze(0)
            nt_embedding = self.NT_emb(nt_vector)
            h, c = self.stack[-1][0]
            h_t, c_t = self.stack_lstm(nt_embedding, (h, c))
            # here we get a new open NT
            self.stack.append(((h_t, c_t), "(" + self.nt_vocab.i2w[nt_index]))
            open_nt_index = len(self.stack) - 1
        return loss, open_nt_index, term

    def generate(self, train_sent=None, train_acts=None):
        self.stack = [((torch.zeros((1, LSTM_DIM), dtype=torch.float32),
                        torch.zeros((1, LSTM_DIM), dtype=torch.float32)),
                       "<ROOT>")]  # something like ((h_0, c_0), action_name)
        terms, actions, open_nts, losses = [], [], [], []
        while len(terms) == 0 or len(self.stack) > 2:  # 回归出一棵语法树再停

            '''
            Here is the problem. There are many sentences that have multiple NT(S), for example this one:
            
            ['``', 'Right', 'now', ',', 'we', "'re", 'lucky', 'if', 'after', 'five', 
            'years', 'we', 'keep', 'one', 'new', 'ringer', 'out', 'of', '10', ',', "''", 'he', 'adds', '.']
            
            ['NT(S)', 'SHIFT', 'NT(S)', 'NT(ADVP)', 'SHIFT', 'SHIFT', 'REDUCE', 'SHIFT', 'NT(NP)', 
            'SHIFT', 'REDUCE', 'NT(VP)', 'SHIFT', 'NT(ADJP)', 'SHIFT', 'REDUCE', 'NT(SBAR)', 'SHIFT',
             'NT(S)', 'NT(PP)', 'SHIFT', 'NT(NP)', 'SHIFT', 'SHIFT', 'REDUCE', 'REDUCE', 'NT(NP)', 
             'SHIFT', 'REDUCE', 'NT(VP)', 'SHIFT', 'NT(NP)', 'SHIFT', 'SHIFT', 'SHIFT', 'NT(QP)', 
             'SHIFT', 'SHIFT', 'SHIFT', 'REDUCE', 'REDUCE', 'REDUCE', 'REDUCE', 'REDUCE', 'REDUCE', 
             'REDUCE', 'SHIFT', 'SHIFT', 'NT(NP)', 'SHIFT', 'REDUCE', 'NT(VP)', 'SHIFT', 'REDUCE', 
             'SHIFT', 'REDUCE']
             
             So the terminal should not be the length. There should be a STOP action
             
            '''
            if len(terms) > 50:  # during inference
                break
            if train_sent:
                if len(terms) == len(train_sent):
                    break
            valid_actions = self.get_valid_actions(open_nts)
            action, loss = self.get_action(valid_actions, len(actions), train_acts)
            if loss:
                losses.append(loss)
            actions.append(action)
            word_loss, open_nt_index, term = self.do_action(action, open_nts, len(terms), train_sent)
            if word_loss:
                losses.append(word_loss)
            if open_nt_index:
                open_nts.append(open_nt_index)
            if term:
                terms.append(term)
            print(" ".join(terms))
        return losses
        # if train_sent:
        #     final_loss = sum(losses)
        #     self.optimizer.zero_grad()
        #     final_loss.backward()
        #     self.optimizer.step()

    def train(self, data_list: List[Tuple], epoch=2):
        # 马德，什么垃圾data，，毕竟按照gold来，还是会有oov什么的，还是会生成不完整的语法树。。干脆所有action用完就撤
        # in each tuple, the first is tree, second is list of str(which is sentence), the third is action
        for i in range(epoch):
            np.random.shuffle(data_list)
            running_loss = 0.0
            for data in data_list:
                print(data[1])
                print(data[2])
                losses = self.generate(data[1], data[2])
                print(losses)
                if len(losses) > 0:
                    self.optimizer.zero_grad()
                    final_loss = sum(losses)
                    running_loss += final_loss.item()
                    final_loss.backward()
                    self.optimizer.step()
            running_loss = running_loss / len(data)
            print("On epoch %d, current loss is: %f" % (i, running_loss))


def read_oracle(fname, gen=True):
    sent_idx = 1 if gen else 4  # using non-UNKified sentences
    act_idx = 3 if gen else 5
    with open(fname) as fh:
        sent_ctr = 0
        tree, sent, acts = "", [], []
        for line in fh:
            sent_ctr += 1
            line = line.strip()
            if line.startswith("#"):
                sent_ctr = 0
                if tree:
                    yield tree, sent, acts
                tree, sent, acts = line, [], []
            if sent_ctr == sent_idx:
                sent = line.split()
            if sent_ctr >= act_idx:
                if line:
                    acts.append(line)


def load_data(tr=train_file, d=dev_file, ts=test_file):
    train, dev, test = [], [], []
    if tr:
        train = list(read_oracle(tr))
    if d:
        dev = list(read_oracle(d))
    if ts:
        test = list(read_oracle(ts))
    return train, dev, test


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


# %%
# toy dataset
sent = "Due to an editing error , a letter to the editor in yesterday 's edition from Frederick H. Hallett mistakenly identified the NRDC .".split(
    " ")
word_vocab = Vocab.from_list(list(set(sent)))
train_act = ["NT(S)", "NT(PP)", "SHIFT", "NT(PP)", "SHIFT", "NT(NP)", "SHIFT", "SHIFT", "SHIFT", "REDUCE", "REDUCE",
             "REDUCE",
             "SHIFT", "NT(NP)", "NT(NP)", "SHIFT", "SHIFT", "REDUCE", "NT(PP)", "SHIFT", "NT(NP)", "SHIFT", "SHIFT",
             "REDUCE",
             "REDUCE", "NT(PP)", "SHIFT", "NT(NP)", "NT(NP)", "SHIFT", "SHIFT", "REDUCE", "SHIFT", "REDUCE", "REDUCE",
             "NT(PP)",
             "SHIFT", "NT(NP)", "SHIFT", "SHIFT", "SHIFT", "REDUCE", "REDUCE", "REDUCE", "NT(ADVP)", "SHIFT", "REDUCE",
             "NT(VP)",
             "SHIFT", "NT(NP)", "SHIFT", "SHIFT", "REDUCE", "REDUCE", "SHIFT", "REDUCE"]
act = list(set(train_act))
act_vocab = Vocab.from_list(act)
nt_vocab = Vocab.from_list(["VP", "ADVP", "NP", "S", "PP"])
tp = TransitionParser(word_vocab, act_vocab, nt_vocab)
# %%
tp.generate(sent, train_act)  # train

# %%
tp.generate()

# Here is the action
# tp.do_action(4, [], 0, sent)  NT(S)
# tp.do_action(5, [1], 0, sent)  NT(PP)
# tp.do_action(6, [1, 2], 0, sent)  SHIFT
# tp.do_action(3, [1, 2], 1, sent) NT(PP)
# tp.do_action(6, [1, 2, 4], 1, sent)  SHIFT
# %%

train, dev, test = load_data(train_file, dev_file, test_file)
# for the time being...
train += test
word_vocab = Vocab.from_file(cluster_file)
act_vocab = create_vocab([x[2] for x in train])
nt_vocab = Vocab.from_list(get_NTs(act_vocab.w2i.keys()))
tp = TransitionParser(word_vocab, act_vocab, nt_vocab)
tp.train(train)
