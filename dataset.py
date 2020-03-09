import os
from typing import Tuple

import youtokentome as yttm
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import tensor

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def read_corupuses(en_file, ru_file):
    if not (os.path.exists(en_file) and os.path.exists(ru_file)):
        raise FileNotFoundError('Couldn\'t find corpus files')
    with open(en_file, 'rb') as file:
        content = file.read().decode()
        en_corpus = list(filter(lambda x: len(x) > 0, content.split('\n')))
    with open(ru_file, 'rb') as file:
        content = file.read().decode('utf8')
        ru_corpus = list(filter(lambda x: len(x) > 0, content.split('\n')))
    return en_corpus, ru_corpus


def init_tokenizers(en_file, ru_file) -> Tuple[yttm.youtokentome.BPE, yttm.youtokentome.BPE]:
    if not (os.path.exists(en_file) and os.path.exists(ru_file)):
        raise FileNotFoundError('Couldn\'t find corpus files')
    yttm.BPE.train(data=en_file, vocab_size=5000, model='en_tokenizer.model')
    yttm.BPE.train(data=ru_file, vocab_size=5000, model='ru_tokenizer.model')
    return yttm.BPE(model='en_tokenizer.model'), yttm.BPE(model='ru_tokenizer.model')


class RuEnDataset(data.Dataset):
    def __init__(self, en_file, ru_file,
                 en_tokenizer_file='en_tokenizer.model',
                 ru_tokenizer_file='ru_tokenizer.model'):
        if not (os.path.exists(en_tokenizer_file) and os.path.exists(ru_tokenizer_file)):
            self.en_tokenizer, self.ru_tokenizer = init_tokenizers(en_file, ru_file)
        else:
            self.en_tokenizer = yttm.BPE(model=en_tokenizer_file)
            self.ru_tokenizer = yttm.BPE(model=ru_tokenizer_file)
        self.en_corpus, self.ru_corpus = read_corupuses(en_file, ru_file)

    def __len__(self):
        return len(self.en_corpus)

    def __getitem__(self, idx):
        ru_sent_enc = self.ru_tokenizer.encode([self.ru_corpus[idx]], output_type=yttm.OutputType.ID, bos=True,
                                               eos=True)
        en_sent_enc = self.en_tokenizer.encode([self.en_corpus[idx]], output_type=yttm.OutputType.ID, bos=True,
                                               eos=True)
        return tensor(ru_sent_enc), tensor(en_sent_enc)

def collate_fn(batch):
    (xx, yy) = zip(*batch)
    len_x = len(xx)
    xx = [torch.squeeze(el, dim=0) for el in xx]
    yy = [torch.squeeze(el, dim=0) for el in yy]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=PAD_ID)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=PAD_ID)
    return xx_pad, yy_pad
