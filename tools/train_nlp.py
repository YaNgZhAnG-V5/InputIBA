import torch
from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm
import numpy as np
import mmcv
import argparse

sys.path.insert(0, "..")
import iba
from iba.models import Attributer

# define IMDB dataset to return ID
import torchtext
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
import io

URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

MD5 = '7c2ac02c03563afcf9b574c7e56c153a'

NUM_LINES = {
    'train': 25000,
    'test': 25000,
}

_PATH = 'aclImdb_v1.tar.gz'

DATASET_NAME = "IMDB"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_wrap_split_argument(('train', 'test'))
def IMDB(root, split, cls=None):
    def generate_imdb_data(key, extracted_files):
        for fname in extracted_files:
            if 'urls' in fname:
                continue
            elif key in fname:
                if cls is None and ('pos' in fname or 'neg' in fname):
                    with io.open(fname, encoding="utf8") as f:
                        label = 'pos' if 'pos' in fname else 'neg'
                        yield label, f.read(), fname
                elif cls == 'pos' and ('pos' in fname):
                    with io.open(fname, encoding="utf8") as f:
                        label = 'pos'
                        yield label, f.read(), fname
                elif cls == 'neg' and ('neg' in fname):
                    with io.open(fname, encoding="utf8") as f:
                        label = 'neg'
                        yield label, f.read(), fname

    dataset_tar = download_from_url(URL, root=root,
                                    hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)
    iterator = generate_imdb_data(split, extracted_files)
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split], iterator)


cfg = mmcv.Config.fromfile(os.path.join(os.getcwd(), '../configs/deep_lstm.py'))
dev = torch.device('cuda:0')

# load the data
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

vec = torchtext.vocab.GloVe(name='6B', dim=100)
tokenizer = get_tokenizer('basic_english')
train_iter = torchtext.datasets.IMDB(split='train')

counter = Counter()
for (label, line) in train_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter, max_size=25000)
vocab.load_vectors(vec)

train_iter = IMDB(split='train', cls='pos')

from torch.nn.utils.rnn import pad_sequence

# train_iter = torchtext.datasets.IMDB(split='train')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 0 if x == 'neg' else 1


def collate_batch(batch):
    label_list, text_list, text_length_list, fname_list = [], [], [], []
    for (_label, _text, _fname) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        text_length_list.append(torch.tensor([processed_text.shape[0]]))
        fname_list.append(int(_fname.split('/')[-1].split('.')[0].replace('_', '')))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    padded_text_list = pad_sequence(text_list)
    text_length_list = torch.cat(text_length_list)
    fname_list = torch.tensor(fname_list, dtype=torch.int64)
    return label_list.to(device), padded_text_list.to(device), text_length_list.to(device), fname_list.to(device)


dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

# estimate
attibuter = Attributer(cfg.attributer, device=dev)
attibuter.estimate(dataloader, cfg.estimation_cfg)

# get masks
cfg.attribution_cfg.img_iba.beta = 0.25 
work_dir = os.path.join(os.getcwd(),'../4_layer/NLP_masks_0_25_rnn4')
train_iter = IMDB(split='test', cls='pos')
for count, (label, text, filename) in tqdm(enumerate(train_iter)):
    if count >=100:
        break
    filename = filename.split('/')[-1].split('.')[0]
    feat_mask_file = os.path.join(work_dir, 'feat_masks', filename)
    img_mask_file = os.path.join(work_dir, 'img_masks', filename)
    if not (os.path.isfile(feat_mask_file + '.png') and os.path.isfile(img_mask_file + '.png')):
        img = torch.tensor(text_pipeline(text))
        target = torch.tensor([label_pipeline(label)]).double().expand(10, -1)
        attibuter.set_text(text)
        attibuter.make_attribution(img.to(dev),
                                   target.to(dev),
                                   attribution_cfg=cfg.attribution_cfg)
        attibuter.show_feat_mask(**cfg.attribution_cfg.get('feat_mask', {}), tokenizer=tokenizer,
                                 out_file=feat_mask_file)
        attibuter.show_img_mask(**cfg.attribution_cfg.get('img_mask', {}), tokenizer=tokenizer, out_file=img_mask_file)
