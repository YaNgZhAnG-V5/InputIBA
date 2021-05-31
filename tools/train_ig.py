import torch
from torch.utils.data import DataLoader
import sys
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import mmcv
import argparse
from PIL import Image

sys.path.insert(0, "..")
import iba
from iba.models import Attributer
from iba.datasets.imdb import IMDB

cfg = mmcv.Config.fromfile(os.path.join(os.getcwd(), '../configs/deep_lstm.py'))
dev = torch.device('cuda:0')

# load the data
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

from torch.nn.utils.rnn import pad_sequence
from captum.attr import LayerIntegratedGradients

vec = torchtext.vocab.GloVe(name='6B', dim=100)
tokenizer = get_tokenizer('basic_english')
vocab_iter = torchtext.datasets.IMDB(split='train')

counter = Counter()
for (label, line) in vocab_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter, max_size=25000)
vocab.load_vectors(vec)


# normalize saliency to be in [0.2, 1]
def normalize_saliency(saliency):
    saliency -= saliency.min()
    saliency /= saliency.max()
    return saliency


# get masks
def generate_lime_masks():
    work_dir = os.path.join(os.getcwd(), '../NLP_masks_lime')
    tokenizer = get_tokenizer('basic_english')
    train_iter = IMDB(split='test', cls='pos')
    model = iba.models.model_zoo.build_classifiers()
    for count, (label, text, filename) in tqdm(enumerate(train_iter)):
        if count >= 2000:
            break
        filename = filename.split('/')[-1].split('.')[0]
        mask_file = os.path.join(work_dir, filename)
        if not os.path.isfile(mask_file + '.png'):
            attributor = LayerIntegratedGradients(model_fn_given_length(text_length[0].unsqueeze(0)), model.embedding)
            saliency = attributor.attribute()

            # normalize saliency
            saliency = normalize_saliency(saliency)

            # save mask as png image
            mask = (saliency * 255).astype(np.uint8)
            mask = np.resize(np.expand_dims(mask, 0), (50, mask.shape[0]))
            dir_name = osp.abspath(osp.dirname(mask_file))
            mmcv.mkdir_or_exist(dir_name)
            mask = Image.fromarray(mask, mode='L')
            mask.save(mask_file + '.png')


if __name__ == '__main__':
    generate_lime_masks()
