from torch.utils.data import DataLoader
import torch
import os.path as osp
from argparse import ArgumentParser
from copy import deepcopy
import mmcv
from mmcv.runner.utils import set_random_seed
from iba.models.model_zoo import build_classifiers
from iba.datasets import build_dataset
from iba.evaluation import InsertionDeletion
from iba.utils import get_valid_set
import cv2
from tqdm import tqdm

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


from torch.nn.utils.rnn import pad_sequence
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


def text_pipeline(x):
    return [vocab[token] for token in tokenizer(x)]


def label_pipeline(label):
    return 0 if label == 'neg' else 1


def collate_batch(device):
    def collate_batch_fn(batch):
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

    return collate_batch_fn


def parse_args():
    parser = ArgumentParser('Insertion Deletion evaluation')
    parser.add_argument('config',
                        help='config file of the attribution method')
    parser.add_argument('heatmap_dir',
                        help='directory of the heatmaps')
    parser.add_argument('work_dir',
                        help='directory to save the result file')
    parser.add_argument('file_name',
                        help='file name with extension of the results to be saved')
    parser.add_argument('--scores-file',
                        help='File that records the predicted probability of corresponding target class')
    parser.add_argument('--scores-threshold',
                        type=float,
                        default=0.6,
                        help='Threshold for filtering the samples with low predicted target probabilities')
    parser.add_argument('--num-samples',
                        type=int,
                        default=0,
                        help='Number of samples to check, 0 means checking all the (filtered) samples')
    parser.add_argument('--pixel-batch-size',
                        type=int,
                        default=10,
                        help='Batch size for inserting or deleting the heatmap pixels')
    parser.add_argument('--sigma',
                        type=float,
                        default=5.0,
                        help='Sigma of the gaussian blur filter')
    parser.add_argument('--gpu-id',
                        type=int,
                        default=0,
                        help='GPU id')
    parser.add_argument('--seed',
                        type=int,
                        default=2021,
                        help='Random seed')
    args = parser.parse_args()
    return args


def insertion_deletion(cfg,
                       heatmap_dir,
                       work_dir,
                       file_name,
                       scores_file=None,
                       scores_threshold=0.6,
                       num_samples=0,
                       pixel_batch_size=10,
                       sigma=5.0,
                       device='cuda:0'):
    mmcv.mkdir_or_exist(work_dir)

    classifier = build_classifiers().to(device)
    insertion_deletion_eval = InsertionDeletion(classifier)

    val_iter = IMDB(split='test', cls='pos')

    results = {}
    count = 0
    try:
        for batch in tqdm(val_iter, total=2000):
            target, text, img_name = batch
            img_name = img_name.split('/')[-1].split('.')[0]
            target = label_pipeline(target)
            count += 1
            text = torch.tensor(text_pipeline(text)).to(device)
            heatmap = cv2.imread(osp.join(heatmap_dir, img_name + '.png'),
                                 cv2.IMREAD_UNCHANGED)
            heatmap = torch.from_numpy(heatmap).to(text) / 255.0

            res_single = insertion_deletion_eval.evaluate(heatmap,
                                                          text,
                                                          target)
            ins_auc = res_single['ins_auc']
            del_auc = res_single['del_auc']

            results.update(
                {img_name: dict(ins_auc=ins_auc, del_auc=del_auc)})
            if count>=2000:
                break
    except KeyboardInterrupt:
        mmcv.dump(results, file=osp.join(work_dir, file_name))
        return

    mmcv.dump(results, file=osp.join(work_dir, file_name))


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    set_random_seed(args.seed)
    insertion_deletion(cfg=cfg,
                       heatmap_dir=args.heatmap_dir,
                       work_dir=args.work_dir,
                       file_name=args.file_name,
                       scores_file=args.scores_file,
                       scores_threshold=args.scores_threshold,
                       num_samples=args.num_samples,
                       pixel_batch_size=args.pixel_batch_size,
                       sigma=args.sigma,
                       device=f'cuda:{args.gpu_id}')


if __name__ == '__main__':
    main()
