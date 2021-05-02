import torch
from torch.utils.data import DataLoader
import os.path as osp
from argparse import ArgumentParser
from copy import deepcopy
import mmcv
from mmcv.runner.utils import set_random_seed
from iba.models import build_classifiers
from iba.datasets import build_dataset
from iba.evaluation import SensitivityN
import cv2
from tqdm import tqdm
import numpy as np
from iba.utils import get_valid_set
import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

from iba.datasets.imdb import IMDB


def parse_args():
    parser = ArgumentParser('Sensitivity-N evaluation')
    parser.add_argument('config', help='config file of the attribution method')
    parser.add_argument('heatmap_dir', help='directory of the heatmaps')
    parser.add_argument('work_dir', help='directory to save the result file')
    parser.add_argument('file_name', help='file name fo saving the results')
    parser.add_argument('--scores-file',
                        help='File that records the predicted probability of corresponding target class')
    parser.add_argument('--scores-threshold',
                        type=float,
                        default=0.6,
                        help='Threshold for filtering the samples with low predicted target probabilities')
    parser.add_argument('--log-n-max',
                        type=float,
                        default=4.5,
                        help='maximal N of Sensitivity-N')
    parser.add_argument('--log-n-ticks',
                        type=float,
                        default=0.1,
                        help='Ticks for determining the Ns')
    parser.add_argument('--num-masks',
                        type=int,
                        default=100,
                        help='Number of random masks of Sensitivity-N')
    parser.add_argument('--num-samples',
                        type=int,
                        default=0,
                        help='Number of samples to evaluate, 0 means checking all the samples')
    parser.add_argument('--gpu-id',
                        type=int,
                        default=0,
                        help='GPU id')
    parser.add_argument('--seed',
                        type=int,
                        default=2021,
                        help='random seed')
    args = parser.parse_args()
    return args


def sensitivity_n(cfg,
                  heatmap_dir,
                  work_dir,
                  file_name,
                  scores_file=None,
                  scores_threshold=0.6,
                  num_masks=100,
                  num_samples=0,
                  device='cuda:0'):
    logger = mmcv.get_logger('iba')
    mmcv.mkdir_or_exist(work_dir)

    val_iter = IMDB(split='test', cls='pos')
    classifier = build_classifiers().to(device)

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

    results = {}

    try:
        n_list = np.linspace(0, 0.5, 1, dtype=int)
        # to eliminate the duplicate elements caused by rounding
        n_list = np.unique(n_list)
        logger.info(f"n_list: [{', '.join(map(str,n_list))}]")
        pbar = tqdm(total=len(n_list) * 2000)
        for n in n_list:
            count = 0

            corr_all = []
            for batch in val_iter:
                count += 1
                target, text, img_name = batch
                text = torch.tensor(text_pipeline(text)).to(device)
                target = label_pipeline(target)
                heatmap = cv2.imread(osp.join(heatmap_dir, img_name + '.png'),
                                     cv2.IMREAD_UNCHANGED)
                heatmap = torch.from_numpy(heatmap).to(text) / 255.0

                evaluator = SensitivityN(classifier,
                                         text.shape[0],
                                         n=int(text.shape[0] * n),
                                         num_masks=num_masks)
                res_single = evaluator.evaluate(heatmap, text, target, calculate_corr=True)
                corr = res_single['correlation'][1, 0]
                corr_all.append(corr)
                pbar.update(1)
                if count > 2000:
                    break
            results.update({int(n): np.mean(corr_all)})
    except KeyboardInterrupt:
        mmcv.dump(results, file=osp.join(work_dir, file_name))
    mmcv.dump(results, file=osp.join(work_dir, file_name))


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    set_random_seed(args.seed)
    sensitivity_n(cfg=cfg,
                  heatmap_dir=args.heatmap_dir,
                  work_dir=args.work_dir,
                  file_name=args.file_name,
                  scores_file=args.scores_file,
                  scores_threshold=args.scores_threshold,
                  num_masks=args.num_masks,
                  num_samples=args.num_samples,
                  device=f'cuda:{args.gpu_id}')


if __name__ == '__main__':
    main()
