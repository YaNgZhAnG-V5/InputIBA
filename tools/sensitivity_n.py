import torch
from torch.utils.data import DataLoader, Subset
import os.path as osp
from argparse import ArgumentParser
from copy import deepcopy
import mmcv
from iba.models import build_classifiers
from iba.datasets import build_dataset
from iba.evaluation import SensitivityN
import cv2
from tqdm import tqdm
import numpy as np
from iba.utils import get_logger


def parse_args():
    parser = ArgumentParser('Sensitivity-N evaluation')
    parser.add_argument('config', help='config file of the attribution method')
    parser.add_argument('heatmap_dir', help='directory of the heatmaps')
    parser.add_argument('work_dir', help='directory to save the result file')
    parser.add_argument('file_name', help='file name fo saving the results')
    parser.add_argument('--log-n-max',
                        type=float,
                        default=5,
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
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id')
    args = parser.parse_args()
    return args


def sensitivity_n(cfg,
                  heatmap_dir,
                  work_dir,
                  file_name,
                  log_n_max=5.0,
                  log_n_ticks=0.1,
                  num_masks=100,
                  num_samples=0,
                  device='cuda:0'):
    assert log_n_max > 1.0, f"log_n_max must be larger than 1.0, but got {log_n_max}"
    logger = get_logger('iba')
    mmcv.mkdir_or_exist(work_dir)
    val_set = build_dataset(cfg.data['val'])
    if num_samples > 0:
        inds = np.arange(num_samples)
        val_set = Subset(val_set, inds)
    val_loader_cfg = deepcopy(cfg.data['data_loader'])
    val_loader_cfg.update({'shuffle': False})
    val_loader = DataLoader(val_set, **val_loader_cfg)
    classifier = build_classifiers(cfg.attributor['classifier']).to(device)

    sample = val_set[0]['img']
    h, w = sample.shape[1:]
    results = {}

    try:
        log_n_list = np.arange(0, log_n_max + log_n_ticks, log_n_ticks)
        n_list = (10 ** log_n_list).astype(int)
        # to eliminate the duplicate elements caused by rounding
        n_list = np.unique(n_list)
        logger.info(f"n_list: [{', '.join(map(str,n_list))}]")
        pbar = tqdm(total=len(n_list) * len(val_loader))
        for n in n_list:
            evaluator = SensitivityN(classifier,
                                     img_size=(h, w),
                                     n=n,
                                     num_masks=num_masks)

            score_diffs_all = []
            sum_attrs_all = []

            for batch in val_loader:
                imgs = batch['img']
                targets = batch['target']
                img_names = batch['img_name']

                for img, target, img_name in zip(imgs, targets, img_names):
                    img = img.to(device)
                    target = target.item()
                    heatmap = cv2.imread(osp.join(heatmap_dir, img_name + '.png'),
                                         cv2.IMREAD_UNCHANGED)
                    heatmap = torch.from_numpy(heatmap).to(img) / 255.0

                    res_single = evaluator.evaluate(heatmap, img, target)
                    score_diffs = res_single['score_diffs']
                    sum_attrs = res_single['sum_attributions']
                    score_diffs_all.append(score_diffs)
                    sum_attrs_all.append(sum_attrs)
                pbar.update(1)
            score_diffs_all = np.concatenate(score_diffs_all, 0)
            sum_attrs_all = np.concatenate(sum_attrs_all, 0)
            corr_matrix = np.corrcoef(score_diffs_all, sum_attrs_all)
            results.update({int(n): corr_matrix[1, 0]})
    except KeyboardInterrupt:
        mmcv.dump(results, file=osp.join(work_dir, file_name))
    mmcv.dump(results, file=osp.join(work_dir, file_name))


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    sensitivity_n(cfg=cfg,
                  heatmap_dir=args.heatmap_dir,
                  work_dir=args.work_dir,
                  file_name=args.file_name,
                  log_n_max=args.log_n_max,
                  log_n_ticks=args.log_n_ticks,
                  num_masks=args.num_masks,
                  num_samples=args.num_samples,
                  device=f'cuda:{args.gpu_id}')


if __name__ == '__main__':
    main()