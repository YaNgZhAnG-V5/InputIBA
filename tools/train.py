import torchvision
from torch.utils.data import DataLoader
import os
import os.path as osp
import warnings
from argparse import ArgumentParser
from copy import deepcopy
import mmcv
from iba.models import Net, IBA
from tqdm import tqdm
from iba.datasets import build_dataset


def parse_args():
    parser = ArgumentParser('train a model')
    parser.add_argument('config', help='configuration file')
    parser.add_argument('--work-dir', help='working directory', default=os.getcwd())
    parser.add_argument('--gpu-id', help='gpu id', type=int, default=0)
    args = parser.parse_args()
    return args


def train(config, work_dir, gpu_id=0):
    cfg = mmcv.Config.fromfile(config)
    mmcv.mkdir_or_exist(work_dir)
    if len(os.listdir(work_dir)) > 0:
        warnings.warn('The working directory is not empty!')
    cfg.dump(osp.join(work_dir, 'config.py'))

    logger = mmcv.get_logger('iba', log_file=osp.join(work_dir, 'log_file.log'))
    device = f'cuda:{gpu_id}'
    train_net(cfg, logger, work_dir=work_dir, device=device)


def train_net(cfg: mmcv.Config, logger, work_dir, device='cuda:0'):
    train_set = build_dataset(cfg.data['train'])
    val_set = build_dataset(cfg.data['val'])
    train_loader = DataLoader(train_set, **cfg.data['data_loader'])
    val_loader_cfg = deepcopy(cfg.data['data_loader'])
    val_loader_cfg.update({'shuffle': False})
    val_loader = DataLoader(val_set, **val_loader_cfg)

    # currently only support VGG
    model = torchvision.models.vgg16(pretrained=True).to(device)
    model.eval()
    iba = IBA(model.features[17])
    iba.sigma = None

    iba.reset_estimate()
    iba.estimate(model, train_loader, device=device, **cfg.train_cfg)

    for imgs, targets, img_names in tqdm(val_loader, total=len(val_loader)):
        for img, target, img_name in zip(imgs, targets, img_names):
            target = target.item()
            rel_dir = val_set.get_ind_to_cls()[target]
            feat_mask_file = osp.join(work_dir, 'feat_masks', rel_dir, img_name + '.png')
            img_mask_file = osp.join(work_dir, 'img_masks', rel_dir, img_name + '.png')

            net = Net(image=img,
                      target=target,
                      model=model,
                      IBA=iba,
                      dev=device, **cfg.model['net'])
            net.train(logger)
            net.show_feat_mask(out_file=feat_mask_file, **cfg.test_cfg.pop('feat_mask', {}))
            net.show_img_mask(out_file=img_mask_file, **cfg.test_cfg.pop('img_mask', {}))


def main():
    args = parse_args()
    train(config=args.config,
          work_dir=args.work_dir,
          gpu_id=args.gpu_id)


if __name__ == '__main__':
    main()