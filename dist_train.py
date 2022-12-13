import sys
import random
import os
import os.path as osp

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch import optim

from models import SentiFormer
from core import (get_arguments, cfg, cfg_from_file, cfg_from_list, is_enabled, is_main_process, reduce_dict, reduce,
                  build_dataloader, init_process_group, convert_model)  # noqa
from core.checkpointer import Checkpointer
from core.writer import Writer
from core.eval import evaluate
from datasets import Tweets


def build_dataset(cfg, split):
    dataset_name = cfg.DATASET.NAME
    if dataset_name == 'tweets':
        dataset = Tweets(sent_len=cfg.DATASET.SENT_LEN, split=split)

    return dataset


def build_model(cfg):
    if cfg.NET.MODEL == 'sentiformer':
        return SentiFormer(cfg.NET.NUM_CLASSES)


def build_optim(params, cfg):

    if cfg.OPT == 'Adam':
        upd = optim.Adam(params, lr=cfg.LR, betas=(cfg.BETA1, 0.999), weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPT == 'Adamw':
        upd = optim.AdamW(params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPT == 'SGD':
        print("Using SGD >>> learning rate = {:4.3e}, momentum = {:4.3e}, weight decay = {:4.3e}".format(
            cfg.LR, cfg.MOMENTUM, cfg.WEIGHT_DECAY))
        upd = optim.SGD(params, lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    else:
        print("Optimiser {} not supported".format(cfg.OPT))
        raise NotImplementedError

    upd.zero_grad()

    return upd


def main_worker(rank, num_gpus, args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    init_process_group('nccl', num_gpus, rank)

    if args.work_dir is None:
        cfg_name = osp.splitext(osp.basename(args.cfg_file))[0]
        dir_name = osp.split(osp.dirname(args.cfg_file))[1]
        cfg.WORK_DIR = osp.join('work_dirs', dir_name, cfg_name)
    else:
        cfg.WORK_DIR = args.work_dir

    if not osp.exists(cfg.WORK_DIR):
        os.makedirs(cfg.WORK_DIR, 0o775)

    # Reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if is_enabled():
        cfg.NET.BN_TYPE = 'syncbn'

    model = build_model(cfg)
    optim = build_optim(model.parameters(), cfg.NET)

    checkpointer = Checkpointer(cfg.WORK_DIR, max_n=3)
    checkpointer.add_model(model, optim)

    writer = Writer(cfg.WORK_DIR)
    writer._build_writers()

    if args.resume is not None:
        checkpointer.load(args.resume)

    # to ddp model
    model = convert_model(model, find_unused_parameters=True)

    train_dataset = build_dataset(cfg, 'train')
    val_dataset = build_dataset(cfg, 'val')

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=cfg.TRAIN.NUM_WORKERS)

    val_loader = build_dataloader(
        val_dataset,
        batch_size=num_gpus,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=cfg.TRAIN.NUM_WORKERS)

    # show basic config information.
    if is_main_process():
        writer.logger_writer.info(model)
        writer.logger_writer.info("Config: ")
        writer.logger_writer.info(cfg)

    start_epoch = checkpointer.start_epoch
    start_iter = checkpointer.start_iter

    _iter = start_iter
    _epoch = start_epoch

    for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS + 1):
        assert _epoch == epoch
        model.train()

        writer.update_data_timepoint()
        for dataset_dict in train_loader:
            writer.update_forward_timepoint()
            # to cuda
            for k in ['txt', 'txt_gt']:
                if isinstance(dataset_dict[k], torch.Tensor):
                    dataset_dict[k] = dataset_dict[k].cuda()

            # forward
            output = model(dataset_dict)
            losses = {k: v.mean() for k, v in output.items() if k.startswith('loss')}
            loss = sum(losses.values())

            writer.update_backward_timepoint()
            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            writer.update_data_timepoint()
            # log
            loss = reduce(loss)
            losses = reduce_dict(losses)
            if is_main_process():
                log_losses = {k: v.item() for k, v in losses.items() if k.startswith('loss')}
                log_losses['total_loss'] = loss.item()
                writer._iter_log_losses(log_losses, _epoch, _iter)
            _iter += 1

        _epoch += 1

        with torch.no_grad():
            model.eval()
            class_acc, overall_acc = evaluate(model, val_loader)

        writer._epoch_log_eval(train_dataset.CLASS_NAMES, class_acc, overall_acc, _epoch - 1)
        checkpointer.checkpoint(_epoch - 1, _iter - 1, np.mean(class_acc))


def main():
    args = get_arguments(sys.argv[1:])

    num_gpus = args.num_gpus

    if num_gpus > 1:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        mp.spawn(main_worker, nprocs=num_gpus, args=(num_gpus, args))
    else:
        # Simply call main_worker function
        main_worker(0, num_gpus, args)


if __name__ == "__main__":
    main()
