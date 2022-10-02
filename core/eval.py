import torch
from tqdm import tqdm

from .distributed import all_reduce


def evaluate(model, loader):
    num_classes = loader.dataset.NUM_CLASSES
    total_TP = torch.zeros((num_classes, ))
    total_gt = torch.zeros((num_classes, ))

    for n, dataset_dict in enumerate(tqdm(loader)):

        dataset_dict['txt'] = dataset_dict['txt'].cuda()
        dataset_dict['txt_gt'] = dataset_dict['txt_gt'].cuda()

        outputs = model(dataset_dict)

        pred = outputs['cls_logits'][0]
        gt = dataset_dict['txt_gt'][0][0]

        pred = torch.argmax(pred)

        if pred == gt:
            total_TP[gt] += 1

        total_gt[gt] += 1

    total_TP = total_TP.cuda()
    total_gt = total_gt.cuda()

    all_reduce(total_TP)
    all_reduce(total_gt)

    acc = torch.nan_to_num(total_TP / (total_gt + 1e-10), 0.0)
    overall_acc = total_TP.sum() / (total_gt.sum() + 1e-10)
    return acc.cpu().numpy(), overall_acc.cpu().numpy()
