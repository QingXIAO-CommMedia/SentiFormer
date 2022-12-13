import argparse

import spacy
import numpy as np
import torch

from models import SentiFormer
from core import (cfg, cfg_from_file, convert_model)  # noqa
from datasets import Tweets, tokens_to_vecs


def build_dataset(cfg, split):
    dataset_name = cfg.DATASET.NAME
    if dataset_name == 'tweets':
        dataset = Tweets(sent_len=cfg.DATASET.SENT_LEN, split=split)

    return dataset


def build_model(cfg):
    if cfg.NET.MODEL == 'sentiformer':
        return SentiFormer(cfg.NET.NUM_CLASSES)


def main_worker(args):
    # Reading the config
    cfg_from_file(args.cfg_file)

    model = build_model(cfg)

    if args.resume is not None:
        state_dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state_dict['model'])

    # to ddp model
    model = convert_model(model, find_unused_parameters=True)

    embedding_func = spacy.load('en_vectors_web_lg')
    input_text = args.input_text
    embedding = tokens_to_vecs(input_text.split(), 15, embedding_func)

    batched_input = {}
    batched_input['txt'] = torch.as_tensor(np.ascontiguousarray(embedding, np.float32))[None].cuda()
    with torch.no_grad():
        model.eval()
        cls_logits = model(batched_input)['cls_logits']
        senti_type = torch.argmax(cls_logits, dim=1)
        senti_type = senti_type[0].item()

    print(f"{'Input Text':<12}: {input_text}")
    print(f"{'Sentiment':<12}: {Tweets.REVERSE_CLASS_TABLE[senti_type]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument("--input_text", type=str, help="Input text.")
    parser.add_argument("--resume", type=str, default=None, help="Snapshot \"ID,iter\" to load")

    #
    # Configuration
    #
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True, help='Config file for training (and optionally testing)')

    return parser.parse_args()


def main():
    args = parse_args()
    # Simply call main_worker function
    main_worker(args)


if __name__ == "__main__":
    main()
