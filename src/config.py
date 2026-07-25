#!/usr/bin/python3
"""Unified configuration for CFSP tasks."""

import argparse
import os


def construct_hyper_param():
    parser = argparse.ArgumentParser()

    # -------- train settings --------
    parser.add_argument('--num_train_epochs', default=10, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument("--batch_size", default=2, type=int,
                        help="Batch size (reduced for 4GB VRAM)")
    parser.add_argument("--accumulate_gradients", default=2, type=int,
                        help="Gradient accumulation steps")
    parser.add_argument('--lr', default=2e-5, type=float, help='Learning rate')

    # -------- model settings --------
    parser.add_argument("--model_dir",
                        default='./models/chinese-roberta-wwm-ext',
                        type=str,
                        help="Path to pretrained model directory")
    parser.add_argument("--with_adv_train", action='store_true',
                        help="Enable FGM adversarial training")

    # -------- data settings --------
    parser.add_argument("--data_dir",
                        default='./data/cfn-dataset',
                        type=str,
                        help="Dataset directory")
    parser.add_argument("--save_dir",
                        default='./saves',
                        type=str,
                        help="Model save directory")

    # -------- hardware settings --------
    parser.add_argument("--use_fp16", action='store_true', default=True,
                        help="Use mixed precision training")
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="Maximum sequence length")

    args = parser.parse_args()
    return args


args = construct_hyper_param()
