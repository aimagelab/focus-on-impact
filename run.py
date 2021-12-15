#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import errno
import os
import random
import socket
import sys
import time

sys.path.append('.')
if os.path.exists('/m100/home/userexternal/flandi00/FD_baselines/environment/habitat-lab'):
    sys.path.insert(0, '/m100/home/userexternal/flandi00/FD_baselines/environment/habitat-lab')

import numpy as np
import torch

import intrinsic_baselines
from habitat_baselines.common.baseline_registry import baseline_registry
from occant_baselines.config.default import get_config

from torch import distributed

torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def init_distributed(world_size, rank):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    addr = 'localhost'
    port = 4242 + int(str(time.time())[-4:])
    if 'SLURM_STEP_GPUS' in os.environ.keys():
        port = port + int(os.environ['SLURM_STEP_GPUS'].split(',')[0])
    port_notok = True
    while port_notok:
        try:
            s.bind(("localhost", port))
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                print("Port is already in use")
                port = port + 1
            else:
                # something else raised the socket.error exception
                print(e)
        port_notok = False
    s.close()

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(port)

    distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)


def run_exp(exp_config: str, run_type: str, world_size, rank, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)

    config.defrost()
    config.WORLD_SIZE = world_size
    config.RANK = rank
    config.DISTRIBUTED = True
    config.freeze()

    init_distributed(world_size, rank)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    torch.backends.cudnn.benchmark = True

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    main()
