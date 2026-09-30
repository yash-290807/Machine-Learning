# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import sys
import traceback
from argparse import ArgumentParser

import submitit
import torch

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from training.utils.train_utils import makedir, register_omegaconf_resolvers

os.environ["HYDRA_FULL_ERROR"] = "1"


def single_proc_run(local_rank, main_port, cfg, world_size):
    """Single GPU process."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def single_node_runner(cfg, main_port: int):
    assert cfg.launcher.num_nodes == 1

    num_proc = cfg.launcher.gpus_per_node

    torch.multiprocessing.set_start_method(
        "spawn",
        force=True,
    )

    if num_proc == 1:
        single_proc_run(
            local_rank=0,
            main_port=main_port,
            cfg=cfg,
            world_size=num_proc,
        )
    else:
        mp_runner = torch.multiprocessing.start_processes

        args = (main_port, cfg, num_proc)

        mp_runner(
            single_proc_run,
            args=args,
            nprocs=num_proc,
            start_method="spawn",
        )


def format_exception(e: Exception, limit=20):
    traceback_str = "".join(
        traceback.format_tb(e.__traceback__, limit=limit)
    )
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


class SubmititRunner(submitit.helpers.Checkpointable):
    """A callable which is passed to submitit to launch the jobs."""

    def __init__(self, port, cfg):
        self.cfg = cfg
        self.port = port
        self.has_setup = False

    def run_trainer(self):
        job_env = submitit.JobEnvironment()

        add_pythonpath_to_sys_path()

        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

        register_omegaconf_resolvers()

        cfg_resolved = OmegaConf.to_container(
            self.cfg,
            resolve=False,
        )
        cfg_resolved = OmegaConf.create(cfg_resolved)

        trainer = instantiate(
            cfg_resolved.trainer,
            _recursive_=False,
        )

        trainer.run()

    def __call__(self):
        job_env = submitit.JobEnvironment()

        self.setup_job_info(
            job_env.job_id,
            job_env.global_rank,
        )

        try:
            self.run_trainer()

        except Exception as e:
            message = format_exception(e)
            logging.error(message)
            raise e

    def setup_job_info(self, job_id, rank):
        """Set up slurm job info."""

        self.job_info = {
            "job_id": job_id,
            "rank": rank,
            "cluster": self.cfg.get("cluster", None),
            "experiment_log_dir": self.cfg.launcher.experiment_log_dir,
        }

        self.has_setup = True


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return

    # Windows uses ';' instead of ':' for PYTHONPATH.
    pythonpath = os.environ["PYTHONPATH"]

    if os.name == "nt":
        paths = pythonpath.split(";")
    else:
        paths = pythonpath.split(":")

    sys.path = paths + sys.path


def main(args) -> None:

    # Load the selected Hydra configuration.
    cfg = compose(
        config_name=args.config,
    )

    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(),
            "sam2_logs",
            args.config,
        )

    print(
        "###################### Train App Config ####################"
    )

    print(
        OmegaConf.to_yaml(cfg)
    )

    print(
        "############################################################"
    )

    add_pythonpath_to_sys_path()

    makedir(
        cfg.launcher.experiment_log_dir
    )

    with g_pathmgr.open(
        os.path.join(
            cfg.launcher.experiment_log_dir,
            "config.yaml",
        ),
        "w",
    ) as f:
        f.write(
            OmegaConf.to_yaml(cfg)
        )

    cfg_resolved = OmegaConf.to_container(
        cfg,
        resolve=False,
    )

    cfg_resolved = OmegaConf.create(
        cfg_resolved
    )

    with g_pathmgr.open(
        os.path.join(
            cfg.launcher.experiment_log_dir,
            "config_resolved.yaml",
        ),
        "w",
    ) as f:
        f.write(
            OmegaConf.to_yaml(
                cfg_resolved,
                resolve=True,
            )
        )

    submitit_conf = cfg.get(
        "submitit",
        None,
    )

    assert submitit_conf is not None, "Missing submitit config"

    submitit_dir = os.path.join(
        cfg.launcher.experiment_log_dir,
        "submitit_logs",
    )

    # Prioritize command-line arguments.
    cfg.launcher.gpus_per_node = (
        args.num_gpus
        if args.num_gpus is not None
        else cfg.launcher.gpus_per_node
    )

    cfg.launcher.num_nodes = (
        args.num_nodes
        if args.num_nodes is not None
        else cfg.launcher.num_nodes
    )

    submitit_conf.use_cluster = (
        args.use_cluster
        if args.use_cluster is not None
        else submitit_conf.use_cluster
    )

    if submitit_conf.use_cluster:

        executor = submitit.AutoExecutor(
            folder=submitit_dir
        )

        submitit_conf.partition = (
            args.partition
            if args.partition is not None
            else submitit_conf.get(
                "partition",
                None,
            )
        )

        submitit_conf.account = (
            args.account
            if args.account is not None
            else submitit_conf.get(
                "account",
                None,
            )
        )

        submitit_conf.qos = (
            args.qos
            if args.qos is not None
            else submitit_conf.get(
                "qos",
                None,
            )
        )

        job_kwargs = {
            "timeout_min": 60 * submitit_conf.timeout_hour,
            "name": (
                submitit_conf.name
                if hasattr(submitit_conf, "name")
                else args.config
            ),
            "slurm_partition": submitit_conf.partition,
            "gpus_per_node": cfg.launcher.gpus_per_node,
            "tasks_per_node": cfg.launcher.gpus_per_node,
            "cpus_per_task": submitit_conf.cpus_per_task,
            "nodes": cfg.launcher.num_nodes,
            "slurm_additional_parameters": {
                "exclude": " ".join(
                    submitit_conf.get(
                        "exclude_nodes",
                        [],
                    )
                ),
            },
        }

        if "include_nodes" in submitit_conf:

            assert (
                len(submitit_conf["include_nodes"])
                >= cfg.launcher.num_nodes
            ), "Not enough nodes"

            job_kwargs["slurm_additional_parameters"][
                "nodelist"
            ] = " ".join(
                submitit_conf["include_nodes"]
            )

        if submitit_conf.account is not None:
            job_kwargs["slurm_additional_parameters"][
                "account"
            ] = submitit_conf.account

        if submitit_conf.qos is not None:
            job_kwargs["slurm_additional_parameters"][
                "qos"
            ] = submitit_conf.qos

        if submitit_conf.get(
            "mem_gb",
            None,
        ) is not None:

            job_kwargs["mem_gb"] = submitit_conf.mem_gb

        elif submitit_conf.get(
            "mem",
            None,
        ) is not None:

            job_kwargs["slurm_mem"] = submitit_conf.mem

        if submitit_conf.get(
            "constraints",
            None,
        ) is not None:

            job_kwargs["slurm_constraint"] = (
                submitit_conf.constraints
            )

        if submitit_conf.get(
            "comment",
            None,
        ) is not None:

            job_kwargs["slurm_comment"] = (
                submitit_conf.comment
            )

        if submitit_conf.get(
            "srun_args",
            None,
        ) is not None:

            job_kwargs["slurm_srun_args"] = []

            if submitit_conf.srun_args.get(
                "cpu_bind",
                None,
            ) is not None:

                job_kwargs["slurm_srun_args"].extend(
                    [
                        "--cpu-bind",
                        submitit_conf.srun_args.cpu_bind,
                    ]
                )

        print(
            "###################### SLURM Config ####################"
        )

        print(job_kwargs)

        print(
            "##########################################"
        )

        executor.update_parameters(
            **job_kwargs
        )

        main_port = random.randint(
            submitit_conf.port_range[0],
            submitit_conf.port_range[1],
        )

        runner = SubmititRunner(
            main_port,
            cfg,
        )

        job = executor.submit(
            runner
        )

        print(
            f"Submitit Job ID: {job.job_id}"
        )

        runner.setup_job_info(
            job.job_id,
            rank=0,
        )

    else:

        cfg.launcher.num_nodes = 1

        main_port = random.randint(
            submitit_conf.port_range[0],
            submitit_conf.port_range[1],
        )

        single_node_runner(
            cfg,
            main_port,
        )


if __name__ == "__main__":

    # IMPORTANT:
    # The SAM2 repository stores its Hydra configs under:
    #
    # sam2/configs/
    #
    # We explicitly point Hydra to that directory instead of using
    # initialize_config_module("sam2.configs").
    config_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "sam2",
            "configs",
        )
    )

    initialize_config_dir(
        config_dir=config_dir,
        version_base="1.2",
    )

    parser = ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help=(
            "path to config file "
            "(e.g. sam2.1_training/"
            "sam2.1_hiera_b+_MOSE_finetune.yaml)"
        ),
    )

    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help=(
            "whether to launch on a cluster, "
            "0: run locally, 1: run on a cluster"
        ),
    )

    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="SLURM partition",
    )

    parser.add_argument(
        "--account",
        type=str,
        default=None,
        help="SLURM account",
    )

    parser.add_argument(
        "--qos",
        type=str,
        default=None,
        help="SLURM qos",
    )

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="number of GPUS per node",
    )

    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="Number of nodes",
    )

    args = parser.parse_args()

    args.use_cluster = (
        bool(args.use_cluster)
        if args.use_cluster is not None
        else None
    )

    register_omegaconf_resolvers()

    main(args)