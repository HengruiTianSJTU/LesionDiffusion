import argparse, os, sys, glob, re
import numpy as np
import time
import torch
import wandb
import h5py
import pytorch_lightning as pl

from omegaconf import OmegaConf
from functools import partial
from queue import Queue
from torch.utils.data import _utils

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import instantiate_from_config, get_obj_from_str
from inference.utils import image_logger, TwoStreamBatchSampler, combine_mask_and_im, combine_mask_and_im_v2, visualize
from main import get_parser, nondefault_trainer_args, worker_init_fn, WrappedDataset, DataLoader


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, test=None, predict=None, batch_sampler=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.batch_sampler = batch_sampler
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _test_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
            
        has_batch_sampler = False
        if self.batch_sampler is not None:
            sampler = get_obj_from_str(self.batch_sampler["target"])
            if sampler == TwoStreamBatchSampler:
                has_batch_sampler = True
                try:
                    primary_batch_size = self.batch_sampler["params"].get("primary_batch_size", 1)
                except Exception: primary_batch_size = 1
                self.batch_sampler = sampler(primary_indices=self.datasets["test"].fine_labeled_indices, 
                                             secondary_indices=self.datasets["test"].coarse_labeled_indices,
                                             batch_size=self.batch_size,
                                             secondary_batch_size=self.batch_size-primary_batch_size, **self.batch_sampler["params"])

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and not has_batch_sampler
        test_dataloader = partial(DataLoader, self.datasets["test"],
                                  num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle, drop_last=False, 
                                  collate_fn=getattr(self.datasets["test"], "collate", _utils.collate.default_collate))
        if not has_batch_sampler: return test_dataloader(batch_size=self.batch_size)
        else: return test_dataloader(batch_sampler=self.batch_sampler, sampler=None)

    def _predict_dataloader(self, shuffle=False):
        return self._test_dataloader(shuffle)


class ImageLogger(Callback):
    def __init__(self, test_batch_frequency=1, max_images=-1, clamp=False,
                 disabled=False, log_on_batch_idx=True, log_first_step=False, log_images_kwargs=None, logger={},
                 # metrics logging
                 log_metrics=False, log_separate=False):
        super().__init__()
        self.batch_freq = test_batch_frequency
        self.max_images = max_images
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_separate = log_separate
        
        def _get_logger(target, params):
            if target == "mask_rescale":
                return lambda x: visualize(x.long(), **params)
            if target == "image_and_mask":
                return lambda x: combine_mask_and_im_v2(x, **params)
            if target == "image_rescale" or True:
                return lambda x: visualize((x - x.min()) / (x.max() - x.min()), **params)
        
        self.keep_queue = Queue(self.max_images)
        
        self.logger = {}
        self.log_metrics = log_metrics
        for name, val in logger.items():
            self.logger[name] = _get_logger(val["target"], val.get("params", {}))
            
    @staticmethod
    def _maybe_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path
    
    def _enqueue_and_dequeue(self, entry, split="test"):
        if self.keep_queue.full():
            to_remove = self.keep_queue.get_nowait()
            os.remove(to_remove)
        self.keep_queue.put_nowait(entry)

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx, metrics=None):
        root = os.path.join(save_dir, "images", split)
        self._maybe_mkdir(root)
        filename = "gs-{:06}_e-{:06}_b-{:06}.png".format(
                    global_step,
                    current_epoch,
                    batch_idx)
        path = os.path.join(root, filename)
        if metrics is not None: images = images | {"metrics": str(metrics)}
        if self.log_separate:
            path = lambda x: os.path.join(self._maybe_mkdir(os.path.join(root, x)), filename)
            local_images = image_logger(images, path, n_grid_images=16, log_separate=True, **self.logger)
            for k in local_images.keys(): self._enqueue_and_dequeue(path(k))
        else:
            local_images = image_logger(images, path, n_grid_images=16, log_separate=False, **self.logger)
            self._enqueue_and_dequeue(path)
        return local_images

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx, split) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images)):
            logger = type(pl_module.logger)

            is_train = False
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                metrics, images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            local_images = self.log_local(getattr(pl_module, "base", pl_module.logger.save_dir), 
                                          split, images,
                                          pl_module.global_step, pl_module.current_epoch, batch_idx, metrics)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, local_images, batch_idx, split)

    def check_frequency(self, check_idx, split="test"):
        log_steps = self.log_steps
        batch_freq = self.batch_freq
        if ((check_idx % batch_freq) == 0 or (check_idx in log_steps)) and (
                check_idx > 0 or self.log_first_step):
            # try:
            #     log_steps.pop(0)
            # except IndexError as e:
            #     print(e)
            #     pass
            return True
        return False

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="test")
            
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="test")


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = cfg_name
        else:
            name = ""
        nowname = name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["strategy"] = "ddp"
        trainer_config["accelerator"] = "gpu"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["strategy"], trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            if isinstance(gpuinfo, int):
                gpuinfo = str(gpuinfo)           
            trainer_config["devices"] = len(re.sub(r"[^0-9]+", "", gpuinfo))
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "save_dir": logdir,
                    "name": nowname,
                    "version": "tensorboard",
                }
            }
        }
        default_logger_cfg = default_logger_cfgs["tensorboard"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "image_logger": {
                "target": "test.ImageLogger",
                "params": {
                    "test_batch_frequency": 750,
                    "max_images": 40,
                }
            },
        }

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            if isinstance(lightning_config.trainer.gpus,int): ngpu = 1
            else: ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # test
        trainer.test(model, data)
        
    except Exception:
        # if opt.debug and trainer.global_rank == 0:
        #     import pdb as debugger
        #     debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        # if opt.debug and not opt.resume and trainer.global_rank == 0:
        #     dst, name = os.path.split(logdir)
        #     dst = os.path.join(dst, "debug_runs", name)
        #     os.makedirs(os.path.split(dst)[0], exist_ok=True)
        #     os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())