# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from fcos_core.utils.model_serialization import load_state_dict
from fcos_core.utils.c2_model_loading import load_c2_format
from fcos_core.utils.imports import import_file
from fcos_core.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)
    def load_pretrain(self, f=None):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint
        
    def load(self, f=None):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded and "model_backbone" not in loaded:
            loaded = dict(model=loaded)
        return loaded

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        for key in self.model.keys():
           data['model_' + key] = self.model[key].state_dict()
        for key in self.optimizer.keys():
           data['optimizer_' + key] = self.optimizer[key].state_dict()
        for key in self.scheduler.keys():
           data['scheduler_' + key] = self.scheduler[key].state_dict()
        
        # data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)
    
    def load_pretrain(self, f=None):
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)

        load_state_dict(self.model["backbone"], checkpoint.pop("model_backbone"))
        load_state_dict(self.model["genbox"], checkpoint.pop("model_genbox"))
        if self.cfg.MODEL.INS.USE_DIS_GLOBAL:
            if "model_ddis" in checkpoint:
                self.logger.info("Global alignment discriminator checkpoint found. Initializing model from the checkpoint")
                load_state_dict(self.model["ddis"], checkpoint.pop("model_ddis"))
            else:
                self.logger.info(
                    "No global discriminator found in the checkpoint. Initializing model from scratch"
                )
        return checkpoint
        

    def load(self, f=None, load_dis=True, load_opt_sch=False):
        if not f and not self.has_checkpoint():
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        elif not f and self.has_checkpoint():
           f = self.get_checkpoint_file()
            
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint, load_dis)

        if load_opt_sch:
            if self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))

                self.optimizer["backbone"].load_state_dict(checkpoint.pop("optimizer_backbone"))
                
                if "optimizer_genbox" in checkpoint:
                     self.optimizer["genbox"].load_state_dict(checkpoint.pop("optimizer_genbox"))
                if "optimizer_genfeature" in checkpoint:
                 self.optimizer["genfeature"].load_state_dict(checkpoint.pop("optimizer_genfeature"))
                self.optimizer["fcos"].load_state_dict(checkpoint.pop("optimizer_fcos"))

                if self.cfg.MODEL.ADV.USE_DIS_GLOBAL:
                    self.optimizer["fdis"].load_state_dict(checkpoint.pop("optimizer_fdis"))
                    

                if self.cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
                    self.optimizer["dis_ca"].load_state_dict(checkpoint.pop("optimizer_dis_ca"))
                
                if self.cfg.MODEL.INS.USE_DIS_GLOBAL:
                    self.optimizer["ddis"].load_state_dict(checkpoint.pop("optimizer_ddis"))
                    
                if self.cfg.MODEL.CM.USE_CM_GLOBAL:
                    self.optimizer["dis_CM."].load_state_dict(checkpoint.pop("optimizer_dis_CM."))
            else:
                self.logger.info(
                    "No optimizer found in the checkpoint. Initializing model from scratch"
                )

            if  self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))

                self.scheduler["backbone"].load_state_dict(checkpoint.pop("scheduler_backbone"))
                if "scheduler_genbox" in checkpoint:
                     self.scheduler["genbox"].load_state_dict(checkpoint.pop("scheduler_genbox"))
                if "scheduler_genfeature" in checkpoint:
                 self.scheduler["genfeature"].load_state_dict(checkpoint.pop("scheduler_genfeature"))
                self.scheduler["fcos"].load_state_dict(checkpoint.pop("scheduler_fcos"))

                if self.cfg.MODEL.ADV.USE_DIS_GLOBAL:
                    self.scheduler["fdis"].load_state_dict(checkpoint.pop("scheduler_fdis"))

                if self.cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
                    self.scheduler["dis_ca"].load_state_dict(checkpoint.pop("scheduler_dis_ca"))
                
                if self.cfg.MODEL.INS.USE_DIS_GLOBAL:
                    self.scheduler["ddis"].load_state_dict(checkpoint.pop("scheduler_ddis"))
                
                if self.cfg.MODEL.CM.USE_CM_GLOBAL:
                   self.scheduler["dis_ca"].load_state_dict(checkpoint.pop("scheduler_dis_ca"))
            else:
                self.logger.info(
                    "No scheduler found in the checkpoint. Initializing model from scratch"
                )

        # return any further checkpoint data
        return checkpoint

    def _load_model(self, checkpoint, load_dis=True):
        if "model_backbone" in checkpoint:
            print('------load our model', checkpoint.keys())
            # load checkpoint of our model
            load_state_dict(self.model["backbone"], checkpoint.pop("model_backbone"))
            if "model_genbox" in  checkpoint:
                load_state_dict(self.model["genbox"], checkpoint.pop("model_genbox"))
            if "model_genfeature" in  checkpoint:
                load_state_dict(self.model["genfeature"], checkpoint.pop("model_genfeature"))
            load_state_dict(self.model["fcos"], checkpoint.pop("model_fcos"))
            if self.cfg.MODEL.ADV.USE_DIS_GLOBAL and load_dis:
                if "model_fdis" in checkpoint:
                    self.logger.info("Global alignment discriminator checkpoint found. Initializing model from the checkpoint")
                    load_state_dict(self.model["fdis"], checkpoint.pop("model_fdis"))
                else:
                    self.logger.info(
                        "No global discriminator found in the checkpoint. Initializing model from scratch"
                    )

            if self.cfg.MODEL.ADV.USE_DIS_CENTER_AWARE and load_dis:
                if "model_dis_ca" in checkpoint:
                    self.logger.info("Center-aware alignment discriminator checkpoint found. Initializing model from the checkpoint")
                    load_state_dict(self.model["dis_ca"], checkpoint.pop("model_dis_ca"))
                else:
                    self.logger.info(
                        "No center-aware discriminator found in the checkpoint. Initializing model from scratch"
                    )
            
            if self.cfg.MODEL.INS.USE_DIS_GLOBAL and load_dis:
                if "model_ddis" in checkpoint:
                    self.logger.info("detector alignment discriminator checkpoint found. Initializing model from the checkpoint")
                    load_state_dict(self.model["ddis"], checkpoint.pop("model_ddis"))
                else:
                    self.logger.info(
                        "No global discriminator found in the checkpoint. Initializing model from scratch"
                    )
            
            if self.cfg.MODEL.CM.USE_CM_GLOBAL and load_dis:
                if "model_dis_CM." in checkpoint:
                    self.logger.info("detector alignment discriminator checkpoint found. Initializing model from the checkpoint")
                    load_state_dict(self.model["dis_CM."], checkpoint.pop("model_dis_CM."))
                else:
                    self.logger.info(
                        "No global discriminator found in the checkpoint. Initializing model from scratch"
                    )
        else:
            # load others, e.g., Imagenet pretrained pkl
            load_state_dict(self.model["backbone"], checkpoint.pop("model"))
