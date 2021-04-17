from argparse import ArgumentParser
from typing import Union
import math

import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from torch.optim import Adam
from torch import nn
from torch.nn import functional as F

from pl_bolts.models.self_supervised.cpc.cpc_module import CPCV2
from pl_bolts.losses.self_supervised_learning import CPCTask
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder

from ecgnet import EmbedCNN
from dataset import ECGCPCDataModule
from logger import MLFlowLogger2


class CPCV2(pl.LightningModule):
    def __init__(
        self,
        datamodule: pl.LightningDataModule = None,
        encoder: Union[str, torch.nn.Module, pl.LightningModule] = "cpc_encoder",
        patch_size: int = 8,
        patch_overlap: int = 4,
        task: str = "cpc",
        num_workers: int = 4,
        learning_rate: int = 1e-4,
        data_dir: str = "",
        batch_size: int = 32,
        **kwargs,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.datamodule = datamodule

        # init encoder
        self.encoder = encoder
        if isinstance(encoder, str):
            self.encoder = self.init_encoder()
        self.encoder = EmbedCNN(pooling=True).feature

        # info nce loss
        c, h = self.__compute_final_nb_c(self.hparams.patch_size)
        self.contrastive_task = CPCTask(num_input_channels=c, target_dim=64, embed_scale=0.1)

        self.z_dim = c * h * h
        #  self.num_classes = self.datamodule.num_classes
        #  self.num_classes = 8

    def init_encoder(self):
        encoder_name = self.hparams.encoder
        return torchvision_ssl_encoder(encoder_name, return_all_feature_maps=self.hparams.task == "amdim")

    def __compute_final_nb_c(self, patch_size):
        dummy_batch = torch.zeros((2 * 49, 1, patch_size, patch_size))
        dummy_batch = self.encoder(dummy_batch)

        # other encoders return a list
        #  if self.hparams.encoder != 'cpc_encoder':
        #  dummy_batch = dummy_batch[0]

        dummy_batch = self.__recover_z_shape(dummy_batch, 2)
        b, c, h, w = dummy_batch.size()
        return c, h

    def __recover_z_shape(self, Z, b):
        # recover shape
        Z = Z.squeeze(-1)
        nb_feats = int(math.sqrt(Z.size(0) // b))
        Z = Z.view(b, -1, Z.size(1))
        Z = Z.permute(0, 2, 1).contiguous()
        Z = Z.view(b, -1, nb_feats, nb_feats)

        return Z

    def forward(self, img_1):
        # put all patches on the batch dim for simultaneous processing
        b, p, c, w, h = img_1.size()
        img_1 = img_1.view(-1, c, w, h)

        # Z are the latent vars
        Z = self.encoder(img_1)

        # non cpc resnets return a list
        # (?) -> (b, -1, nb_feats, nb_feats)
        Z = self.__recover_z_shape(Z, b)

        return Z

    def training_step(self, batch, batch_nb):
        # calculate loss
        nce_loss = self.shared_step(batch)

        # result
        result = pl.TrainResult(nce_loss)
        result.log("train_nce_loss", nce_loss)
        return result

    def validation_step(self, batch, batch_nb):
        # calculate loss
        nce_loss = self.shared_step(batch)

        # result
        result = pl.EvalResult(checkpoint_on=nce_loss)
        result.log("val_nce", nce_loss, prog_bar=True)
        return result

    def shared_step(self, batch):
        img_1 = batch

        # generate features
        # Latent features
        Z = self(img_1)

        # infoNCE loss
        nce_loss = self.contrastive_task(Z)
        return nce_loss

    def configure_optimizers(self):
        opt = Adam(
            params=self.parameters(), lr=self.hparams.learning_rate, betas=(0.8, 0.999), weight_decay=1e-5, eps=1e-7
        )

        return [opt]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser.add_argument("--task", type=str, default="cpc")
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--num_workers", default=24, type=int)
        parser.add_argument("--batch_size", type=int, default=640)
        parser.add_argument("--encoder", type=str, default="resnet18")
        parser.add_argument("--patch_size", type=int, default=16)
        # add
        parser.add_argument("--exp", type=str, default="Pretrain")
        parser.add_argument("--name", type=str, default="test")
        parser.add_argument("--amp_level", type=str, default="O0")
        parser.add_argument("--num_sanity_val_steps", type=int, default=0)
        parser.add_argument("--max_epochs", type=int, default=1000)
        return parser


def main():
    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = CPCV2.add_model_specific_args(parser)
    args = parser.parse_args()

    # datamodule
    dm = ECGCPCDataModule(args)
    args.num_samples = dm.num_samples

    # model
    model = CPCV2(**args.__dict__)

    # trainer
    trainer = pl.Trainer.from_argparse_args(args)
    if args.name != "test":
        logger = MLFlowLogger2(experiment_name=args.exp, run_name=args.name)
    else:
        logger = None
    trainer.logger = logger
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
