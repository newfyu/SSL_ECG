from argparse import ArgumentParser
import os
import shutil

import pytorch_lightning as pl
from torch.utils.checkpoint import checkpoint_sequential
import torch
from pl_bolts.losses.self_supervised_learning import nt_xent_loss
from pl_bolts.models.self_supervised.evaluator import Flatten
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torchvision.models import densenet

from dataset import ECGSSLDataModule
from ecgnet import EmbedCNN
from logger import MLFlowLogger2


class Projection(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(pl.LightningModule):
    def __init__(
        self, batch_size, num_samples, warmup_epochs=10, lr=1e-4, opt_weight_decay=1e-6, loss_temperature=0.5, **kwargs
    ):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()

        self.nt_xent_loss = nt_xent_loss
        self.encoder = self.init_encoder()

        # 使用checkpoint节省显存
        if self.hparams.save_mem:
            self.encoder = torch.nn.Sequential(*list(self.encoder.feature.children()))

        # h -> || -> z
        self.projection = Projection()

    def init_encoder(self):
        encoder = EmbedCNN(pooling=True)
        return encoder

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=["bias", "bn"]):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def setup(self, stage):
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size


    def configure_optimizers(self):
        parameters = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.hparams.opt_weight_decay)

        optimizer = Adam(parameters, lr=self.hparams.lr)
        optimizer = LARSWrapper(optimizer)

        # Trick 2 (after each step)
        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=self.hparams.warmup_epochs,
        max_epochs=self.hparams.max_epochs,
        warmup_start_lr=0,
        eta_min=0,
        )

        scheduler = {
        "scheduler": linear_warmup_cosine_decay,
        "interval": "step",
        "frequency": 1,
        }

        return [optimizer], [scheduler]

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]

        result = self.encoder(x)
        if isinstance(result, list):
            result = result[-1]
        return result

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)

        result = pl.TrainResult(minimize=loss)
        result.log_dict(
            {"train/loss": loss, "train/acc": acc}, on_epoch=True, prog_bar=True,
        )
        return result

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict(
            {"valid/loss": loss, "valid/acc": acc}, on_epoch=True, prog_bar=True,
        )
        return result

    def shared_step(self, batch, batch_idx):
        img1, img2 = batch

        # ENCODE
        # encode -> representations
        # (b, 3, 32, 32) -> (b, 2048, 2, 2)
        if self.hparams.save_mem:
            img1 = torch.autograd.Variable(img1, requires_grad=True)
            img2 = torch.autograd.Variable(img2, requires_grad=True)
            h1 = checkpoint_sequential(self.encoder, 9, img1)
            h2 = checkpoint_sequential(self.encoder, 9, img2)
        else:
            h1 = self.encoder(img1)
            h2 = self.encoder(img2)

        # the bolts resnets return a list of feature maps
        if isinstance(h1, list):
            h1 = h1[-1]
            h2 = h2[-1]

        # PROJECT
        # img -> E -> h -> || -> z
        # (b, 2048, 2, 2) -> (b, 128)
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss, sim = self.nt_xent_loss(z1, z2, self.hparams.loss_temperature)
        acc = self.cale_acc(sim)

        return loss, acc

    @staticmethod
    def cale_acc(sim):
        """利用相似矩阵计算准确度"""
        ones_mat = torch.ones_like(sim)
        mask = torch.eye(*sim.size(), dtype=sim.dtype, layout=sim.layout, device=sim.device).bool()
        ones_mat.masked_fill_(mask, float("-inf"))
        sim = sim * ones_mat
        label1 = torch.arange(0, sim.size(0) / 2) + sim.size(0) / 2
        label2 = torch.arange(0, sim.size(0) / 2)
        label = torch.cat([label1, label2], dim=0).to(sim.device)
        acc = accuracy(sim, label)
        return acc

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        (args, _) = parser.parse_known_args()
        # Training
        parser.add_argument("--exp", type=str, default="Pretrain")
        parser.add_argument("--name", type=str, default="simclr")
        parser.add_argument("--optimizer", choices=["adam", "lars"], default="lars")
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--batch_size", type=int, default=2048)
        parser.add_argument("--lr", type=float, default=1)
        parser.add_argument("--lars_momentum", type=float, default=0.9)
        parser.add_argument("--lars_eta", type=float, default=0.001)
        parser.add_argument("--lr_sched_step", type=float, default=30, help="lr scheduler step")
        parser.add_argument(
            "--lr_sched_gamma", type=float, default=0.5, help="lr scheduler step",
        )
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--benchmark", type=bool, default=True)
        parser.add_argument("--amp_level", type=str, default="O1")
        parser.add_argument("--amp_backend", type=str, default="apex")
        parser.add_argument("--precision", type=int, default=16)
        parser.add_argument("--distributed_backend", default="ddp", type=str)
        parser.add_argument("--num_sanity_val_steps", type=int, default=0)
        # Model
        parser.add_argument("--loss_temperature", type=float, default=0.5)
        parser.add_argument("--num_workers", default=10, type=int)
        parser.add_argument("--save_mem", default=False, type=bool)

        return parser


def main():
    parser = ArgumentParser()
    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    # model args
    parser = SimCLR.add_model_specific_args(parser)
    args = parser.parse_args()
    # datamodule
    dm = ECGSSLDataModule(args)
    args.num_samples = dm.num_samples
    # model
    model = SimCLR(**args.__dict__)
    # trainer
    trainer = pl.Trainer.from_argparse_args(args)
    if args.name != "test":
        logger = MLFlowLogger2(experiment_name=args.exp, run_name=args.name)
        # save source files to mlflow
        save_dir = f"mlruns/{logger.experiment_id}/{logger.run_id}/artifacts"
        print(f"run_id: {logger.run_id}")
        save_files = [i for i in os.listdir() if ".py" in i]
        for i in save_files:
            shutil.copy(i, f"{save_dir}/{i}")
    else:
        logger = None
    trainer.logger = logger
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
