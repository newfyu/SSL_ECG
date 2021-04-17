import math
from argparse import ArgumentParser
from typing import Callable, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import AMPType
from torch import nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from ecgnet import EmbedCNN
from dataset import ECGSSLDataModule
from logger import MLFlowLogger2
from pl_bolts.models.self_supervised.evaluator import Flatten


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class SiameseArm(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
        input_dim: int, # 2048
        hidden_size: int, # 4096
        output_dim: int, # 256
    ) -> None:
        super().__init__()

        if encoder is None:
            encoder = EmbedCNN(pooling=True)
        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = MLP(input_dim, hidden_size, output_dim)
        # Predictor
        self.predictor = MLP(output_dim, hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.encoder(x)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h


class SimSiam(pl.LightningModule):
    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        num_nodes: int = 1,
        arch: str = "resnet18",
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        temperature: float = 0.1,
        optimizer: str = "adam",
        lars_wrapper: bool = True,
        exclude_bn_bias: bool = False,
        start_lr: float = 0.0,
        learning_rate: float = 1e-3,
        final_lr: float = 0.0,
        weight_decay: float = 1e-6,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim

        self.optim = optimizer
        self.lars_wrapper = lars_wrapper
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.init_model()

        # compute iters per epoch
        
        nb_gpus = len(self.gpus) if isinstance(gpus, (list, tuple)) else self.gpus
        if isinstance(nb_gpus,str):
            nb_gpus = len(nb_gpus)-1
        #  print('nb_gpus =',nb_gpus)
        assert isinstance(nb_gpus, int)
        global_batch_size = self.num_nodes * nb_gpus * self.batch_size if nb_gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        # define LR schedule
        warmup_lr_schedule = np.linspace(
            self.start_lr, self.learning_rate, self.train_iters_per_epoch * self.warmup_epochs
        )
        iters = np.arange(self.train_iters_per_epoch * (self.max_epochs - self.warmup_epochs))
        cosine_lr_schedule = np.array(
            [
                self.final_lr
                + 0.5
                * (self.learning_rate - self.final_lr)
                * (1 + math.cos(math.pi * t / (self.train_iters_per_epoch * (self.max_epochs - self.warmup_epochs))))
                for t in iters
            ]
        )

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    def init_model(self):
        encoder = EmbedCNN(pooling=True)
        self.online_network = SiameseArm(
            encoder, input_dim=512, hidden_size=self.hidden_mlp, output_dim=self.feat_dim
        )

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def cosine_similarity(self, a, b):
        b = b.detach()  # stop gradient of backbone + projection mlp
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = -1 * (a * b).sum(-1).mean()
        return sim

    def training_step(self, batch, batch_idx):
        img_1, img_2 = batch

        # Image 1 to image 2 loss
        _, z1, h1 = self.online_network(img_1)
        _, z2, h2 = self.online_network(img_2)
        loss = self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2

        # log results
        result = pl.TrainResult(minimize=loss)
        result.log_dict({"train/loss": loss}, on_epoch=True, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):
        img_1, img_2 = batch

        # Image 1 to image 2 loss
        _, z1, h1 = self.online_network(img_1)
        _, z2, h2 = self.online_network(img_2)
        loss = self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2

        # log results
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({"valid/loss": loss}, on_epoch=True, prog_bar=True)

        return result

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

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.lars_wrapper:
            optimizer = LARSWrapper(optimizer, eta=0.001, clip=False)  # trust coefficient

        return optimizer

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_idx: int,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        # warm-up + decay schedule placed here since LARSWrapper is not optimizer class
        # adjust LR of optim contained within LARSWrapper
        if self.lars_wrapper:
            for param_group in optimizer.optim.param_groups:
                param_group["lr"] = self.lr_schedule[self.trainer.global_step]
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr_schedule[self.trainer.global_step]

        # log LR (LearningRateLogger callback doesn't work with LARSWrapper)
        #  self.log("learning_rate", self.lr_schedule[self.trainer.global_step], on_step=True, on_epoch=False)

        # from lightning
        if self.trainer.amp_backend == AMPType.NATIVE:
            optimizer_closure()
            self.trainer.scaler.step(optimizer)
        elif self.trainer.amp_backend == AMPType.APEX:
            optimizer_closure()
            optimizer.step()
        else:
            optimizer.step(closure=optimizer_closure)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # specify flags to store false
        parser.add_argument("--hidden_mlp", default=1024, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--fp32", action="store_true")

        # training params
        parser.add_argument("--num_workers", default=16, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/sgd")
        parser.add_argument("--lars_wrapper", action="store_true", help="apple lars wrapper over optimizer used")
        parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

        parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        # custom
        parser.add_argument("--exp", type=str, default="Pretrain")
        parser.add_argument("--name", type=str, default="simsiam")
        parser.add_argument("--benchmark", type=bool, default=True)
        parser.add_argument("--amp_level", type=str, default="O0")
        parser.add_argument("--num_sanity_val_steps", type=int, default=0)

        return parser


def main():
    seed_everything(32)
    parser = ArgumentParser()
    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    # model args
    parser = SimSiam.add_model_specific_args(parser)
    args = parser.parse_args()
    # pick data
    dm = ECGSSLDataModule(args)
    args.num_samples = dm.num_samples
    # init datamodule
    model = SimSiam(**args.__dict__)

    trainer = pl.Trainer.from_argparse_args(args)
    if args.name != 'test':
        logger = MLFlowLogger2(experiment_name=args.exp, run_name=args.name)
        trainer.logger = logger
    else:
        trainer.logger = None
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
