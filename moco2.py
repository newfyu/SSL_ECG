from argparse import ArgumentParser
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from pl_bolts.metrics import mean, precision_at_k
from pytorch_lightning import Trainer, seed_everything
from torch import nn
from dataset import ECGSSLDataModule
from ecgnet import ResNet, BasicBlock
from logger import MLFlowLogger2
import os
import shutil


class EmbedCNNfc(nn.Module):
    def __init__(self, out_dim=128, layers=[2, 2, 2, 2], dim=64):
        super().__init__()

        self.feature = ResNet(BasicBlock, layers, dim=dim)  # 34:[3,4,6,3]
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        B = x.size(0)
        out = self.feature(x)
        out = out.reshape(B, -1)
        out = self.fc(out)
        return out


class MocoV2(pl.LightningModule):
    def __init__(
        self,
        base_encoder: Union[str, torch.nn.Module] = "resnet18",
        emb_dim: int = 128,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        lr: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
        batch_size: int = 256,
        use_mlp: bool = False,
        num_workers: int = 8,
        *args,
        **kwargs
    ):
        """
        Args:
            base_encoder: torchvision model name or torch.nn.Module
            emb_dim: feature dimension (default: 128)
            num_negatives: queue size; number of negative keys (default: 65536)
            encoder_momentum: moco momentum of updating key encoder (default: 0.999)
            softmax_temperature: softmax temperature (default: 0.07)
            learning_rate: the learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            datamodule: the DataModule (train, val, test dataloaders)
            batch_size: batch size
            use_mlp: add an mlp to the encoders
            num_workers: workers for the loaders
        """

        super().__init__()
        self.save_hyperparameters()
        self.datamodule = datamodule

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k = self.init_encoders(base_encoder)

        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(),
                self.encoder_q.fc,
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(),
                self.encoder_k.fc,
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_encoders(self, base_encoder):
        """
        Override to add your own encoders
        """
        encoder_q = EmbedCNNfc()
        encoder_k = EmbedCNNfc()

        return encoder_q, encoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer


        self.queue_ptr[0] = ptr

    def forward(self, img_q, img_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def training_step(self, batch, batch_idx):
        img_1, img_2 = batch

        output, target = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output.float(), target.long())
        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        return {"loss": loss, "log": log, "progress_bar": log}

    def validation_step(self, batch, batch_idx):
        img_1, img_2 = batch
        output, target = self(img_q=img_1, img_k=img_2)
        val_loss = F.cross_entropy(output, target.long())

        val_acc1, val_acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
        }
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, "val_loss")
        val_acc1 = mean(outputs, "val_acc1")
        val_acc5 = mean(outputs, "val_acc5")

        log = {
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
        }
        return {"val_loss": val_loss, "log": log, "progress_bar": log}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), self.hparams.lr,
        momentum=self.hparams.momentum,
        weight_decay=self.hparams.weight_decay)
        #  optimizer = torch.optim.Adam(self.parameters(), self.hparams.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser.add_argument("--exp", type=str, default="Pretrain")
        parser.add_argument("--name", type=str, default="test")
        parser.add_argument("--base_encoder", type=str, default="resnet18")
        parser.add_argument("--emb_dim", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=20)
        parser.add_argument("--num_negatives", type=int, default=65536)
        parser.add_argument("--encoder_momentum", type=float, default=0.999)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--lr", type=float, default=0.03)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--batch_size", type=int, default=2048)
        parser.add_argument("--max_epochs", type=int, default=1000)
        parser.add_argument("--use_mlp", action="store_true")
        parser.add_argument("--benchmark", type=bool, default=True)
        parser.add_argument("--amp_level", type=str, default="O0")

        return parser


def main():
    parser = ArgumentParser()
    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    # model args
    parser = MocoV2.add_model_specific_args(parser)
    args = parser.parse_args()
    # datamodule
    dm = ECGSSLDataModule(args)
    args.num_samples = dm.num_samples
    # model
    model = MocoV2(**args.__dict__)
    # trainer
    trainer = pl.Trainer.from_argparse_args(args)
    if args.name != "test":
        logger = MLFlowLogger2(experiment_name=args.exp, run_name=args.name)
        # save source files to mlflow
        save_dir = f"mlruns/{logger.experiment_id}/{logger.run_id}/artifacts"
        print(f'run_id: {logger.run_id}')
        save_files = [i for i in os.listdir() if ".py" in i]
        for i in save_files:
            shutil.copy(i, f"{save_dir}/{i}")
    else:
        logger = None
    trainer.logger = logger
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
