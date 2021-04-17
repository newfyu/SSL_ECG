import os
import shutil
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.metrics.functional import accuracy
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from cpc import CPCV2
from dataset import MultiLabelDataModule
from ecgnet import LeadWishCNN
from logger import MLFlowLogger2
from moco2 import MocoV2
from simclr import SimCLR
from simsiam import SimSiam
from swav import SwAV
from byol import BYOL


class LinearEvaluator(pl.LightningModule):
    def __init__(self, input_dim=64, output_dim=8, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = LeadWishCNN(num_cls=output_dim, dim=input_dim)
        if self.hparams.ptmodel == "moco2":
            self.ptmodel = MocoV2.load_from_checkpoint(self.hparams.ptmodel_path, strict=False)
            self.ptmodel = self.ptmodel.encoder_q
            self.ptmodel = torch.nn.Sequential(*(list(self.ptmodel.children())[:-1]))
            self.model.feature = self.ptmodel

        if self.hparams.ptmodel == "simclr":
            self.ptmodel = SimCLR.load_from_checkpoint(self.hparams.ptmodel_path, strict=False)
            self.model.feature = self.ptmodel.encoder.feature

        if self.hparams.ptmodel == "byol":
            self.ptmodel = BYOL.load_from_checkpoint(self.hparams.ptmodel_path, strict=False)
            self.model.feature = self.ptmodel.online_network.encoder

        if self.hparams.ptmodel == "simsiam":
            self.ptmodel = SimSiam.load_from_checkpoint(self.hparams.ptmodel_path, strict=False)
            self.model.feature = self.ptmodel.online_network.encoder.feature

        if self.hparams.ptmodel == "swav":
            self.ptmodel = SwAV.load_from_checkpoint(self.hparams.ptmodel_path, strict=False)
            self.model.feature = self.ptmodel.model.feature
            self.model.feature = nn.Sequential(self.model.feature, nn.AdaptiveAvgPool2d(1))

        if self.hparams.ptmodel == "imagenet":
            self.ptmodel = torchvision.models.resnet18(pretrained=True)
            self.ptmodel = torch.nn.Sequential(*(list(self.ptmodel.children())[:-1]))
            self.ptmodel[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.feature = self.ptmodel

        if self.hparams.ptmodel == "cpc":
            self.ptmodel = CPCV2.load_from_checkpoint(self.hparams.ptmodel_path)
            self.model.feature = self.ptmodel.encoder

        # frozen conv layer
        if self.hparams.frozen == "true":
            print("Frozen all convolution layers")
            for m in self.model.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.weight.requires_grad = False
        elif self.hparams.frozen == "false":
            print("Train all layers")
        elif self.hparams.frozen == "group":
            print("Hierarchical learning rate")

        if self.hparams.classifier == "linear":
            if not self.hparams.dropout:
                self.model.classifier = nn.Linear(input_dim * 8 * 12, output_dim)
        elif self.hparams.classifier == "mlp":
            hidden_dim = 512
            self.model.classifier == nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(input_dim * 8 * 12, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, output_dim),
            )

        # LeadWishCNN 分组
        self.model_groups = []
        model_list = list(self.model.feature.children())
        self.model_groups.append(nn.Sequential(*model_list[:5]))
        self.model_groups.append(nn.Sequential(*model_list[5:6]))
        self.model_groups.append(nn.Sequential(*model_list[6:7]))
        self.model_groups.append(nn.Sequential(*model_list[7:]))
        self.model_groups.append(self.model.classifier)

    def forward(self, x):
        #  self.model.feature.eval()
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.view(N * C, 1, H, W)
        #  with torch.no_grad():
        x = self.model.feature(x)
        x = x.reshape(N, -1)
        x = self.model.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
        result = pl.TrainResult(minimize=loss)
        result.log_dict({"loss/train_loss": loss})
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
        return y, out, loss

    def validation_epoch_end(self, outputs):
        outputs = list(zip(*outputs))
        ys = torch.cat(outputs[0], dim=0).cpu().numpy()
        outs = torch.cat(outputs[1], dim=0).cpu().numpy()
        loss = sum(outputs[2]) / len(outputs[2])

        auc = roc_auc_score(ys, outs)
        ap = average_precision_score(ys, outs)
        f1 = f1_score(ys, outs > 0.5, average="macro")
        result = pl.EvalResult(checkpoint_on=-torch.FloatTensor([ap]))
        result.log_dict({"loss/valid_loss": loss, "auc": auc, "ap": ap, "f1": f1}, prog_bar=True)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
        return y, out, loss

    def test_epoch_end(self, outputs):
        outputs = list(zip(*outputs))
        ys = torch.cat(outputs[0], dim=0).cpu().numpy()
        outs = torch.cat(outputs[1], dim=0).cpu().numpy()
        loss = sum(outputs[2]) / len(outputs[2])
        result = pl.EvalResult(checkpoint_on=loss)
        auc = roc_auc_score(ys, outs)
        ap = average_precision_score(ys, outs)
        f1 = f1_score(ys, outs > 0.5, average="macro")
        result.log_dict({"test_auc": auc, "test_ap": ap, "test_f1": f1})
        return result

    def configure_optimizers(self):
        # Hierarchical learning rate
        if self.hparams.frozen == "group":
            lr = self.hparams.lr
            optimizer = torch.optim.Adam(
                [
                    {"params": self.model_groups[0].parameters(), "lr": lr / 10},
                    {"params": self.model_groups[1].parameters(), "lr": lr / 10},
                    {"params": self.model_groups[2].parameters(), "lr": lr / 10},
                    {"params": self.model_groups[3].parameters(), "lr": lr / 10},
                    {"params": self.model_groups[4].parameters()},
                ],
                lr=lr,
            )
        else:  # Frozen conv layers
            parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = torch.optim.Adam(parameters, lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60], gamma=0.1, verbose=False)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser.add_argument("--exp", type=str, default="Evaluate6")
        parser.add_argument("--name", type=str, default="test")
        parser.add_argument("--img_dir", type=str, default="data/EVD300K_crops/")
        parser.add_argument("--train_label", type=str, default="label/ECG200K_ctg8_train_nspc50.csv")
        parser.add_argument(
            "--ptmodel",
            choices=["moco2", "simclr", "byol", "cpc", "swav", "imagenet", "simsiam", "none"],
            default="none",
        )
        parser.add_argument("--epoch_cut", type=int, default=1)
        parser.add_argument("--ptmodel_path", type=str, default="")
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--warmup_epochs", type=int, default=5)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--frozen", type=str, default="true")
        parser.add_argument("--num_workers", default=24, type=int)
        parser.add_argument("--max_epochs", default=200, type=int)
        parser.add_argument("--input_dim", default=64, type=int)
        parser.add_argument("--classifier", type=str, default="linear")
        parser.add_argument("--dropout", default=True, type=bool)
        parser.add_argument("--test", type=str, default=None)
        parser.add_argument("--weight_sample", type=bool, default=True)
        parser.add_argument("--seed", type=int, default=320)
        parser.add_argument("--benchmark", type=bool, default=True)
        parser.add_argument("--amp_level", type=str, default="O0")
        parser.add_argument("--num_sanity_val_steps", type=int, default=0)
        return parser


def main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LinearEvaluator.add_model_specific_args(parser)
    args = parser.parse_args()

    dm = MultiLabelDataModule(args)
    model = LinearEvaluator(**args.__dict__)
    pl.seed_everything(args.seed)

    if args.test != None:
        print("test")
        print(len(dm.test_dataloader().dataset))
        model = model.load_from_checkpoint(args.test)
        trainer = pl.Trainer.from_argparse_args(args)
        trainer.logger = None
        trainer.test(model, test_dataloaders=dm.test_dataloader())
    else:
        print(len(dm.train_dataloader().dataset))
        print(len(dm.val_dataloader().dataset))
        trainer = pl.Trainer.from_argparse_args(args)
        if args.name != "test":
            logger = MLFlowLogger2(experiment_name=args.exp, run_name=args.name)
            # save source files to mlflow
            save_dir = f"mlruns/{logger.experiment_id}/{logger.run_id}/artifacts"
            print(f"{args.name}, run_id: {logger.run_id}")
            save_files = [i for i in os.listdir() if ".py" in i]
            for i in save_files:
                shutil.copy(i, f"{save_dir}/{i}")
        else:
            logger = None
        trainer.logger = logger
        trainer.fit(model, dm)
        trainer.test()


if __name__ == "__main__":
    main()
