from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from typing import Any
from ecgnet import ResNet, BasicBlock
from logger import MLFlowLogger2
from dataset import ECGSSLDataModule

from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.callbacks.self_supervised import BYOLMAWeightUpdate

class MLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_size=4096, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x

class SiameseArm(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()

        if encoder == None:
            encoder = ResNet(BasicBlock, [2, 2, 2, 2], dim=64) 
        self.encoder = encoder
        # Projector
        self.projector = MLP(input_dim=512,hidden_size=1024,output_dim=256)
        # Predictor
        self.predictor = MLP(input_dim=256,hidden_size=1024)

    def forward(self, x):
        y = self.encoder(x)
        y = y.view(y.size(0), -1)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h


class BYOL(pl.LightningModule):
    def __init__(self,
                 lr: float = 0.2,
                 weight_decay: float = 15e-6,
                 input_height: int = 32,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 warmup_epochs: int = 10,
                 max_epochs: int = 1000,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.online_network = SiameseArm()
        self.target_network = deepcopy(self.online_network)
        self.weight_callback = BYOLMAWeightUpdate()

    def on_train_batch_end(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # Add callback for user automatically since it's key to BYOL weight update
        self.weight_callback.on_train_batch_end(self.trainer, self, batch, batch_idx, dataloader_idx)

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def shared_step(self, batch, batch_idx):
        img_1, img_2 = batch

        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(img_1)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_2)
        # L2 normalize
        h1_norm = F.normalize(h1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        loss_a = F.mse_loss(h1_norm, z2_norm)

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(img_2)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_1)
        # L2 normalize
        h1_norm = F.normalize(h1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        loss_b = F.mse_loss(h1_norm, z2_norm)

        # Final loss
        total_loss = loss_a + loss_b

        return loss_a, loss_b, total_loss

    def training_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        result = pl.TrainResult(minimize=total_loss)
        #  result.log_dict({'1_2_loss': loss_a, '2_1_loss': loss_b, 'train_loss': total_loss})
        result.log_dict({'train_loss': total_loss})

        return result

    def validation_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        result = pl.EvalResult(early_stop_on=total_loss, checkpoint_on=total_loss)
        #  result.log_dict({'1_2_loss': loss_a, '2_1_loss': loss_b, 'train_loss': total_loss})
        result.log_dict({'valid_loss': total_loss})

        return result

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        optimizer = LARSWrapper(optimizer)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
        (args, _) = parser.parse_known_args()
        # Data
        parser.add_argument('--data_dir', type=str, default='.')
        parser.add_argument('--num_workers', default=10, type=int)
        # optim
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=15e-6)
        parser.add_argument('--warmup_epochs', type=float, default=10)
        # Model
        parser.add_argument('--meta_dir', default='.', type=str)
        # Custom
        parser.add_argument('--exp', type=str, default='Pretrain')
        parser.add_argument('--name', type=str, default='test')
        parser.add_argument('--benchmark', type=bool, default=True)
        parser.add_argument('--amp_level', type=str, default='O0')

        return parser


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = BYOL.add_model_specific_args(parser)
    args = parser.parse_args()

    # datamodule
    dm = ECGSSLDataModule(args)
    args.num_samples = dm.num_samples

    model = BYOL(**args.__dict__)

    trainer = pl.Trainer.from_argparse_args(args, max_steps=100000)
    if args.name != 'test':
        logger = MLFlowLogger2(experiment_name=args.exp, run_name=args.name)
    else:
        logger = None
    trainer.logger = logger
    trainer.fit(model, dm)
