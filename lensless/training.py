import torch
import piq

from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger


vgg_loss = piq.LPIPS()


class TrainingSystem(LightningModule):
    """
    Camera calibration system
    Reports results to TensorBoard
    """
    def __init__(self, model, region_of_interest):
        """
        composition > inheritance
        """
        super().__init__()
        self.model = model
        self.region_of_interest = region_of_interest

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0005)
        return {
            "optimizer": optimizer,
        }

    def forward(self, inputs):
        return self.model.forward(inputs)
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "Train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "Val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "Test")

    def step(self, batch, batch_idx, label):
        x, y = batch
        y_hat = self.model.forward(x)

        mse = mse_loss(y_hat, y)
        self.log(f"Reconstruction Loss/{label}", mse)
        
        accuracy = mse_loss(self.region_of_interest(y_hat), self.region_of_interest(y))
        self.log(f"Accuracy/{label}", accuracy)

        if batch_idx % 50 == 0:
            vgg = vgg_loss(y_hat, y)
            self.log(f"LPIPS/{label}", vgg)
            self.visualize_predictions(label, y_hat, y)

        return mse

    def visualize_predictions(self, label, y_hat, y):
        image = torch.cat([self.region_of_interest(y), self.region_of_interest(y_hat)], dim=-2)
        self.logger.experiment.add_images(f"Preds/{label}", image, self.global_step)
