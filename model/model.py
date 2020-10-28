from pytorch_lightning import LightningModule
from torch import nn
from torch import optim


class TradeModule(LightningModule):
    def __init__(self, look_back, data_size, full_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(full_size * look_back, full_size * 2),
            nn.ReLU(),
            nn.Linear(full_size * 2, data_size * 2),
            nn.ReLU(),
            nn.Linear(data_size * 2, data_size),
            nn.ReLU(),
            nn.Linear(data_size, data_size * 3)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        pred = self.layers(x).view(batch_size, 3, -1)
        loss = self.loss_fn(pred, y)

        return loss

    def training_step(self, batch, _):
        loss = self(**batch)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, _):
        loss = self(**batch)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self): return optim.Adam(self.parameters(), lr=5e-3)
