import pytorch_lightning as pl
from data.dataset import TradeDataModule
from model.model import TradeModule
import neptune
from pytorch_lightning.loggers.neptune import NeptuneLogger
import hydra
from hydra.core.config_store import ConfigStore
from config.trade_config import TradeConfig


@hydra.main(config_name='trade')
def main(cfg: TradeConfig):
    data = TradeDataModule(**dict(cfg.data))
    model = TradeModule(
        cfg.data.look_back, data.ds.data.shape[1], data.ds.full_data.shape[1]
    )

    trainer = pl.Trainer(
        logger=[NeptuneLogger(project_name='yoniosin/Trade')],
        max_epochs=cfg.max_epochs,
        fast_dev_run=True
    )

    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    # neptune.set_project('yoniosin/Trade')
    cs = ConfigStore()
    cs.store(name='trade', node=TradeConfig)
    main()
