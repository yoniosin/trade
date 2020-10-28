from dataclasses import dataclass, field


@dataclass
class TradeNetConfig:
    lr: float = 5e-3
    # layer_sizes: list = field(default_factory=[])


@dataclass
class TradeDataConfig:
    train_ratio: float = 0.8
    threshold: float = 100.
    look_back: int = 5
    batch_size: int = 10


@dataclass
class TradeConfig:
    net: TradeNetConfig = TradeNetConfig()
    data: TradeDataConfig = TradeDataConfig()
    max_epochs: int = 20
