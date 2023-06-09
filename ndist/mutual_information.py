if __name__ == "__main__":
    print(
        """
-. .-.   .-. .-.   .-  MUTUAL | Computing mutual information with swyft  
||\|||\ /|||\|||\ /||  Model: Normal Distributions
|/ \|||\|||/ \|||\|||  Module: mutual_information.py
~   `-~ `-`   `-~ `-`  Authors: J. Alvey"""
    )

import swyft
import swyft.lightning as sl
import numpy as np
import os
import torch
import glob
import pickle
from datetime import datetime
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

"""
--------------------------------------------------------------
SIMULATOR AND CLASS ESTIMATOR NETWORKS (REQUIRES MODIFICATION)
--------------------------------------------------------------
"""


class Simulator(swyft.Simulator):
    def __init__(self):
        super().__init__()
        self.transform_samples = swyft.to_numpy32

    def sample_prior(self):
        return np.array([np.random.uniform(-1., 1.)])

    def sample_data(self, theta):
        return np.random.normal(theta, 0.3)

    def build(self, graph):
        theta = graph.node("theta", self.sample_prior)
        data = graph.node("data", self.sample_data, theta)


class RatioEstimator(swyft.SwyftModule):
    def __init__(self):
        super().__init__()
        self.compression = None
        self.model_estimator = swyft.LogRatioEstimator_1dim(
            num_features=1, num_params=1, varnames="theta"
        )

    def forward(self, A, B):
        data = A["data"]
        theta = B["theta"]
        if self.compression is not None:
            summary = self.compression(data)
            return self.model_estimator(summary, theta)
        else:
            return self.model_estimator(data, theta)


"""
--------------------------------------------
UTILITY FUNCTIONS (NO MODIFICATION REQUIRED)
--------------------------------------------
"""


def setup_zarr_store(store_name, store_size, chunk_size, simulator):
    store = swyft.ZarrStore(file_path=store_name)
    shapes, dtypes = simulator.get_shapes_and_dtypes()
    store.init(N=store_size, chunk_size=chunk_size, shapes=shapes, dtypes=dtypes)
    return store


def load_zarr_store(store_path):
    if os.path.exists(path=store_path):
        return swyft.ZarrStore(file_path=store_path)
    else:
        raise ValueError(f"store path ({store_path}) does not exist")


def simulate(store, simulator, batch_size, one_batch=False):
    if one_batch:
        store.simulate(sampler=simulator, batch_size=batch_size, max_sims=batch_size)
    else:
        store.simulate(sampler=simulator, batch_size=batch_size)


def info(msg):
    print(f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [real-data.py] | {msg}")


def config_info(config):
    print("--------\nSETTINGS\n--------")
    for key in config.keys():
        print(key + " | " + str(config[key]))
    print("\n")


def setup_dataloader(
    store,
    simulator,
    trainer_dir,
    num_workers,
    train_fraction,
    val_fraction,
    train_batch_size,
    val_batch_size,
    resampler_targets=None,
):
    if not os.path.isdir(trainer_dir):
        os.mkdir(trainer_dir)
    if resampler_targets is not None:
        if type(resampler_targets) is not list:
            raise ValueError(
                f"resampler targets must be a list | resampler_targets = {resampler_targets}"
            )
        resampler = simulator.get_resampler(targets=resampler_targets)
    else:
        resampler = None
    train_data = store.get_dataloader(
        num_workers=num_workers,
        batch_size=train_batch_size,
        idx_range=[0, int(train_fraction * len(store))],
        on_after_load_sample=resampler,
    )
    val_data = store.get_dataloader(
        num_workers=num_workers,
        batch_size=val_batch_size,
        idx_range=[
            int(train_fraction * len(store)),
            len(store) - 1,
        ],
        on_after_load_sample=None,
    )
    return train_data, val_data


def setup_trainer(trainer_dir, early_stopping, device, n_gpus, min_epochs, max_epochs):
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=early_stopping,
        verbose=False,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{trainer_dir}",
        filename="{epoch}_{val_loss:.2f}_{train_loss:.2f}",
        mode="min",
    )
    logger_tbl = pl_loggers.TensorBoardLogger(
        save_dir=trainer_dir,
        name="swyft-trainer",
        version=None,
        default_hp_metric=False,
    )
    trainer = sl.SwyftTrainer(
        accelerator=device,
        gpus=n_gpus,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger=logger_tbl,
        callbacks=[lr_monitor, early_stopping_callback, checkpoint_callback],
    )
    return trainer

def compute_mutual_information(logratios_sample):
    lrs = logratios_sample.logratios
    return torch.mean(lrs)


if __name__ == "__main__":
    config = {
        "store_name": "ndist-store",
        "store_size": 10_000,
        "chunk_size": 500,
        "logratios_path": "ndist-logratios",
        "trainer_dir": "ndist-trainer",
        "resampler_targets": None,
        "train_fraction": 0.9,
        "val_fraction": 0.1,
        "train_batch_size": 128,
        "val_batch_size": 128,
        "num_workers": 8,
        "device": "gpu",
        "n_gpus": 1,
        "min_epochs": 1,
        "max_epochs": 10,
        "early_stopping": 7,
        "infer_only": True,
    }
    config_info(config=config)
    simulator = Simulator()
    if os.path.exists(path=config["store_name"]):
        info(f"Loading ZarrStore: {config['store_name']}")
        store = load_zarr_store(config["store_name"])
    else:
        info(f"Initialising ZarrStore: {config['store_name']}")
        store = setup_zarr_store(
            store_name=config["store_name"],
            store_size=config["store_size"],
            chunk_size=config["chunk_size"],
            simulator=simulator,
        )

    print(
        "\n----------------------------\nSTEP 1: GENERATE SIMULATIONS\n----------------------------\n"
    )
    if store.sims_required > 0:
        info(f"Simulating data into ZarrStore: {config['store_name']}")
        simulate(
            store=store,
            simulator=simulator,
            batch_size=config["chunk_size"],
            one_batch=False,
        )
        info(f"Simulations completed")
    else:
        info(f"ZarrStore ({config['store_name']}) already full, skipping simulations")
    print(
        "\n-----------------------\nSTEP 2: TRAIN ESTIMATOR\n-----------------------\n"
    )
    info(
        f"Setting up dataloaders and trainer with:"
        + f"\ntrainer_dir\t | {config['trainer_dir']}"
        + f"\ntrain_fraction\t | {config['train_fraction']}"
        + f"\nval_fraction\t | {config['val_fraction']}"
        + f"\ntrain_batch_size | {config['train_batch_size']}"
        + f"\nval_batch_size\t | {config['val_batch_size']}"
        + f"\nnum_workers\t | {config['num_workers']}"
    )
    train_data, val_data = setup_dataloader(
        store=store,
        simulator=simulator,
        trainer_dir=config["trainer_dir"],
        num_workers=config["num_workers"],
        train_fraction=config["train_fraction"],
        val_fraction=config["val_fraction"],
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
        resampler_targets=config["resampler_targets"],
    )
    trainer = setup_trainer(
        trainer_dir=config["trainer_dir"],
        early_stopping=config["early_stopping"],
        device=config["device"],
        n_gpus=config["n_gpus"],
        min_epochs=config["min_epochs"],
        max_epochs=config["max_epochs"],
    )
    network = RatioEstimator()
    if (
        not config["infer_only"]
        or len(glob.glob(f"{config['trainer_dir']}/epoch*.ckpt")) == 0
    ):
        info(f"Starting ratio estimator training")
        trainer.fit(network, train_data, val_data)
        info(
            f"Training completed, checkpoint available at {glob.glob(config['trainer_dir'] + '/epoch*.ckpt')[0]}"
        )
        info("To avoid re-training network, set infer_only = True in config")
    info(
        f"Loading optimal network weights from {glob.glob(config['trainer_dir'] + '/epoch*.ckpt')[0]}"
    )
    trainer.test(
        network, val_data, glob.glob(config["trainer_dir"] + "/epoch*.ckpt")[0]
    )
    print("\n-----------------\nSTEP 3: INFERENCE\n-----------------\n")
    info("Generating joint samples from simulator")
    joint_samples = simulator.sample(100_000)
    logratios_sample = trainer.infer(network, joint_samples, joint_samples)
    info(f"Saving joint logratios samples to {config['logratios_path']}")
    with open(config["logratios_path"], "wb") as f:
        pickle.dump(logratios_sample, f)
    info(f"Mutual Information: {compute_mutual_information(logratios_sample=logratios_sample)}")