import argparse
import json
import logging
from pathlib import Path, PosixPath

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)

torch.serialization.add_safe_globals([PosixPath])

def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to pre-processed dataset directory"
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        help="Save checkpoint every N epochs (default: 1)",
    )
    parser.add_argument(
        "--quality",
        default="medium",
        choices=("x-low", "medium", "high"),
        help="Quality/size of model (default: medium)",
    )
    parser.add_argument(
        "--resume_from_single_speaker_checkpoint",
        help="For multi-speaker models only. Converts a single-speaker checkpoint to multi-speaker and resumes training",
    )

    VitsModel.add_model_specific_args(parser)
    
    # Trainer-specific args
    parser.add_argument(
        "--accelerator",
        type=str,
    )
    parser.add_argument(
        "--devices",
        type=int,
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234
    )
    parser.add_argument(
        "--random_seed",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
    )
    parser.add_argument(
        "--precision",
        type=str,
    )
    parser.add_argument(
        "--num_ckpt",
        type=int,
        default=1,
        help="# of ckpts saved."
    )
    parser.add_argument(
        "--default_root_dir",
        type=str,
        help="Default root dir for checkpoints and logs."
    )
    parser.add_argument(
        "--save_last",
        type=bool,
        default=None,
        help="Always save the last checkpoint."
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        help="Metric to monitor."
    )
    parser.add_argument(
        "--monitor_mode",
        type=str,
        default="min",
        help="Mode to monitor."
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Number of validation cycles to allow to pass without improvement before stopping training"
    )
    args = parser.parse_args()

    
    _LOGGER.debug(args)

    args.dataset_dir = Path(args.dataset_dir)
    if not args.default_root_dir:
        args.default_root_dir = args.dataset_dir

    seed_everything(args.seed)

    config_path = args.dataset_dir / "config.json"
    dataset_path = args.dataset_dir / "dataset.jsonl"

    with open(config_path, "r", encoding="utf-8") as config_file:
        # See preprocess.py for format
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        num_speakers = int(config["num_speakers"])
        sample_rate = int(config["audio"]["sample_rate"])

    # Build checkpoint callback if needed
    callbacks = []
    if args.checkpoint_epochs:
        callbacks.append(ModelCheckpoint(
            every_n_epochs=args.checkpoint_epochs,
            save_last=args.save_last,
            save_top_k=args.num_ckpt,
            monitor=args.monitor,
            mode=args.monitor_mode,
        ))
        _LOGGER.debug(
            "Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs
        )
        
    if args.early_stop_patience > 0:
        callbacks.append(EarlyStopping(
            monitor=args.monitor,
            patience=args.early_stop_patience,
            mode=args.monitor_mode,
        ))

    # Prepare model args
    model_args = vars(args).copy()
    model_args["dataset"] = [dataset_path]
    if args.quality == "x-low":
        model_args.update({
            "hidden_channels": 96,
            "inter_channels": 96,
            "filter_channels": 384,
        })
    elif args.quality == "high":
        model_args.update({
            "resblock": "1",
            "resblock_kernel_sizes": (3, 7, 11),
            "resblock_dilation_sizes": ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            "upsample_rates": (8, 8, 2, 2),
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": (16, 16, 4, 4),
        })

    # Remove keys not accepted by model
    for k in list(model_args):
        if k not in VitsModel.__init__.__code__.co_varnames:
            model_args.pop(k)

    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        **model_args
    )

    if args.resume_from_single_speaker_checkpoint:
        assert (
            num_speakers > 1
        ), "--resume_from_single_speaker_checkpoint is only for multi-speaker models. Use --resume_from_checkpoint for single-speaker models."

        # Load single-speaker checkpoint
        _LOGGER.debug(
            "Resuming from single-speaker checkpoint: %s",
            args.resume_from_single_speaker_checkpoint,
        )
        model_single = VitsModel.load_from_checkpoint(
            args.resume_from_single_speaker_checkpoint,
            dataset=None,
        )
        g_dict = model_single.model_g.state_dict()
        for key in list(g_dict.keys()):
            # Remove keys that can't be copied over due to missing speaker embedding
            if (
                key.startswith("dec.cond")
                or key.startswith("dp.cond")
                or ("enc.cond_layer" in key)
            ):
                g_dict.pop(key, None)

        # Copy over the multi-speaker model, excluding keys related to the
        # speaker embedding (which is missing from the single-speaker model).
        load_state_dict(model.model_g, g_dict)
        load_state_dict(model.model_d, model_single.model_d.state_dict())
        _LOGGER.info(
            "Successfully converted single-speaker checkpoint to multi-speaker"
        )


    # Set up trainer with only valid Trainer args
    trainer_args = {
        "accelerator": args.accelerator,
        "devices": args.devices,
        "log_every_n_steps": args.log_every_n_steps,
        "max_epochs": args.max_epochs,
        "precision": args.precision,
        "default_root_dir": args.default_root_dir,
        "callbacks": callbacks,
    }

    trainer_args = {k: v for k, v in trainer_args.items() if v is not None}

    trainer = Trainer(**trainer_args)
    checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainer.fit(model)


def load_state_dict(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in saved_state_dict:
            # Use saved value
            new_state_dict[k] = saved_state_dict[k]
        else:
            # Use initialized value
            _LOGGER.debug("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
