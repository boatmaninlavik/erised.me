"""
Entry points for the Erised system.

Usage:
    python -m erised serve          — Start the API server
    python -m erised generate       — Generate a song from CLI
    python -m erised train          — Run GRPO training on collected preferences (recommended)
    python -m erised dpo-train      — Run DPO training (legacy, uses more memory)
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("erised")


def cmd_serve(args):
    from .server import main as serve_main
    serve_main()


def cmd_generate(args):
    import torch
    from .config import ErisedConfig
    from .pipeline import ErisedPipeline

    config = ErisedConfig.from_env()
    if args.model_path:
        config.model_path = args.model_path

    pipeline = ErisedPipeline(config)

    with torch.no_grad():
        result = pipeline.generate(
            prompt=args.prompt,
            lyrics=args.lyrics,
            max_audio_length_ms=args.max_length,
        )

    print(f"Generated: {result.audio_path}")
    print(f"Tags used: {result.tags_used}")
    print(f"Tokens saved: {result.tokens_path}")


def cmd_train(args):
    """GRPO training (recommended) - uses less memory than DPO."""
    from .config import ErisedConfig
    from .pipeline import ErisedPipeline
    from .dpo.data import PreferenceStore
    from .dpo.grpo_trainer import GRPOTrainer

    config = ErisedConfig.from_env()
    if args.model_path:
        config.model_path = args.model_path
    config.lazy_load = False  # need model resident for training

    pipeline = ErisedPipeline(config)
    pref_store = PreferenceStore(config.dpo_db_path)

    n = pref_store.count()
    if n == 0:
        logger.error("No preference data found. Use /generate-pair and /rate endpoints first.")
        sys.exit(1)

    logger.info("Found %d preference pairs. Starting GRPO training...", n)

    trainer = GRPOTrainer(
        pipeline=pipeline,
        config=config,
        pref_store=pref_store,
    )
    trainer.train(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )


def cmd_dpo_train(args):
    """Legacy DPO training - uses more memory (needs reference model)."""
    from .config import ErisedConfig
    from .pipeline import ErisedPipeline
    from .dpo.data import PreferenceStore
    from .dpo.trainer import DPOTrainer

    config = ErisedConfig.from_env()
    if args.model_path:
        config.model_path = args.model_path
    config.lazy_load = False  # need both models resident for training

    pipeline = ErisedPipeline(config)
    pref_store = PreferenceStore(config.dpo_db_path)

    n = pref_store.count()
    if n == 0:
        logger.error("No preference data found. Use /generate-pair and /rate endpoints first.")
        sys.exit(1)

    logger.info("Found %d preference pairs. Starting DPO training...", n)

    trainer = DPOTrainer(
        pipeline=pipeline.pipe,
        preference_store=pref_store,
        beta=config.dpo_beta,
        learning_rate=config.dpo_learning_rate,
        epochs=config.dpo_epochs,
    )
    trainer.train()


def main():
    parser = argparse.ArgumentParser(prog="erised", description="Erised Music Generation System")
    sub = parser.add_subparsers(dest="command")

    # serve
    sub.add_parser("serve", help="Start the API server")

    # generate
    gen = sub.add_parser("generate", help="Generate a song from CLI")
    gen.add_argument("--prompt", type=str, required=True, help="Musical description/prompt")
    gen.add_argument("--lyrics", type=str, required=True, help="Lyrics text or path to .txt file")
    gen.add_argument("--model-path", type=str, default=None)
    gen.add_argument("--max-length", type=int, default=240_000)

    # train (GRPO - recommended)
    train = sub.add_parser("train", help="Run GRPO training on collected preferences (recommended)")
    train.add_argument("--model-path", type=str, default=None)
    train.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    train.add_argument("--lr", type=float, default=1e-7, help="Learning rate")
    train.add_argument("--checkpoint-dir", type=str, default="/workspace/grpo_checkpoints")

    # dpo-train (legacy)
    dpo = sub.add_parser("dpo-train", help="Run DPO training (legacy, uses more memory)")
    dpo.add_argument("--model-path", type=str, default=None)

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "dpo-train":
        cmd_dpo_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
