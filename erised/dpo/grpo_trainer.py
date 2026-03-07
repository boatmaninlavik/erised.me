"""
GRPO (Group Relative Policy Optimization) trainer for HeartMuLa.

GRPO advantages over DPO:
- No reference model needed (saves ~6GB VRAM)
- Simpler loss computation
- Works with any group size (including pairs)

For pairs with binary preference:
- Winner reward = 1, Loser reward = 0
- Advantage_winner = +0.5, Advantage_loser = -0.5
- Loss pushes model to increase P(winner) and decrease P(loser)
"""

import os
import logging
from typing import Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from ..config import ErisedConfig
from .data import PreferenceStore
from .forward import compute_sequence_log_probs, build_training_sequence

logger = logging.getLogger(__name__)


class GRPOTrainer:
    def __init__(
        self,
        pipeline,  # ErisedPipeline
        config: ErisedConfig,
        pref_store: Optional[PreferenceStore] = None,
    ):
        self.pipeline = pipeline
        self.config = config
        self.pref_store = pref_store or PreferenceStore(config.dpo_db_path)
        self.device = torch.device(config.mula_device)

    def compute_grpo_loss(
        self,
        winner_log_prob: torch.Tensor,
        loser_log_prob: torch.Tensor,
        beta: float = 0.1,
    ) -> torch.Tensor:
        """
        GRPO loss for a pair (group size = 2).
        
        With binary rewards (winner=1, loser=0):
        - mean_reward = 0.5
        - advantage_winner = 1 - 0.5 = 0.5
        - advantage_loser = 0 - 0.5 = -0.5
        
        Loss = -sum(advantage_i * log_prob_i)
             = -0.5 * winner_log_prob - (-0.5) * loser_log_prob
             = -0.5 * winner_log_prob + 0.5 * loser_log_prob
             = 0.5 * (loser_log_prob - winner_log_prob)
        
        We want to minimize this, which means:
        - Increase winner_log_prob
        - Decrease loser_log_prob
        
        Adding beta scaling for stability:
        """
        advantage = 0.5  # For binary rewards with group size 2
        loss = advantage * (loser_log_prob - winner_log_prob)
        
        # Scale by beta (similar to DPO's beta for controlling update strength)
        return beta * loss

    def train(
        self,
        num_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        beta: Optional[float] = None,
        checkpoint_dir: str = "./grpo_checkpoints",
    ):
        """
        Run GRPO training on collected preferences.
        """
        num_epochs = num_epochs or self.config.dpo_epochs
        learning_rate = learning_rate or self.config.dpo_learning_rate
        beta = beta or self.config.dpo_beta

        pairs = self.pref_store.get_all()
        if not pairs:
            logger.warning("No preference pairs found. Skipping training.")
            return

        logger.info("Starting GRPO training with %d preference pairs", len(pairs))

        model = self.pipeline.get_model()
        model.train()

        # Enable gradient checkpointing to save memory
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        os.makedirs(checkpoint_dir, exist_ok=True)

        tokenizer = self.pipeline.pipe.tokenizer
        global_step = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for pair in pairs:
                # Load saved token tensors
                winner_tokens = torch.load(pair.winner_tokens, map_location=self.device)
                loser_tokens = torch.load(pair.loser_tokens, map_location=self.device)

                # Build full sequences (text + audio tokens)
                winner_seq = build_training_sequence(
                    tokenizer, pair.prompt, pair.lyrics, winner_tokens
                )
                loser_seq = build_training_sequence(
                    tokenizer, pair.prompt, pair.lyrics, loser_tokens
                )

                # Compute log probabilities
                winner_log_prob = compute_sequence_log_probs(
                    model, winner_seq, self.device
                )
                loser_log_prob = compute_sequence_log_probs(
                    model, loser_seq, self.device
                )

                # GRPO loss
                loss = self.compute_grpo_loss(winner_log_prob, loser_log_prob, beta)

                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % 10 == 0:
                    logger.info(
                        "Step %d | Loss: %.4f | Winner LP: %.2f | Loser LP: %.2f",
                        global_step, loss.item(), winner_log_prob.item(), loser_log_prob.item()
                    )

            avg_loss = epoch_loss / len(pairs)
            logger.info("Epoch %d/%d complete. Avg loss: %.4f", epoch + 1, num_epochs, avg_loss)

            # Save checkpoint
            ckpt_path = os.path.join(checkpoint_dir, f"grpo_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            logger.info("Saved checkpoint to %s", ckpt_path)

        logger.info("GRPO training complete.")
        return model
