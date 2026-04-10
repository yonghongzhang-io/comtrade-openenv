"""
train_grpo.py — GRPO fine-tuning for LLM-based UN Comtrade agent.

Overview
--------
Group Relative Policy Optimization (GRPO) trains the LLM to maximise
the reward signal from the comtrade_env environment.  For each task we
sample G rollouts, normalise rewards within the group, and update the
model with a clipped policy gradient loss — exactly the algorithm from
DeepSeekMath (Shao et al., 2024).

Training loop (one iteration):
  1. Sample a batch of tasks from comtrade_env (one per GPU worker).
  2. For each task, run G=4 agentic rollouts with the current policy.
  3. Score each rollout → reward ∈ [0, 1].
  4. Compute GRPO advantage: A_i = (r_i − mean(r)) / (std(r) + ε).
  5. Compute clipped surrogate loss + KL penalty against a reference model.
  6. Gradient step, repeat.

Usage
-----
# Quickstart with a local Ollama/vLLM API (no GPU, rollout-only mode):
  python train_grpo.py \
      --api-url http://localhost:11434/v1 \
      --api-model qwen2.5:7b \
      --num-iterations 200 \
      --batch-size 4 \
      --group-size 4

# HuggingFace model (requires GPU, full gradient training):
  python train_grpo.py \
      --hf-model Qwen/Qwen2.5-7B-Instruct \
      --num-iterations 200

No external OpenEnv server needed — the environment runs in-process.

Requirements
------------
  pip install torch transformers trl accelerate peft requests openai
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_grpo")


# ======================================================================
# Import local modules
# ======================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent))

from env_client import InProcessEnvClient
from agent import ComtradeAgent, LLMBackend


# ======================================================================
# Task IDs available in comtrade_env
# ======================================================================
ALL_TASK_IDS = [
    "T1_single_page",
    "T2_multi_page",
    "T3_duplicates",
    "T4_rate_limit_429",
    "T5_server_error_500",
    "T6_page_drift",
    "T7_totals_trap",
    "T8_mixed_faults",
    "T9_adaptive_adversary",
    "T10_multi_agent_coop",
]


# ======================================================================
# GRPO loss (no TRL dependency required — pure PyTorch)
# ======================================================================

def grpo_loss(
    log_probs: torch.Tensor,        # [B*G, T] current policy log-probs (with grad)
    old_log_probs: torch.Tensor,    # [B*G, T] rollout-time log-probs (no grad, detached)
    ref_log_probs: torch.Tensor,    # [B*G, T] reference model log-probs (no grad)
    advantages: torch.Tensor,       # [B*G]    normalised GRPO advantages
    attention_mask: torch.Tensor,   # [B*G, T] 1 for completion tokens
    clip_eps: float = 0.2,
    kl_coeff: float = 0.04,
) -> tuple[torch.Tensor, dict]:
    """
    Clipped surrogate loss + reverse-KL penalty from reference model.

    ratio = exp(log π_new - log π_old)  — uses rollout-time log probs as π_old,
    so ratio != 1 and clipping is active from the first gradient step.

    Returns (loss, stats_dict).
    """
    # Token-level policy ratio  r_t = exp(log π_new - log π_old)
    # old_log_probs must be computed with no_grad at rollout time (see _gradient_step).
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    # Expand advantages to token level
    adv = advantages.unsqueeze(-1).expand_as(log_probs)

    surrogate = torch.min(ratio * adv, clipped_ratio * adv)
    surrogate = (surrogate * attention_mask).sum(-1) / attention_mask.sum(-1).clamp(min=1)

    # KL penalty (reverse KL): D_KL(π_new || π_ref) = E[π_new/π_ref - 1 - log(π_new/π_ref)]
    # = E[exp(log_π_new - log_π_ref) - 1 - (log_π_new - log_π_ref)]
    log_ratio_ref = log_probs - ref_log_probs
    kl = torch.exp(log_ratio_ref) - 1 - log_ratio_ref
    kl = (kl * attention_mask).sum(-1) / attention_mask.sum(-1).clamp(min=1)

    loss = -(surrogate - kl_coeff * kl).mean()

    stats = {
        "loss": loss.item(),
        "kl": kl.mean().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
    }
    return loss, stats


# ======================================================================
# Tokenise trajectories for gradient computation
# ======================================================================

def tokenise_trajectories(
    tokenizer,
    prompts: list[str],
    completions: list[str],
    max_length: int = 2048,
    device: str = "cpu",
) -> dict:
    """
    Tokenise (prompt, completion) pairs.

    Returns:
      input_ids    [B, T]
      attention_mask [B, T]
      completion_mask [B, T]  — 1 only for completion tokens (for loss)
    """
    input_ids_list, attn_list, comp_list = [], [], []
    for prompt, completion in zip(prompts, completions):
        full_text = prompt + completion
        enc_full = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Compute prompt length in the joint encoding robustly.
        # Strategy: encode prompt WITH the same add_special_tokens setting as
        # full_text, but WITHOUT padding.  Then prompt_len = number of tokens
        # produced (includes BOS if the tokenizer adds one).  This is safe for
        # all tokenizer families (Qwen, Llama, Mistral, GPT-NeoX, etc.) because
        # the prefix of full_text is the prompt — the tokenizer will produce the
        # same tokens at the start regardless of what follows.
        enc_prompt_for_len = tokenizer(
            prompt,
            add_special_tokens=True,   # same as enc_full
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt_len = enc_prompt_for_len["input_ids"].shape[1]
        # If the standalone encoding added EOS but enc_full did not (because
        # text continues), subtract 1.  Detect by checking if the last prompt
        # token is EOS and the corresponding position in full_ids differs.
        full_ids = enc_full["input_ids"][0]
        if (
            tokenizer.eos_token_id is not None
            and prompt_len > 0
            and prompt_len <= full_ids.shape[0]
            and enc_prompt_for_len["input_ids"][0, -1].item() == tokenizer.eos_token_id
            and full_ids[prompt_len - 1].item() != tokenizer.eos_token_id
        ):
            prompt_len -= 1
        # Clamp to valid range in case of truncation
        prompt_len = min(prompt_len, full_ids.shape[0])
        comp_mask = torch.zeros_like(enc_full["attention_mask"])
        comp_mask[0, prompt_len:] = enc_full["attention_mask"][0, prompt_len:]

        input_ids_list.append(enc_full["input_ids"])
        attn_list.append(enc_full["attention_mask"])
        comp_list.append(comp_mask)

    return {
        "input_ids": torch.cat(input_ids_list).to(device),
        "attention_mask": torch.cat(attn_list).to(device),
        "completion_mask": torch.cat(comp_list).to(device),
    }


# ======================================================================
# Rollout collection
# ======================================================================

def collect_rollouts(
    llm: LLMBackend,
    task_ids: list[str],
    group_size: int,
    seed_offset: int = 0,
    max_workers: int = 4,
    max_steps: int = 30,
    temperature: float = 0.7,
) -> list[dict]:
    """
    For each task_id, run `group_size` episodes in parallel.

    Each worker creates its own InProcessEnvClient so episodes are fully
    isolated — no shared state between concurrent rollouts.

    Returns list of dicts:
      task_id, prompt, completion, reward, episode_idx, score, breakdown, error
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_one(task_id: str, g: int) -> dict:
        seed = seed_offset + hash(task_id) % 10_000 + g
        env = InProcessEnvClient()
        worker_agent = ComtradeAgent(
            llm=llm,
            env_client=env,
            max_steps=max_steps,
            temperature=temperature,
        )
        try:
            ep = worker_agent.run_episode(task_id=task_id, seed=seed)
            prompt = ComtradeAgent._format_task_description(ep.task_info)
            completion = ep.full_conversation()
            return {
                "task_id": task_id,
                "prompt": prompt,
                "completion": completion,
                "reward": ep.reward,
                "episode_idx": g,
                "score": ep.score,
                "breakdown": ep.breakdown,
                "error": ep.error,
            }
        except Exception as exc:
            logger.error(f"Rollout failed for {task_id} g={g}: {exc}")
            return {
                "task_id": task_id,
                "prompt": "",
                "completion": "",
                "reward": 0.0,
                "episode_idx": g,
                "score": 0.0,
                "breakdown": {},
                "error": str(exc),
            }

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, task_id, g): (task_id, g)
            for task_id in task_ids
            for g in range(group_size)
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    # Log per-task summary
    task_rewards: dict[str, list] = defaultdict(list)
    for r in results:
        task_rewards[r["task_id"]].append(r["reward"])
    for task_id in sorted(task_rewards):
        rewards = task_rewards[task_id]
        logger.info(
            f"  {task_id}: rewards={[f'{r:.3f}' for r in rewards]} "
            f"mean={sum(rewards)/len(rewards):.3f}"
        )

    return results


# ======================================================================
# GRPO advantages
# ======================================================================

def compute_advantages(rollouts: list[dict], group_size: int) -> list[float]:
    """
    Normalise rewards within each group (same task_id).

    A_i = (r_i - mean_group) / (std_group + eps)
    """
    # Group by task_id
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rollouts:
        groups[r["task_id"]].append(r["reward"])

    group_stats: dict[str, tuple[float, float]] = {}
    for task_id, rewards in groups.items():
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / max(len(rewards) - 1, 1))
        group_stats[task_id] = (mean, std)

    advantages = []
    for r in rollouts:
        mean, std = group_stats[r["task_id"]]
        adv = (r["reward"] - mean) / (std + 1e-8)
        advantages.append(adv)

    return advantages


# ======================================================================
# Training loop
# ======================================================================

def train(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # 1. Setup LLM backend
    # ------------------------------------------------------------------
    if args.hf_model:
        logger.info(f"Loading HuggingFace model: {args.hf_model}")
        llm = LLMBackend.from_hf(args.hf_model)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        # Reference model (frozen copy for KL divergence)
        from copy import deepcopy
        ref_model = deepcopy(model)
        for p in ref_model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
        )
        use_gradient_update = True

    elif args.api_url:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "none")
        logger.info(f"Using API backend: {args.api_url} model={args.api_model}")
        llm = LLMBackend.from_api(args.api_url, args.api_model, api_key=api_key)
        tokenizer = None
        model = None
        ref_model = None
        optimizer = None
        device = "cpu"
        use_gradient_update = False  # API mode: collect rollouts + log only

    else:
        raise ValueError("Specify either --hf-model or --api-url + --api-model")

    # ------------------------------------------------------------------
    # 2. Warm-up the in-process environment (starts mock service subprocess)
    # ------------------------------------------------------------------
    logger.info("Initialising in-process comtrade_env (first-time startup may take ~2s) ...")
    _warmup_env = InProcessEnvClient()
    _warmup_env.wait_until_ready()
    logger.info("Environment ready.")

    # ------------------------------------------------------------------
    # 3. Output directory
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    # ------------------------------------------------------------------
    # 4. Main training loop
    # ------------------------------------------------------------------
    logger.info(
        f"Starting GRPO training: {args.num_iterations} iterations, "
        f"batch={args.batch_size} tasks, group={args.group_size} rollouts"
    )

    global_step = 0
    all_task_ids = ALL_TASK_IDS.copy()

    for iteration in range(args.num_iterations):
        t_start = time.time()

        # Sample a batch of task IDs (with repetition allowed)
        batch_task_ids = random.choices(all_task_ids, k=args.batch_size)
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration+1}/{args.num_iterations} | tasks={batch_task_ids}")

        # ---- Collect rollouts (parallel, each worker gets its own env) ----
        rollouts = collect_rollouts(
            llm=llm,
            task_ids=batch_task_ids,
            group_size=args.group_size,
            seed_offset=iteration * 1000,
            max_workers=args.max_workers,
            max_steps=args.max_steps,
            temperature=args.temperature,
        )

        # Filter out failed rollouts
        valid = [r for r in rollouts if not r.get("error") and r["completion"]]

        if not valid:
            logger.warning("No valid rollouts this iteration. Skipping gradient step.")
            continue

        # ---- Compute GRPO advantages ----
        advantages = compute_advantages(rollouts, args.group_size)

        # Align advantages with valid rollouts only
        valid_advantages = [
            advantages[i]
            for i, r in enumerate(rollouts)
            if not r.get("error") and r["completion"]
        ]

        # ---- Metrics ----
        rewards = [r["reward"] for r in rollouts]
        mean_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)

        iter_metrics = {
            "iteration": iteration + 1,
            "mean_reward": mean_reward,
            "max_reward": max_reward,
            "n_valid": len(valid),
            "n_total": len(rollouts),
            "elapsed_s": time.time() - t_start,
        }

        # Task-level breakdown
        task_rewards: dict[str, list] = defaultdict(list)
        for r in rollouts:
            task_rewards[r["task_id"]].append(r["reward"])
        iter_metrics["task_rewards"] = {k: sum(v)/len(v) for k, v in task_rewards.items()}

        # ---- Gradient update (HF model only) ----
        if use_gradient_update and model is not None:
            loss_val, kl_val = _gradient_step(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                valid_rollouts=valid,
                valid_advantages=valid_advantages,
                max_length=args.max_seq_length,
                device=device,
                clip_eps=args.clip_eps,
                kl_coeff=args.kl_coeff,
            )
            iter_metrics["loss"] = loss_val
            iter_metrics["kl"] = kl_val
            global_step += 1

            # Periodic checkpoint
            if (iteration + 1) % args.save_every == 0:
                ckpt_path = out_dir / f"checkpoint-{iteration+1}"
                model.save_pretrained(str(ckpt_path))
                tokenizer.save_pretrained(str(ckpt_path))
                logger.info(f"Checkpoint saved: {ckpt_path}")

        # ---- Log ----
        logger.info(
            f"  mean_reward={mean_reward:.4f}  max_reward={max_reward:.4f}  "
            f"valid={len(valid)}/{len(rollouts)}"
        )
        if "loss" in iter_metrics:
            logger.info(f"  loss={iter_metrics['loss']:.4f}  kl={iter_metrics['kl']:.4f}")

        with open(metrics_path, "a") as f:
            f.write(json.dumps(iter_metrics) + "\n")

    logger.info("Training complete.")
    if use_gradient_update and model is not None:
        final_path = out_dir / "final"
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        logger.info(f"Final model saved: {final_path}")


# ======================================================================
# Gradient step helper
# ======================================================================

def _gradient_step(
    model,
    ref_model,
    tokenizer,
    optimizer,
    valid_rollouts: list[dict],
    valid_advantages: list[float],
    max_length: int,
    device: str,
    clip_eps: float,
    kl_coeff: float,
) -> tuple[float, float]:
    """Run one GRPO gradient step. Returns (loss, kl)."""
    model.train()

    prompts = [r["prompt"] for r in valid_rollouts]
    completions = [r["completion"] for r in valid_rollouts]

    batch = tokenise_trajectories(
        tokenizer=tokenizer,
        prompts=prompts,
        completions=completions,
        max_length=max_length,
        device=device,
    )
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    completion_mask = batch["completion_mask"]

    advantages_t = torch.tensor(valid_advantages, dtype=torch.float32, device=device)

    labels = input_ids[:, 1:]
    comp_mask_shifted = completion_mask[:, 1:]

    # --- Step 1: collect old_log_probs and ref_log_probs with no_grad ---
    # old_log_probs = π_old (current policy before this gradient step).
    # Must be computed BEFORE the optimizer step so ratio != 1.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            old_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            old_logits = old_outputs.logits[:, :-1, :]
            old_lp_full = torch.nn.functional.log_softmax(old_logits, dim=-1)
            old_token_log_probs = old_lp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            old_token_log_probs = (old_token_log_probs * comp_mask_shifted).detach()

            ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_lp_full = torch.nn.functional.log_softmax(ref_logits, dim=-1)
            ref_token_log_probs = ref_lp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            ref_token_log_probs = (ref_token_log_probs * comp_mask_shifted).detach()

    # --- Step 2: forward pass with grad to get new log_probs ---
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        log_probs_full = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * comp_mask_shifted

        loss, stats = grpo_loss(
            log_probs=token_log_probs,
            old_log_probs=old_token_log_probs,
            ref_log_probs=ref_token_log_probs,
            advantages=advantages_t,
            attention_mask=comp_mask_shifted,
            clip_eps=clip_eps,
            kl_coeff=kl_coeff,
        )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return stats["loss"], stats["kl"]


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training for comtrade LLM agent")

    # Model backend (mutually exclusive)
    backend = p.add_mutually_exclusive_group()
    backend.add_argument("--hf-model", type=str, default=None,
                         help="HuggingFace model name for local training "
                              "(e.g. Qwen/Qwen2.5-7B-Instruct)")
    backend.add_argument("--api-url", type=str, default=None,
                         help="OpenAI-compatible API base URL "
                              "(e.g. http://localhost:11434/v1)")
    p.add_argument("--api-model", type=str, default="qwen2.5:7b",
                   help="Model name for API backend (default: qwen2.5:7b)")
    p.add_argument("--api-key", type=str, default=None,
                   help="API key for the backend (or set OPENAI_API_KEY env var)")

    # Rollout parallelism
    p.add_argument("--max-workers", type=int, default=4,
                   help="Parallel rollout workers (each gets its own in-process env, default: 4)")

    # GRPO hyperparameters
    p.add_argument("--num-iterations", type=int, default=200,
                   help="Number of training iterations (default: 200)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Tasks per iteration (default: 4)")
    p.add_argument("--group-size", type=int, default=4,
                   help="Rollouts per task for GRPO normalisation (default: 4)")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Learning rate (default: 1e-5)")
    p.add_argument("--clip-eps", type=float, default=0.2,
                   help="PPO clip epsilon (default: 0.2)")
    p.add_argument("--kl-coeff", type=float, default=0.04,
                   help="KL divergence penalty coefficient (default: 0.04)")
    p.add_argument("--max-seq-length", type=int, default=2048,
                   help="Max token length for training sequences (default: 2048)")
    p.add_argument("--max-steps", type=int, default=30,
                   help="Max agentic steps per episode (default: 30)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature for rollouts (default: 0.7)")

    # Output
    p.add_argument("--output-dir", type=str, default="./grpo_output",
                   help="Directory for checkpoints + metrics (default: ./grpo_output)")
    p.add_argument("--save-every", type=int, default=50,
                   help="Save checkpoint every N iterations (default: 50)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
