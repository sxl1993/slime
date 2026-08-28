from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import CriticManifest, tokenizer_fingerprint, write_critic_artifact

DEFAULT_DATASET_NAME = "microsoft/Orchard"
DEFAULT_DATASET_CONFIG = "swe"
DEFAULT_DATASET_REVISION = "b642e9248ee3c0a87259193c5c5e6adc70322e9f"
DEFAULT_MAX_SEQ_LENGTH = 98_304


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare an offline Slime critic artifact from Orchard.")
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-per-instance", type=int, default=4)
    parser.add_argument("--canary-count", type=int, default=4096)
    return parser


def _existing_manifest_is_compatible(path: Path, args, tokenizer) -> bool:
    if not path.is_file():
        return False
    current = json.loads(path.read_text())
    return (
        current.get("dataset_revision") == args.dataset_revision
        and current.get("tokenizer_fingerprint") == tokenizer_fingerprint(tokenizer)
        and current.get("seed") == args.seed
        and current.get("max_per_instance") == args.max_per_instance
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if _existing_manifest_is_compatible(manifest_path, args, tokenizer):
            print(f"critic artifact already matches requested inputs: {manifest_path}")
            return 0
        raise SystemExit(f"refusing non-empty incompatible output directory: {output_dir}")

    if args.dataset_name != DEFAULT_DATASET_NAME or args.dataset_config != DEFAULT_DATASET_CONFIG:
        raise SystemExit("this workflow currently supports only microsoft/Orchard with config swe")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split="train",
        revision=args.dataset_revision,
    )
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator

    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")
    manifest: CriticManifest = write_critic_artifact(
        dataset,
        output_dir,
        tokenizer=tokenizer,
        mask_generator=mask_generator,
        dataset_revision=args.dataset_revision,
        shard_size=args.shard_size,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        max_per_instance=args.max_per_instance,
        canary_count=args.canary_count,
    )
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
