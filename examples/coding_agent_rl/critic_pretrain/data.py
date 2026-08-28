from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

Outcome = Literal["resolved", "unresolved"]
Split = Literal["train", "dev", "test"]
_OUTCOMES: tuple[Outcome, Outcome] = ("resolved", "unresolved")
_SPLITS: tuple[Split, Split, Split] = ("train", "dev", "test")
DEFAULT_MAX_SEQ_LENGTH = 98_304


@dataclass(frozen=True)
class CriticCandidate:
    row_index: int
    record_id: str
    instance_id: str
    outcome: Outcome


@dataclass(frozen=True)
class CriticRecord:
    record_id: str
    instance_id: str
    repo: str
    source: str
    outcome: Outcome
    reward: float
    tokens: list[int]
    response_length: int
    loss_mask: list[int]
    returns: list[float]


@dataclass(frozen=True)
class SplitManifest:
    valid_resolved: int
    valid_unresolved: int
    selected_resolved: int
    selected_unresolved: int
    raw: int = 0
    capped: int = 0


@dataclass(frozen=True)
class CriticManifest:
    schema_version: int
    dataset_revision: str
    tokenizer_fingerprint: str
    gamma: float
    lambd: float
    seed: int
    max_per_instance: int
    max_seq_length: int
    canary_count: int
    splits: dict[Split, SplitManifest]
    skipped: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["splits"] = {name: asdict(value) for name, value in self.splits.items()}
        return result


def _decode_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _decode_json(row.get("metadata", {}))
    return metadata if isinstance(metadata, Mapping) else {}


def _outcome(row: Mapping[str, Any]) -> Outcome | None:
    metadata = _metadata(row)
    value = metadata.get("verify_status") or metadata.get("outcome") or row.get("verify_status") or row.get("outcome")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _OUTCOMES:
            return normalized  # type: ignore[return-value]
    return None


def _instance_id(row: Mapping[str, Any]) -> str:
    metadata = _metadata(row)
    value = metadata.get("instance_id") or row.get("instance_id") or row.get("label")
    if not value:
        raise ValueError("Orchard row is missing metadata.instance_id")
    return str(value)


def _record_id(row: Mapping[str, Any], row_index: int | None = None) -> str:
    metadata = _metadata(row)
    instance_id = _instance_id(row)
    identity = next(
        (
            value
            for value in (
                metadata.get("sample_idx"),
                metadata.get("trajectory_id"),
                metadata.get("rollout_id"),
                row.get("id"),
                row_index,
            )
            if value is not None
        ),
        None,
    )
    if identity is None:
        messages = _decode_json(row.get("messages", row.get("conversation", [])))
        identity = hashlib.sha256(json.dumps(messages, sort_keys=True, default=str).encode()).hexdigest()[:16]
    status = _metadata(row).get("verify_status") or row.get("verify_status") or "unknown"
    return f"{instance_id}:{identity}:{status}"


def _stable_rank(record_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).digest()


def assign_instance_split(instance_id: str, seed: int = 17) -> Split:
    bucket = int.from_bytes(hashlib.sha256(f"{seed}:{instance_id}".encode()).digest()[:8], "big") % 100
    return "train" if bucket < 90 else "dev" if bucket < 95 else "test"


def select_instance_candidates(
    candidates: Sequence[CriticCandidate], *, max_per_instance: int = 4, seed: int = 17
) -> list[CriticCandidate]:
    """Select a deterministic, outcome-balanced prefix for each instance."""
    if max_per_instance < 1:
        raise ValueError("max_per_instance must be positive")

    grouped: dict[str, list[CriticCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.instance_id, []).append(candidate)

    selected: list[CriticCandidate] = []
    per_outcome = max_per_instance // len(_OUTCOMES)
    for instance_id in sorted(grouped):
        ranked = {
            outcome: sorted(
                (item for item in grouped[instance_id] if item.outcome == outcome),
                key=lambda item: _stable_rank(item.record_id, seed),
            )
            for outcome in _OUTCOMES
        }
        chosen: list[CriticCandidate] = []
        for outcome in _OUTCOMES:
            chosen.extend(ranked[outcome][:per_outcome])
        chosen_ids = {item.record_id for item in chosen}
        remaining = sorted(
            (item for item in grouped[instance_id] if item.record_id not in chosen_ids),
            key=lambda item: _stable_rank(item.record_id, seed),
        )
        chosen.extend(remaining[: max_per_instance - len(chosen)])
        selected.extend(sorted(chosen, key=lambda item: _stable_rank(item.record_id, seed)))
    return selected


def _messages_and_tools(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Any]:
    messages = _decode_json(row.get("messages", row.get("conversation")))
    tools = _decode_json(row.get("tools", []))
    if not isinstance(messages, list) or not messages:
        raise ValueError("missing messages")
    return messages, tools


def normalize_orchard_row(
    row: Mapping[str, Any], *, mask_generator: Any, max_seq_length: int, row_index: int | None = None
) -> tuple[CriticRecord | None, str | None]:
    """Render one Orchard conversation into a response-aligned critic record."""
    outcome = _outcome(row)
    if outcome is None:
        return None, "unknown_outcome"
    try:
        messages, tools = _messages_and_tools(row)
        tokens, full_mask = mask_generator.get_loss_mask(messages, tools)
    except Exception:
        return None, "template_error"

    if len(tokens) != len(full_mask):
        return None, "template_error"
    if len(tokens) > max_seq_length:
        return None, "overlength"
    try:
        first_action_index = next(index for index, value in enumerate(full_mask) if value)
    except StopIteration:
        return None, "no_action_tokens"

    response_length = len(tokens) - first_action_index
    if response_length <= 0:
        return None, "no_action_tokens"
    metadata = _metadata(row)
    repo = str(metadata.get("repo") or row.get("repo") or metadata.get("repository") or "unknown")
    record = CriticRecord(
        record_id=_record_id(row, row_index),
        instance_id=_instance_id(row),
        repo=repo,
        source="microsoft/Orchard",
        outcome=outcome,
        reward=1.0 if outcome == "resolved" else 0.0,
        tokens=[int(token) for token in tokens],
        response_length=response_length,
        loss_mask=[1 if value else 0 for value in full_mask[first_action_index:]],
        returns=[1.0 if outcome == "resolved" else 0.0] * response_length,
    )
    if len(record.loss_mask) != record.response_length:
        return None, "template_error"
    return record, None


def tokenizer_fingerprint(tokenizer: Any) -> str:
    name = getattr(tokenizer, "name_or_path", None) or f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    payload = {"name_or_path": str(name), "vocab_size": getattr(tokenizer, "vocab_size", None)}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode())
    path = Path(str(name))
    if path.is_dir():
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                digest.update(str(file_path.relative_to(path)).encode())
                with file_path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
    return digest.hexdigest()


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised on the remote training image
        raise RuntimeError("critic artifact writing requires pyarrow") from exc
    return pa, pq


def _rows_from_parquet(path: Path) -> Iterator[CriticRecord]:
    _, pq = _require_pyarrow()
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches():
        columns = batch.to_pydict()
        for index in range(batch.num_rows):
            yield CriticRecord(
                record_id=str(columns["record_id"][index]),
                instance_id=str(columns["instance_id"][index]),
                repo=str(columns["repo"][index]),
                source=str(columns["source"][index]),
                outcome=str(columns["outcome"][index]),  # type: ignore[arg-type]
                reward=float(columns["reward"][index]),
                tokens=[int(token) for token in columns["tokens"][index]],
                response_length=int(columns["response_length"][index]),
                loss_mask=[int(value) for value in columns["loss_mask"][index]],
                returns=(
                    [float(value) for value in columns["returns"][index]]
                    if "returns" in columns
                    else [float(columns["reward"][index])] * int(columns["response_length"][index])
                ),
            )


def _write_records(path: Path, records: Sequence[CriticRecord]) -> None:
    pa, pq = _require_pyarrow()
    table = pa.Table.from_pylist([asdict(record) for record in records])
    pq.write_table(table, path)


def write_critic_artifact(
    dataset,
    output_dir: Path,
    *,
    tokenizer: Any,
    dataset_revision: str,
    mask_generator: Any | None = None,
    shard_size: int = 256,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    seed: int = 17,
    max_per_instance: int = 4,
    canary_count: int = 4096,
) -> CriticManifest:
    """Write deterministic, balanced split/outcome Parquet shards."""
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if max_seq_length < 1:
        raise ValueError("max_seq_length must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if mask_generator is None:
        from slime.utils.mask_utils import MultiTurnLossMaskGenerator

        mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")

    candidates: list[CriticCandidate] = []
    get_row = getattr(dataset, "__getitem__", None)
    rows: dict[int, Mapping[str, Any]] = {}
    skipped: dict[str, int] = {}
    raw_by_split: dict[Split, int] = {split: 0 for split in _SPLITS}
    for row_index, row in enumerate(dataset):
        row = dict(row)
        if get_row is None:
            rows[row_index] = row
        try:
            instance_id = _instance_id(row)
        except ValueError:
            skipped["missing_instance_id"] = skipped.get("missing_instance_id", 0) + 1
            continue
        split = assign_instance_split(instance_id, seed)
        raw_by_split[split] += 1
        outcome = _outcome(row)
        if outcome is None:
            skipped["unknown_outcome"] = skipped.get("unknown_outcome", 0) + 1
            continue
        candidates.append(
            CriticCandidate(
                row_index=row_index,
                record_id=_record_id(row, row_index),
                instance_id=instance_id,
                outcome=outcome,
            )
        )

    capped_candidates = select_instance_candidates(candidates, max_per_instance=max_per_instance, seed=seed)
    capped_by_split = {split: 0 for split in _SPLITS}
    for candidate in capped_candidates:
        capped_by_split[assign_instance_split(candidate.instance_id, seed)] += 1

    valid: dict[Split, dict[Outcome, list[CriticRecord]]] = {
        split: {outcome: [] for outcome in _OUTCOMES} for split in _SPLITS
    }
    for candidate in capped_candidates:
        split = assign_instance_split(candidate.instance_id, seed)
        record, skip_reason = normalize_orchard_row(
            dict(get_row(candidate.row_index)) if get_row is not None else rows[candidate.row_index],
            mask_generator=mask_generator,
            max_seq_length=max_seq_length,
            row_index=candidate.row_index,
        )
        if record is None:
            reason = skip_reason or "unknown"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        valid[split][record.outcome].append(record)

    splits: dict[Split, SplitManifest] = {}
    for split in _SPLITS:
        for outcome in _OUTCOMES:
            valid[split][outcome].sort(key=lambda record: _stable_rank(record.record_id, seed))
        selected_count = min(len(valid[split]["resolved"]), len(valid[split]["unresolved"]))
        selected = {outcome: valid[split][outcome][:selected_count] for outcome in _OUTCOMES}
        for outcome in _OUTCOMES:
            output_path = output_dir / split / outcome
            output_path.mkdir(parents=True, exist_ok=True)
            for old_path in output_path.glob("part-*.parquet"):
                raise FileExistsError(f"refusing to overwrite existing critic shard: {old_path}")
            for start in range(0, selected_count, shard_size):
                _write_records(
                    output_path / f"part-{start // shard_size:05d}.parquet",
                    selected[outcome][start : start + shard_size],
                )
        splits[split] = SplitManifest(
            valid_resolved=len(valid[split]["resolved"]),
            valid_unresolved=len(valid[split]["unresolved"]),
            selected_resolved=selected_count,
            selected_unresolved=selected_count,
            raw=raw_by_split[split],
            capped=capped_by_split[split],
        )

    manifest = CriticManifest(
        schema_version=1,
        dataset_revision=dataset_revision,
        tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
        gamma=1.0,
        lambd=1.0,
        seed=seed,
        max_per_instance=max_per_instance,
        max_seq_length=max_seq_length,
        canary_count=canary_count,
        splits=splits,
        skipped=skipped,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return manifest


def iter_critic_records(artifact_dir: Path, split: Split, *, limit: int | None = None) -> Iterator[CriticRecord]:
    """Read equal outcome streams in stable resolved/unresolved order."""
    artifact_dir = Path(artifact_dir)
    streams = {
        outcome: (
            record
            for path in sorted((artifact_dir / split / outcome).glob("part-*.parquet"))
            for record in _rows_from_parquet(path)
        )
        for outcome in _OUTCOMES
    }
    emitted = 0
    while limit is None or emitted < limit:
        pair = []
        for outcome in _OUTCOMES:
            try:
                pair.append(next(streams[outcome]))
            except StopIteration:
                return
        for record in pair:
            if limit is not None and emitted >= limit:
                return
            yield record
            emitted += 1


def build_critic_train_data(records: Sequence[CriticRecord]) -> dict[str, Any]:
    """Build the fields consumed by Slime's packed-sequence training path."""
    import torch

    if not records:
        raise ValueError("at least one critic record is required")

    def target_tensor(record: CriticRecord):
        if len(record.returns) != record.response_length:
            raise ValueError("critic returns must match response_length")
        return torch.tensor(record.returns, dtype=torch.float32)

    return {
        "tokens": [torch.tensor(record.tokens, dtype=torch.long) for record in records],
        "loss_masks": [torch.tensor(record.loss_mask, dtype=torch.int) for record in records],
        "total_lengths": [len(record.tokens) for record in records],
        "response_lengths": [record.response_length for record in records],
        "returns": [target_tensor(record) for record in records],
        "rollout_mask_sums": torch.tensor([sum(record.loss_mask) for record in records], dtype=torch.float32),
    }


def build_critic_data_refs(
    args,
    train_parallel_config: Mapping[str, int],
    records: Sequence[CriticRecord],
    *,
    ray_put=None,
) -> list[Any]:
    """Partition one optimizer batch and package one object-store reference per DP rank."""
    import torch

    from slime.observability.rollout_data_utils import tensorize_rollout_data_for_training
    from slime.utils.dp_schedule import build_dp_schedule
    from slime.utils.misc import Box

    if ray_put is None:
        import ray

        ray_put = ray.put
    train_data = build_critic_train_data(records)
    rollout_ids = list(range(len(records)))
    partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
        args,
        dict(train_parallel_config),
        train_data["total_lengths"],
        global_batch_size=len(records),
        rollout_indices=rollout_ids,
    )
    refs = []
    for dp_rank, partition in enumerate(partitions):
        local_data: dict[str, Any] = {
            "tokens": [train_data["tokens"][index] for index in partition],
            "loss_masks": [train_data["loss_masks"][index] for index in partition],
            "returns": [train_data["returns"][index] for index in partition],
            "response_lengths": [train_data["response_lengths"][index] for index in partition],
            "total_lengths": train_data["total_lengths"],
            "rollout_mask_sums": train_data["rollout_mask_sums"][partition].clone(),
            "partition": partition,
            "micro_batch_indices": micro_batch_indices[dp_rank],
            "num_microbatches": num_microbatches,
            "global_batch_sizes": global_batch_sizes,
        }
        tensorize_rollout_data_for_training(local_data)
        local_data["returns"] = [
            torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous() for value in local_data["returns"]
        ]
        refs.append(Box(ray_put(local_data)))
    return refs
