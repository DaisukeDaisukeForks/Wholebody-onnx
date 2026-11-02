"""Small, explicit helpers for preserving logical sampling across worker changes."""
from bisect import bisect_right
from collections import Counter


def layout(config):
    workers, accumulate = config["workers"], config["accumulate"]
    logical_workers = config.get("logical_workers", workers)
    logical_accumulate = config.get("logical_accumulate", accumulate)
    values = (workers, accumulate, logical_workers, logical_accumulate)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Worker and accumulation values must be positive integers")
    if not 2 <= workers <= 8 or workers % logical_workers:
        raise ValueError("Physical workers must be 2..8 and a multiple of logical workers")
    lanes = workers // logical_workers
    if logical_accumulate != accumulate * lanes:
        raise ValueError("Physical lanes must exactly cover the original logical accumulation")
    return dict(logical_workers=logical_workers, logical_accumulate=logical_accumulate,
                workers=workers, accumulate=accumulate, lanes=lanes)


def worker_layout(config, rank):
    value = layout(config)
    if type(rank) is not int or not 0 <= rank < value["workers"]:
        raise ValueError("Worker rank is outside the physical topology")
    lane = rank // value["logical_workers"]
    return dict(logical_rank=rank % value["logical_workers"], lane=lane,
                microstep_start=lane * value["accumulate"])


def canonical_training_config(config):
    value, result = layout(config), dict(config)
    result.update(workers=value["logical_workers"], accumulate=value["logical_accumulate"])
    return result


def record_topology(checkpoint, config, run_key, utc):
    """Record an inclusive next-step boundary without resetting any learned history."""
    value, step, plan = layout(config), checkpoint["step"], checkpoint["plan"]
    records = [dict(entry) for entry in checkpoint.get("topology_history", [])]
    if not records and checkpoint["history"]:
        first = checkpoint["history"][0]
        packets = first["contributors"]
        old_accumulate, remainder = divmod(packets[0]["samples3d"], plan["batch3d"])
        if remainder or not old_accumulate:
            raise ValueError("Cannot infer the legacy accumulation from saved contributions")
        legacy = layout(dict(workers=len(packets), accumulate=old_accumulate))
        records.append(dict(legacy, start_step=0, rng_scheme="legacy-per-worker-round",
                            run_key=first.get("run_key", "legacy"),
                            utc=checkpoint.get("created_at", utc), inferred_from_history=True))
    if records:
        previous = layout(records[-1])
        if any(previous[key] != value[key] for key in ("logical_workers", "logical_accumulate")):
            raise ValueError("A topology change cannot repartition the saved logical streams")
    for task, batch in (("2d", config["batch2d"]), ("3d", config.get("batch3d", plan["batch3d"]))):
        if value["workers"] * value["accumulate"] * batch != plan[f"global_batch{task}"]:
            raise ValueError(f"A topology change cannot alter the global {task} batch")
    entry = dict(value, start_step=step, rng_scheme="per-logical-microstep", run_key=run_key, utc=utc)
    changed = not records or layout(records[-1]) != value or records[-1].get("rng_scheme") != entry["rng_scheme"]
    if changed:
        if records and records[-1]["start_step"] == step:
            records[-1] = entry  # No updates were committed in this empty range.
        else:
            records.append(entry)
    counts = dict(checkpoint["contribution_counts"])
    for rank in range(value["workers"]):
        counts.setdefault(str(rank), 0)
    candidate = dict(checkpoint, topology_history=records, contribution_counts=counts)
    validate_contributions(candidate)
    checkpoint.update(topology_history=records, contribution_counts=counts)
    return records[-1]


def validate_contributions(checkpoint):
    """Reconstruct committed contributions using zero-based topology ranges."""
    step, history = checkpoint["step"], checkpoint["history"]
    records, plan = checkpoint.get("topology_history", []), checkpoint["plan"]
    if type(step) is not int or step < 0 or len(history) != step:
        raise ValueError("Committed history must contain exactly steps 1..checkpoint.step")
    starts = [entry["start_step"] for entry in records]
    if (not starts or starts[0] != 0 or any(type(start) is not int or not 0 <= start <= step for start in starts)
            or any(left >= right for left, right in zip(starts, starts[1:]))):
        raise ValueError("Topology ranges must start at 0 and have increasing committed-step boundaries")
    layouts = [layout(entry) for entry in records]
    if any((value["logical_workers"], value["logical_accumulate"]) !=
           (layouts[0]["logical_workers"], layouts[0]["logical_accumulate"]) for value in layouts):
        raise ValueError("Historical topologies changed the logical sampling layout")
    global2d, global3d = plan["global_batch2d"], plan["global_batch3d"]
    counts = Counter({str(rank): 0 for value in layouts for rank in range(value["workers"])})
    for expected_step, event in enumerate(history, 1):
        if event["step"] != expected_step:
            raise ValueError("Committed history has a duplicate, gap or out-of-order step")
        workers = layouts[bisect_right(starts, expected_step - 1) - 1]["workers"]
        packets = event["contributors"]
        if sorted(packet["worker_id"] for packet in packets) != list(range(workers)):
            raise ValueError(f"Step {expected_step} does not contain its exact expected worker set")
        if global2d % workers or global3d % workers:
            raise ValueError("Global task batches cannot be divided evenly across this topology")
        for packet in packets:
            if packet["samples2d"] != global2d // workers or packet["samples3d"] != global3d // workers:
                raise ValueError(f"Step {expected_step} changed its per-update task sample counts")
            counts[str(packet["worker_id"])] += 1
    if dict(counts) != checkpoint["contribution_counts"]:
        raise ValueError("Saved worker contribution counters differ from committed history")
    if checkpoint.get("samples_seen2d", step * global2d) != step * global2d:
        raise ValueError("Saved 2D sample counter differs from committed history")
    if "source_seen3d" in checkpoint and sum(checkpoint["source_seen3d"].values()) != step * global3d:
        raise ValueError("Saved 3D source counters differ from committed task counts")
    ranges = [dict(entry, end_step=starts[index + 1] if index + 1 < len(starts) else step)
              for index, entry in enumerate(records)]
    return dict(step=step, topology_history=ranges, contribution_counts=dict(counts),
                global_batch2d=global2d, global_batch3d=global3d,
                samples_seen2d=step * global2d, samples_seen3d=step * global3d)
