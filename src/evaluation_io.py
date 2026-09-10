"""Content-aware evaluation caching with a crash journal and atomic output replacement."""

import asyncio
import hashlib
import json
import os
import tempfile

from tqdm.auto import tqdm


def evaluation_key(instance, config):
    payload = {
        "prompt": instance["prompt"],
        "model": instance.get("model"),
        "generations": instance.get("generations"),
        "prompt_paraphrases": instance.get("prompt_paraphrases"),
        "partition": instance.get("partition"),
        "config": config,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def cached_rows(path):
    """Rows by id; a later row for the same id wins (journals may hold retries)."""
    if not os.path.exists(path):
        return {}
    result = {}
    with open(path) as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                result[row["id"]] = row
    return result


def write_rows(path, rows):
    """Keep the previous output intact if evaluation or writing fails."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".jsonl.tmp")
    try:
        with os.fdopen(fd, "w") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def plan(instances, output_file, key_field, config):
    """Rows already valid for this config, their keys, and the instances left to compute.

    Finished rows are journaled to `<output_file>.partial` as they land, so an
    interrupted run resumes; `finalize` replaces the output once all are done."""
    journal = output_file + ".partial"
    existing = cached_rows(output_file) | cached_rows(journal)
    keys = {x["id"]: evaluation_key(x, config) for x in instances}
    todo = [
        x for x in instances if existing.get(x["id"], {}).get(key_field) != keys[x["id"]]
    ]
    print(f"{output_file}: {len(instances) - len(todo)} cached, {len(todo)} to compute")
    return existing, keys, todo


def stamp(instance, fields, key_field, key, config):
    return {
        **instance,
        **fields,
        key_field: key,
        key_field.removesuffix("_key") + "_config": config,
    }


def finalize(instances, output_file, existing):
    write_rows(output_file, [{**x, **existing[x["id"]]} for x in instances])
    journal = output_file + ".partial"
    if os.path.exists(journal):
        os.unlink(journal)


async def run_cached(instances, output_file, key_field, config, compute, concurrency=1):
    """Recompute rows whose inputs or config changed; `compute(instance)` returns new fields."""
    existing, keys, todo = plan(instances, output_file, key_field, config)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(instance):
        async with semaphore:
            fields = await compute(instance)
        return stamp(instance, fields, key_field, keys[instance["id"]], config)

    failures = 0
    with open(output_file + ".partial", "a") as file:
        for task in tqdm(asyncio.as_completed([one(x) for x in todo]), total=len(todo)):
            try:
                row = await task
            except Exception as e:
                print(f"skipping instance: {e!r}")
                failures += 1
                continue
            file.write(json.dumps(row) + "\n")
            file.flush()
            existing[row["id"]] = row
    if failures:
        raise RuntimeError(
            f"{failures} instances failed; finished work kept in {output_file}.partial"
        )
    finalize(instances, output_file, existing)
