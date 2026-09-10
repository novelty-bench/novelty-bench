"""Content-aware evaluation caching and atomic output replacement."""

import hashlib
import json
import os
import tempfile


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
    if not os.path.exists(path):
        return {}
    result = {}
    with open(path) as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in result:
                raise ValueError(f"Duplicate ID {row['id']!r} in {path}:{number}")
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
