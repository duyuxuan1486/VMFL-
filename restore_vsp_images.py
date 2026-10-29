#!/usr/bin/env python3
"""Restore VSP image paths from the original Mirage archives and VMFL JSONL."""

import argparse
import io
import json
import tarfile
from pathlib import Path

from PIL import Image

from evaluate import ACTION_RE, DIRECTIONS, draw_vsp


def records(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def destination(root, image_path):
    relative = Path(image_path).relative_to("data/images")
    if ".." in relative.parts:
        raise ValueError(f"Unsafe image path: {image_path}")
    path = root / "data/images" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def grid(record):
    size = record["level"]
    cells = [[0] * size for _ in range(size)]
    for row, col in record["hole"]:
        cells[row - 1][col - 1] = -1
    row, col = record["start"]
    cells[row - 1][col - 1] = 1
    row, col = record["end"]
    cells[row - 1][col - 1] = 2
    return cells


def archive_image(archive, name):
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(f"Missing archive image: {name}")
    return Image.open(io.BytesIO(member.read())).convert("RGB")


def restore_train(archive, root, rows):
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.extractfile("./train_direct.jsonl")
        if member is None:
            raise ValueError("Mirage training archive lacks train_direct.jsonl")
        sources = [json.loads(line) for line in member if line.strip()]
        if len(rows) != 1000 or len(sources) != 1000:
            raise ValueError("Expected 1,000 VSP training records in both files")
        for index, (record, source) in enumerate(zip(rows, sources)):
            if grid(record) != source["map_desc"]:
                raise ValueError(f"Mirage training map mismatch at record {index}")
            prefix = "./data/vsp_spatial_planning/"
            upstream_path = source["image_input"]
            if not upstream_path.startswith(prefix):
                raise ValueError(f"Unexpected Mirage image path: {upstream_path}")
            relative = Path(upstream_path[len(prefix):])
            if ".." in relative.parts:
                raise ValueError(f"Unsafe Mirage image path: {upstream_path}")
            image = archive_image(bundle, "./" + relative.as_posix())
            image.save(destination(root, record["images"][0]))
            row, col = record["start"]
            committed = image
            history = [(row, col, image)]
            holes = {tuple(point) for point in record["hole"]}
            for step, image_path in enumerate(record["images"][1:]):
                match = ACTION_RE.search(record["messages"][2 + 2 * step]["content"])
                if match is None:
                    raise ValueError(f"Missing action at VSP record {index}, step {step}")
                action = match.group(1).lower()
                dr, dc = DIRECTIONS[action]
                proposed = (row + dr, col + dc)
                inside = 1 <= proposed[0] <= record["level"] and 1 <= proposed[1] <= record["level"]
                feedback = draw_vsp(committed, row, col, action, green=inside)
                if inside and proposed not in holes:
                    if len(history) > 1 and proposed == history[-2][:2]:
                        history.pop()
                        row, col, feedback = history[-1]
                    else:
                        row, col = proposed
                        history.append((row, col, feedback))
                    committed = feedback
                feedback.save(destination(root, image_path))


def restore_test(archive, root, rows):
    with tarfile.open(archive, "r:gz") as bundle:
        for record in rows:
            level, index = record["level"], record["index"]
            if level not in range(3, 7) or index not in range(100):
                raise ValueError(f"Unexpected VSP test record: level={level}, index={index}")
            name = f"imgs_test/level{level}/img/{index}.png"
            archive_image(bundle, name).save(destination(root, record["images"][0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--test-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("."), help="Code repository root")
    args = parser.parse_args()
    root = args.output.resolve()
    images = root / "data/images"
    if images.is_symlink():
        raise ValueError(f"Refusing to write through image symlink: {images}")
    train = records(root / "data/vsp/train.jsonl")
    test = records(root / "data/vsp/test.jsonl")
    if len(train) != 1000 or len(test) != 400:
        raise ValueError("Expected 1,000 VSP training and 400 test records")
    restore_train(args.train_archive, root, train)
    restore_test(args.test_archive, root, test)
    print("Restored VSP images for 1,000 training and 400 test records")


if __name__ == "__main__":
    main()
