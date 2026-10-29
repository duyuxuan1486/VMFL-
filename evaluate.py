#!/usr/bin/env python3
"""Evaluate the paper's Qwen2.5-VL VMFL checkpoint with its online renderer."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


ANSWER_RE = re.compile(r"<answer_start>\s*(.*?)\s*<answer_end>", re.I | re.S)
ACTION_RE = re.compile(r"<action_start>\s*(left|right|up|down)\s*<action_end>", re.I)
MINI_ACTION_RE = re.compile(r"<action_start>\s*(.*?)\s*<action_end>", re.I | re.S)
SIZE_RE = re.compile(r"<size_start>\s*(\d+)\s*<size_end>", re.I)
GRID_RE = re.compile(r"<grid_pixel_size_start>\s*([\d.]+)\s*<grid_pixel_size_end>", re.I)
PIXEL_RE = re.compile(r"<pixel_coor_start>\s*(.*?)\s*<pixel_coor_end>", re.I | re.S)
PIXEL1_RE = re.compile(r"<pixel_coor_start1>\s*(.*?)\s*<pixel_coor_end1>", re.I | re.S)
COOR_RE = re.compile(r"<coor_start>\s*(.*?)\s*<coor_end>", re.I | re.S)
VSP_START_RE = re.compile(r"<start_point_start>\s*(.*?)\s*<start_point_end>", re.I | re.S)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PIXEL_DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
MARKER_SIZE = {4: 12, 5: 11, 6: 10, 7: 9, 8: 8, 9: 7, 10: 6, 12: 4, 15: 3, 20: 3}
NAV_MARKER = {3: 13, 4: 12, 5: 11, 6: 10}
NAV_ARROW = {3: 13, 4: 13, 5: 11, 6: 10}
NAV_DASH = {3: 7, 4: 5, 5: 4, 6: 3}
FEEDBACK = "<generation_start>executing the image editing.<generation_end> The generated image is <image>."
VSP_FEEDBACK = "The generated image is <image>."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["maze_plan", "maze_navigation", "mini_behavior", "vsp"], default="maze_plan")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    return parser.parse_args()


def message(role: str, text: str, image=None) -> dict:
    content = [{"type": "text", "text": text}]
    if image is not None:
        content.append({"type": "image", "image": image})
    return {"role": role, "content": content}


def parse_pair(pattern: re.Pattern, output: str, cast) -> list:
    match = pattern.search(output)
    if not match:
        raise ValueError(f"missing {pattern.pattern.split('>')[0]} tag")
    values = NUMBER_RE.findall(match.group(1))
    if len(values) < 2:
        raise ValueError("coordinate tag has fewer than two values")
    return [cast(values[0]), cast(values[1])]


def callback_status(output: str) -> bool | None:
    lowered = output.lower()
    if "is invalid" in lowered or "cannot move" in lowered:
        return False
    if "is valid" in lowered or "continue to go" in lowered:
        return True
    return None


def draw_candidate(image: Image.Image, pixel: list[float], action: str, grid: float, marker: int) -> tuple[Image.Image, list[float]]:
    candidate = image.copy()
    dx, dy = PIXEL_DIRECTIONS[action]
    proposed = [pixel[0] + dx * grid, pixel[1] + dy * grid]
    draw = ImageDraw.Draw(candidate)
    # The stored VMFL training images use the marker width itself for the
    # smallest large-maze markers (12x12 and above), while larger markers use
    # marker-2. This reproduces the training PNGs pixel-for-pixel.
    line_width = marker if marker <= 4 else marker - 2
    draw.line((pixel[0], pixel[1], proposed[0], proposed[1]), fill=(255, 0, 0), width=line_width)
    radius = marker // 2
    draw.rectangle(
        (proposed[0] - radius, proposed[1] - radius, proposed[0] + radius, proposed[1] + radius),
        fill="green",
    )
    return candidate, proposed


def draw_navigation(image: Image.Image, pixel: list[float], action: str, grid: float, size: int) -> tuple[Image.Image, list[float]]:
    candidate = image.copy()
    dx, dy = PIXEL_DIRECTIONS[action]
    end = [pixel[0] + dx * grid, pixel[1] + dy * grid]
    arrow = NAV_ARROW[size]
    length = math.hypot(end[0] - pixel[0], end[1] - pixel[1])
    ux, uy = (end[0] - pixel[0]) / length, (end[1] - pixel[1]) / length
    px, py = -uy, ux
    base = (end[0] - ux * arrow * 2, end[1] - uy * arrow * 2)
    left = (base[0] + px * arrow / 2, base[1] + py * arrow / 2)
    right = (base[0] - px * arrow / 2, base[1] - py * arrow / 2)
    draw = ImageDraw.Draw(candidate)
    draw.line([tuple(pixel), base], fill=(255, 0, 0), width=arrow // 2)
    draw.polygon([tuple(end), left, right], fill=(255, 0, 0))
    half = NAV_MARKER[size]
    x, y = end
    draw.line([(x - half, y - half), (x + half, y + half)], fill=(255, 0, 0), width=NAV_DASH[size])
    draw.line([(x - half, y + half), (x + half, y - half)], fill=(255, 0, 0), width=NAV_DASH[size])
    return candidate, end


def draw_mini(image: Image.Image, pixel: list[float], action: str, grid: float, color: tuple[int, int, int]) -> tuple[Image.Image, list[float]]:
    candidate = image.copy()
    dx, dy = PIXEL_DIRECTIONS[action.removeprefix("go ")]
    end = [pixel[0] + dx * grid, pixel[1] + dy * grid]
    draw = ImageDraw.Draw(candidate)
    draw.line((*pixel, *end), fill=color, width=6)
    draw.rectangle((end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4), fill="green")
    return candidate, end


def draw_vsp(image: Image.Image, row: int, col: int, action: str, green: bool) -> Image.Image:
    # Matches the VSP source renderer's 64-pixel cells and 10-pixel red line.
    x, y = (col - 0.5) * 64, (row - 0.5) * 64
    dx, dy = PIXEL_DIRECTIONS[action]
    length = 60 if green else 67
    endpoint = (x + dx * length, y + dy * length)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.line(((x, y), endpoint), fill="red", width=10)
    if green:
        ex, ey = endpoint
        if action == "up":
            rectangle = (ex - 6, ey - 12, ex + 6, ey)
        elif action == "down":
            rectangle = (ex - 6, ey, ex + 6, ey + 12)
        elif action == "left":
            rectangle = (ex - 12, ey - 6, ex, ey + 6)
        else:
            rectangle = (ex, ey - 6, ex + 12, ey + 6)
        draw.rectangle(rectangle, fill="green")
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def score_vsp(record: dict, answer: str | None) -> bool:
    if answer is None:
        return False
    actions = [part.strip().lower() for part in answer.split(",")]
    if not actions or any(action not in DIRECTIONS for action in actions):
        return False
    size = int(record["level"])
    row, col = record["start"]
    goal = tuple(record["end"])
    holes = {tuple(position) for position in record["hole"]}
    for action in actions:
        dr, dc = DIRECTIONS[action]
        candidate = (row + dr, col + dc)
        if not (1 <= candidate[0] <= size and 1 <= candidate[1] <= size):
            continue
        if candidate in holes:
            return False
        row, col = candidate
        if candidate == goal:
            return True
    return (row, col) == goal


def score(record: dict, answer: str | None) -> bool:
    if answer is None:
        return False
    actions = [part.strip().lower() for part in answer.split(",")]
    if not actions or any(action not in DIRECTIONS for action in actions):
        return False
    can_down, can_right = record["connection_list"]
    row, col = map(int, record["start"])
    level = int(record["level"])
    for action in actions:
        valid = False
        if action == "right":
            valid = col + 1 < level and can_right[row][col]
        elif action == "left":
            valid = col - 1 >= 0 and can_right[row][col - 1]
        elif action == "down":
            valid = row + 1 < level and can_down[row][col]
        elif action == "up":
            valid = row - 1 >= 0 and can_down[row - 1][col]
        if valid:
            dr, dc = DIRECTIONS[action]
            row, col = row + dr, col + dc
    return [row, col] == [int(value) for value in record["end"]]


class Generator:
    def __init__(self, args: argparse.Namespace):
        # The paper environment predates the Transformers fast-image-processor default.
        self.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map={"": "cuda"}
        ).eval()
        self.max_new_tokens = args.max_new_tokens

    @torch.inference_mode()
    def __call__(self, messages: list[dict]) -> str:
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt], images=images, videos=videos, padding=True, return_tensors="pt"
        ).to("cuda")
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_ids = generated[:, inputs.input_ids.shape[-1]:]
        return self.processor.batch_decode(new_ids, skip_special_tokens=False)[0]


def evaluate_maze_plan(args: argparse.Namespace, generator: Generator, record: dict, index: int) -> dict:
    original = Image.open(record["images"][0]).convert("RGB")
    messages = [
        message("system", record["messages"][0]["content"]),
        message("user", record["messages"][1]["content"], original),
    ]
    committed_image = original.copy()
    committed_pixel = None
    committed_coor = None
    candidate_image = None
    candidate_pixel = None
    pending_action = None
    image_by_coor: dict[tuple[int, int], Image.Image] = {}
    pixel_by_coor: dict[tuple[int, int], list[float]] = {}
    answer = None
    failure = None
    logs = []

    for turn in range(args.max_turns):
        output = generator(messages)
        answer_match = ANSWER_RE.search(output)
        if answer_match:
            answer = answer_match.group(1).strip()
            logs.append({"turn": turn, "output": output, "final": True})
            break

        if turn:
            status = callback_status(output)
            if status is None:
                failure = f"missing callback status at turn {turn}"
                logs.append({"turn": turn, "output": output, "parse_failure": failure})
                break
            if status:
                dr, dc = DIRECTIONS[pending_action]
                committed_coor = [committed_coor[0] + dr, committed_coor[1] + dc]
                committed_pixel = candidate_pixel
                committed_image = candidate_image
                image_by_coor[tuple(committed_coor)] = committed_image.copy()
                pixel_by_coor[tuple(committed_coor)] = list(committed_pixel)

        action_match = ACTION_RE.search(output)
        if not action_match:
            failure = f"missing action at turn {turn}"
            logs.append({"turn": turn, "output": output, "parse_failure": failure})
            break
        pending_action = action_match.group(1).lower()

        if turn == 0:
            size_match = SIZE_RE.search(output)
            level = int(size_match.group(1)) if size_match else int(record["level"])
            grid_match = GRID_RE.search(output)
            if not grid_match:
                failure = "missing grid size at turn 0"
                logs.append({"turn": turn, "output": output, "parse_failure": failure})
                break
            grid = float(grid_match.group(1))
            try:
                committed_pixel = parse_pair(PIXEL_RE, output, float)
                committed_coor = parse_pair(COOR_RE, output, int)
            except ValueError as error:
                failure = str(error)
                logs.append({"turn": turn, "output": output, "parse_failure": failure})
                break
            image_by_coor[tuple(committed_coor)] = committed_image.copy()
            pixel_by_coor[tuple(committed_coor)] = list(committed_pixel)
            marker = MARKER_SIZE.get(level, 3)

        dr, dc = DIRECTIONS[pending_action]
        proposed_coor = (committed_coor[0] + dr, committed_coor[1] + dc)
        if proposed_coor in image_by_coor:
            candidate_image = image_by_coor[proposed_coor].copy()
            candidate_pixel = list(pixel_by_coor[proposed_coor])
        else:
            candidate_image, candidate_pixel = draw_candidate(
                committed_image, committed_pixel, pending_action, grid, marker
            )
        logs.append({"turn": turn, "output": output, "action": pending_action})
        messages.append(message("assistant", output))
        messages.append(message("user", "The generated image is <image>.", candidate_image.copy()))

    return {
        "index": index,
        "level": int(record["level"]),
        "success": score(record, answer),
        "answer": answer,
        "parse_failure": failure,
        "turns": len(logs),
        "logs": logs,
    }


def evaluate_other_task(args: argparse.Namespace, generator: Generator, record: dict, index: int) -> dict:
    original = Image.open(record["images"][0]).convert("RGB")
    messages = [
        message("system", record["messages"][0]["content"]),
        message("user", record["messages"][1]["content"], original),
    ]
    image = original
    pixel = None
    grid = None
    size = None
    color = (255, 0, 0)
    answer = None
    failure = None
    logs = []
    for turn in range(args.max_turns):
        output = generator(messages)
        answer_match = ANSWER_RE.search(output)
        if answer_match:
            answer = answer_match.group(1).strip()
            logs.append({"turn": turn, "output": output, "final": True})
            break
        try:
            if turn == 0:
                size_match = SIZE_RE.search(output)
                grid_match = GRID_RE.search(output)
                if not size_match or not grid_match:
                    raise ValueError("missing size or grid pixel size at turn 0")
                size = int(size_match.group(1))
                grid = float(grid_match.group(1))
                if grid <= 0:
                    raise ValueError("grid pixel size must be positive")
                if args.task == "maze_navigation":
                    if size not in NAV_MARKER:
                        raise ValueError(f"unsupported navigation grid size: {size}")
                    pixel = parse_pair(PIXEL1_RE, output, float)
                else:
                    row, col = parse_pair(COOR_RE, output, int)
                    pixel = [(col + 0.5) * grid, (row + 0.5) * grid]
            if args.task == "maze_navigation":
                match = ACTION_RE.search(output)
                if not match:
                    raise ValueError(f"missing navigation action at turn {turn}")
                action = match.group(1).lower()
                image, pixel = draw_navigation(image, pixel, action, grid, size)
            else:
                actions = [value.strip().lower() for value in MINI_ACTION_RE.findall(output)]
                moves = [value for value in actions if value in {"go up", "go down", "go left", "go right"}]
                if len(moves) != 1:
                    raise ValueError(f"expected one Mini-BEHAVIOR movement at turn {turn}")
                if "pick up" in actions:
                    color = (0, 0, 255)
                action = moves[0]
                image, pixel = draw_mini(image, pixel, action, grid, color)
        except ValueError as error:
            failure = str(error)
            logs.append({"turn": turn, "output": output, "parse_failure": failure})
            break
        logs.append({"turn": turn, "output": output, "action": action})
        messages.append(message("assistant", output))
        messages.append(message("user", FEEDBACK, image.copy()))
    return {
        "index": index,
        "level": int(record["level"]),
        "success": answer is not None and answer.upper() == record["final_answer"].upper(),
        "answer": answer,
        "parse_failure": failure,
        "turns": len(logs),
        "logs": logs,
    }


def evaluate_vsp(args: argparse.Namespace, generator: Generator, record: dict, index: int) -> dict:
    image = Image.open(record["images"][0]).convert("RGB")
    committed_image = image
    messages = [
        message("system", record["messages"][0]["content"]),
        message("user", record["messages"][1]["content"], image),
    ]
    row = col = None
    answer = None
    failure = None
    logs = []
    size = int(record["level"])
    holes = {tuple(position) for position in record["hole"]}
    history = None
    for turn in range(args.max_turns):
        output = generator(messages)
        match = ANSWER_RE.search(output)
        if match:
            answer = match.group(1).strip()
            logs.append({"turn": turn, "output": output, "final": True})
            break
        try:
            if turn == 0:
                row, col = parse_pair(VSP_START_RE, output, int)
                if not (1 <= row <= size and 1 <= col <= size):
                    raise ValueError("starting coordinate outside VSP map")
                history = [(row, col, image)]
            match = ACTION_RE.search(output)
            if not match:
                raise ValueError(f"missing VSP action at turn {turn}")
            action = match.group(1).lower()
            dr, dc = DIRECTIONS[action]
            proposed = (row + dr, col + dc)
            inside = 1 <= proposed[0] <= size and 1 <= proposed[1] <= size
            image = draw_vsp(committed_image, row, col, action, green=inside)
            if inside and proposed not in holes:
                if len(history) > 1 and proposed == history[-2][:2]:
                    history.pop()
                    row, col, image = history[-1]
                else:
                    row, col = proposed
                    history.append((row, col, image))
                committed_image = image
        except ValueError as error:
            failure = str(error)
            logs.append({"turn": turn, "output": output, "parse_failure": failure})
            break
        logs.append({"turn": turn, "output": output, "action": action})
        messages.append(message("assistant", output))
        messages.append(message("user", VSP_FEEDBACK, image.copy()))
    return {
        "index": index,
        "level": size,
        "success": score_vsp(record, answer),
        "answer": answer,
        "parse_failure": failure,
        "turns": len(logs),
        "logs": logs,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen2.5-VL inference")
    records = [json.loads(line) for line in args.jsonl.open(encoding="utf-8") if line.strip()]
    end = min(args.end if args.end is not None else len(records), len(records))
    if not 0 <= args.start < end:
        raise ValueError("selected range must contain at least one record")
    if (args.shard_id is None) != (args.num_shards is None):
        raise ValueError("--shard-id and --num-shards must be specified together")
    if args.num_shards is not None:
        if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
            raise ValueError("shard must satisfy 0 <= shard-id < num-shards")
        # Striding distributes every maze size across all workers. The test JSONL
        # is ordered by size, so contiguous ranges leave the largest mazes on the
        # last workers and cause substantial tail latency.
        indices = range(args.start + args.shard_id, end, args.num_shards)
    else:
        indices = range(args.start, end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generator = Generator(args)
    totals = defaultdict(int)
    correct = defaultdict(int)
    with args.output.open("w", encoding="utf-8") as handle:
        for index in indices:
            evaluator = evaluate_maze_plan if args.task == "maze_plan" else evaluate_vsp if args.task == "vsp" else evaluate_other_task
            result = evaluator(args, generator, records[index], index)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            level = result["level"]
            totals[level] += 1
            correct[level] += int(result["success"])
            print(index, level, result["success"], result["parse_failure"], flush=True)
    print(json.dumps({str(level): correct[level] / totals[level] for level in sorted(totals)}, indent=2))


if __name__ == "__main__":
    main()
