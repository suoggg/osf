import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
REPO_ROOT = FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _as_posix(p: Path) -> str:
    return p.as_posix()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_box_xywh(rect, w, h):
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return None
    x, y, bw, bh = rect
    if not all(isinstance(v, (int, float)) for v in (x, y, bw, bh)):
        return None
    x = max(float(x), 0.0)
    y = max(float(y), 0.0)
    bw = max(float(bw), 0.0)
    bh = max(float(bh), 0.0)
    if x >= w or y >= h:
        return None
    bw = min(bw, float(w) - x)
    bh = min(bh, float(h) - y)
    if bw <= 0.0 or bh <= 0.0:
        return None
    return x, y, bw, bh


def _to_yolo_xywh(x, y, bw, bh, w, h):
    xc = (x + bw / 2.0) / float(w)
    yc = (y + bh / 2.0) / float(h)
    return xc, yc, bw / float(w), bh / float(h)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_one_label(path: Path):
    if not path.exists():
        return None
    s = path.read_text(encoding="utf-8").strip()
    if not s:
        return None
    parts = s.split()
    if len(parts) != 5:
        return None
    try:
        cls = int(float(parts[0]))
        vals = [float(x) for x in parts[1:]]
    except Exception:
        return None
    return cls, *vals


def _make_cube_label(t1, t2, t3):
    cls3, xc3, yc3, w3, h3 = t3
    zeros = (0.0, 0.0, 0.0, 0.0)
    if t1 is None:
        box1 = zeros
    else:
        _, xc1, yc1, w1, h1 = t1
        box1 = (xc1, yc1, w1, h1)
    if t2 is None:
        box2 = zeros
    else:
        _, xc2, yc2, w2, h2 = t2
        box2 = (xc2, yc2, w2, h2)
    box3 = (xc3, yc3, w3, h3)
    vals = (cls3, *box1, *box2, *box3)
    return " ".join([str(vals[0])] + [f"{v:.8f}" for v in vals[1:]]) + "\n"


def _iter_seq_dirs(src_root: Path, split: str):
    base = src_root / split
    if not base.exists():
        print(f"警告: 找不到目录 {base.absolute()}，请检查 --src-root 是否正确")
        return []
    dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    print(f"成功: 在 {split} 中找到 {len(dirs)} 个序列")
    return dirs


def process_one_sequence(
    seq_dir: Path,
    split: str,
    images_root: Path,
    labels_single_root: Path,
    labels_root: Path,
    modality: str,
    cube: bool,
    frame_stride: int,
    max_frames: int | None,
):
    # 增强兼容性：尝试多种可能的视频和 JSON 文件名
    video_names = [
        f"{modality}.mp4", f"{modality.upper()}.mp4", 
        "infrared.mp4" if modality == "ir" else "visible.mp4",
        "IR.mp4" if modality == "ir" else "VIS.mp4",
        "video.mp4", "v.mp4"
    ]
    json_names = [
        f"{modality}.json", f"{modality.upper()}.json", 
        "infrared.json" if modality == "ir" else "visible.json",
        f"{modality.upper()}_label.json", "label.json"
    ]

    video_path = None
    for vn in video_names:
        if (seq_dir / vn).exists():
            video_path = seq_dir / vn
            break

    json_path = None
    for jn in json_names:
        if (seq_dir / jn).exists():
            json_path = seq_dir / jn
            break

    if not video_path:
        print(f"  [跳过] {seq_dir.name}: 找不到视频文件 (尝试了: {video_names})")
        return []
    if not json_path:
        print(f"  [跳过] {seq_dir.name}: 找不到 JSON 标签 (尝试了: {json_names})")
        return []

    try:
        ann = json.load(open(json_path, "r", encoding="utf-8"))
    except Exception as e:
        print(f"  [跳过] {seq_dir.name}: 无法读取 JSON ({e})")
        return []

    exist = ann.get("exist")
    gt = ann.get("gt_rect")
    if not isinstance(exist, list) or not isinstance(gt, list):
        print(f"  [跳过] {seq_dir.name}: JSON 格式不正确 (缺少 exist 或 gt_rect)")
        return []
    n = min(len(exist), len(gt))
    if max_frames and max_frames > 0:
        n = min(n, max_frames)

    out_img_dir = images_root / split / seq_dir.name
    out_single_dir = labels_single_root / split / seq_dir.name
    out_label_dir = labels_root / split / seq_dir.name
    _ensure_dir(out_img_dir)
    _ensure_dir(out_single_dir)
    _ensure_dir(out_label_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [跳过] {seq_dir.name}: 无法打开视频文件 {video_path}")
        return []

    frame_id = 0
    seq_img_paths = []
    while frame_id < n:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_stride > 1 and frame_id % frame_stride != 0:
            continue

        h, w = frame.shape[:2]
        name = f"{frame_id:06d}"
        img_path = out_img_dir / f"{name}.jpg"
        single_lb_path = out_single_dir / f"{name}.txt"
        final_lb_path = out_label_dir / f"{name}.txt"

        cv2.imwrite(str(img_path), frame)
        seq_img_paths.append(str(img_path.resolve()))

        e = exist[frame_id - 1]
        rect = gt[frame_id - 1]

        if e in (1, True):
            box = _safe_box_xywh(rect, w, h)
            if box is None:
                _write_text(single_lb_path, "")
            else:
                x, y, bw, bh = box
                xc, yc, ww, hh = _to_yolo_xywh(x, y, bw, bh, w, h)
                _write_text(single_lb_path, f"0 {xc:.8f} {yc:.8f} {ww:.8f} {hh:.8f}\n")
        else:
            _write_text(single_lb_path, "")

        if not cube:
            _write_text(final_lb_path, single_lb_path.read_text(encoding="utf-8"))
        else:
            t3 = _read_one_label(single_lb_path)
            if t3 is None:
                _write_text(final_lb_path, "")
            else:
                t3_idx = frame_id
                t1_idx = max(t3_idx - 4, 1)
                t2_idx = max(t3_idx - 2, 1)
                t1_path = out_single_dir / f"{t1_idx:06d}.txt"
                t2_path = out_single_dir / f"{t2_idx:06d}.txt"
                t1 = _read_one_label(t1_path)
                t2 = _read_one_label(t2_path)
                t2 = t2 or t3
                t1 = t1 or t2
                _write_text(final_lb_path, _make_cube_label(t1, t2, t3))

    cap.release()
    return seq_img_paths


def prepare_dataset(
    src_root: Path,
    out_root: Path,
    modality: str,
    cube: bool,
    frame_stride: int,
    max_seqs: int | None,
    max_frames: int | None,
    num_workers: int = 4,
):
    splits = ["train", "val", "test"]
    images_root = out_root / "images"
    labels_single_root = out_root / "labels_single"
    labels_root = out_root / "labels"
    _ensure_dir(images_root)
    _ensure_dir(labels_single_root)
    _ensure_dir(labels_root)

    print(f"--- 启动多线程数据预处理 ---")
    print(f"源目录 (src_root): {src_root.absolute()}")
    print(f"输出目录 (out_root): {out_root.absolute()}")
    print(f"模态 (modality): {modality}")
    print(f"时序模式 (cube): {cube}")
    print(f"线程数 (workers): {num_workers}")
    print(f"----------------------")

    all_img_paths = {s: [] for s in splits}

    for split in splits:
        seq_dirs = _iter_seq_dirs(src_root, split)
        if max_seqs and max_seqs > 0:
            seq_dirs = seq_dirs[:max_seqs]

        if not seq_dirs:
            continue

        print(f"正在处理 {split} 分组...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    process_one_sequence,
                    sd, split, images_root, labels_single_root, labels_root,
                    modality, cube, frame_stride, max_frames
                )
                for sd in seq_dirs
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=split):
                res = future.result()
                all_img_paths[split].extend(res)

    for split in splits:
        list_file = out_root / f"{split}.txt"
        _write_text(list_file, "\n".join(all_img_paths[split]) + ("\n" if all_img_paths[split] else ""))

    y = {
        "path": _as_posix(out_root),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {0: "uav"},
    }
    yaml_path = out_root / f"antiuav_{modality}_{'cube' if cube else 'single'}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(y, open(yaml_path, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)

    print("\n--- 处理完成 ---")
    print("out_root", out_root)
    print("yaml", yaml_path)
    print("train_images", len(all_img_paths["train"]))
    print("val_images", len(all_img_paths["val"]))
    print("test_images", len(all_img_paths["test"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=str, default=r"d:\files\dataset_all\ANTIUAV\ANTIUAV_drone_image_IR_aligned")
    ap.add_argument("--out-root", type=str, default=r"d:\files\dataset_all\ANTIUAV\ANTIUAV_drone_image_IR_aligned\yolo_camera")
    ap.add_argument("--modality", type=str, default="visible")
    ap.add_argument("--cube", action="store_true")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    max_seqs = None if args.max_seqs <= 0 else args.max_seqs
    max_frames = None if args.max_frames <= 0 else args.max_frames

    prepare_dataset(
        src_root=src_root,
        out_root=out_root,
        modality=args.modality,
        cube=bool(args.cube),
        frame_stride=max(1, int(args.frame_stride)),
        max_seqs=max_seqs,
        max_frames=max_frames,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
