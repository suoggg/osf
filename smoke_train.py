# 操作前请先激活虚拟环境：conda activate osformer
# 测试时batch=32，正常运行
# 训练日志警告已全部修复 (All Warnings Resolved)
import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
REPO_ROOT = FILE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=r"d:\files\dataset_all\ANTIUAV\ANTIUAV_drone_image_IR_aligned\yolo_camera_test\antiuav_visible_cube.yaml")
    ap.add_argument("--model", type=str, default="ultralytics/models/v8/yolov8s.yaml")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cube", action="store_true", default=True)
    ap.add_argument("--gray", action="store_true", default=True)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--verbose", action="store_true", default=False)
    args = ap.parse_args()

    data_path = str(Path(args.data))
    model = YOLO(args.model)
    model.train(
        data=data_path,
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=args.device,
        workers=int(args.workers),
        cube=bool(args.cube),
        gray=bool(args.gray),
        save=bool(args.save),
        val=bool(args.val),
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()
