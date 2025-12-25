# ANTIUAV camera 数据准备脚本

## prepare.py

把 `ANTIUAV_drone_image_IR_aligned`（按序列存 mp4/json）转换为本项目可训练的 Ultralytics 数据集格式：

- `images/<split>/<seq>/<frame_id>.jpg`
- `labels/<split>/<seq>/<frame_id>.txt`
- `<split>.txt`：每行一个图片路径（供 `data.yaml` 使用）
- `antiuav_<modality>_<cube|single>.yaml`

默认使用 `visible`（camera）模态。

### 单帧标签（cube=False）

每个标签文件为 YOLO 5 列：

- `cls xc yc w h`

### cube 标签（cube=True）

每个标签文件为 13 列：

- `cls + t1(xc yc w h) + t2(xc yc w h) + t3(xc yc w h)`

其中帧间隔与本项目 dataloader 一致：

- `t1=t3-10, t2=t3-5, t3=t3`

## 使用示例

### 生成小样本（用于快速验证）

```bash
python datasets/antiuav_camera/prepare.py --modality visible --cube --max-seqs 1 --max-frames 30 --out-root d:/tmp/yolo_camera_smoke
```

### 全量生成（camera，可见光）

```bash
python datasets/antiuav_camera/prepare.py --modality visible --cube --out-root d:/files/dataset_all/ANTIUAV/ANTIUAV_drone_image_IR_aligned/yolo_camera
```

### 训练快速冒烟（Windows 下必须从文件启动，避免 `<stdin>` 多进程错误）

```bash
python datasets/antiuav_camera/smoke_train.py --data d:/tmp/yolo_camera_smoke/antiuav_visible_cube.yaml --cube --gray --batch 32 --epochs 1 --device 0 --workers 8 --val
```

