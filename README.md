# livePhoto2LRHR

`livePhoto2LRHR` 用于从 Live Photo 风格的图片和视频对中，构建超分训练所需的 `LR/HR` 配对数据集。

整个工程采用 `YAML` 驱动，按可替换阶段（replaceable stages）组织：

```text
image + mp4
  -> frame selection（抽帧匹配）
  -> alignment（对齐）
  -> optional color matching（可选调色）
  -> quality report（质量报告）
  -> final dataset export（最终数据集导出）
```

更完整的生产流程请看 [docs/operation_manual.md](docs/operation_manual.md)。

文档入口：

- 操作手册：[docs/operation_manual.md](/D:/repository/livePhoto2LRHR/docs/operation_manual.md)
- 日志 / CSV / metadata 说明：见手册中的“运行产物说明”章节

## 安装

如果你想直接走一份“默认按 GPU 安装”的 requirements 文件：

```bash
python -m pip install -r requirements.txt
```

这份 [requirements.txt](/D:/repository/livePhoto2LRHR/requirements.txt) 当前默认使用 PyTorch 官方 `CUDA 12.1` 轮子，适合你的这台 `CUDA Version: 12.1` 机器直接安装。

如果你之后换到别的机器，CUDA 版本不是 `12.1`，可以再把 `requirements.txt` 里的 `cu121` 改成 PyTorch 官方支持的其他 CUDA 轮子版本，例如 `cu118`，并同步调整 `torch / torchvision` 版本号。

安装后可快速验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出最后一行是 `True`，说明 PyTorch 已经识别到 GPU。

如果要使用 `DINOv2` 抽帧匹配：

```bash
python -m pip install -e .[dev,dinov2]
```

如果只想使用 CPU / OpenCV baseline：

```bash
python -m pip install -e .[dev]
```

运行测试：

```bash
python -m pytest -q
```

## 快速开始

推荐从完整模板配置开始：

```text
configs/full_pipeline_template.yaml
```

先把输入输出路径改成你自己的绝对路径：

```yaml
data:
  input_dir: /path/to/input
  output_dir: /path/to/output
```

然后执行：

```bash
livephoto2lrhr --config configs/full_pipeline_template.yaml
```

如果你想临时只跑某几个阶段，而不改 YAML，也可以直接在命令行覆盖：

```bash
livephoto2lrhr --config configs/full_pipeline_template.yaml --stages frame_select align
```

这样会只跑你指定的阶段列表，并把这次覆盖后的阶段列表写回 `run_summary.yaml` 里的 `config.pipeline.stages`。

无论是 Windows 还是 Linux，YAML 里都建议使用正斜杠路径，例如：

```text
D:/datasets/input
/data/datasets/input
```

另外现在也支持一个轻量运行时编排块：

```yaml
runtime:
  retry_failed_samples: false
  parallel:
    num_workers: 1
    gpu_ids: []
```

当它开启时，如果某个样本上次在阶段二或阶段三的 metadata 里被记录为 `failed`，下次运行时会优先重试这个失败样本，而不是因为旧输出文件存在就直接跳过。

如果你是在多卡服务器上跑，也可以先把并行 worker 和 GPU 列表写进去。例如 8 卡服务器：

```yaml
runtime:
  retry_failed_samples: true
  parallel:
    num_workers: 8
    gpu_ids: ["0", "1", "2", "3", "4", "5", "6", "7"]
```

当前第一版并行工程化已经支持：

- CLI 覆盖 `--num-workers`
- CLI 覆盖 `--gpu-ids`
- `run_summary.yaml` 中记录 worker 规划和 GPU 分配
- `frame_select` 场景下把“并行模式已启用”明确写入 summary
- `frame_select` 已经支持真实多进程分片执行

当前真实并行执行的范围先限定在 `frame_select`。也就是：

- 阶段一：已支持多 worker 真并发
- 阶段二：当前仍是主进程串行
- 阶段三：当前仍是主进程串行

这样做是为了先把最重、最独立的抽帧阶段稳定跑在多卡服务器上，再逐步把对齐和调色接入并发链。

命令行也可以直接覆盖：

```bash
livephoto2lrhr --config configs/full_pipeline_template.yaml --num-workers 8 --gpu-ids 0 1 2 3 4 5 6 7
```

## 输入约定

图片和视频通过“相同相对 stem 名”进行配对，例如：

```text
input/trip/IMG_0001.jpg
input/trip/IMG_0001.mp4
```

当前图片扩展默认支持：

```text
.jpg
.jpeg
.png
.heic
.heif
```

如果你要读取 `.heic / .heif`，项目当前通过 `pillow-heif` 接入 Pillow 解码支持。只要按当前安装方式安装依赖即可，不需要额外手工转码图片。

## 输出约定

工程会尽量保持 `LR` / `HR` 镜像目录结构一致，方便后续 compare 和训练：

```text
output/
  LR/
  HR/
  metadata/
  LR_aligned_flow/
  reports_flow/
  final_flow/
    LR/
    HR/
    manifest.csv
  run_summary.yaml
```

原始 `LR` 和 `HR` 不会被后续阶段覆盖。对齐、调色、报告、最终导出都会写入各自独立目录。

`run_summary.yaml` 里现在也会额外写出一组运行编排统计：

- `execution.retry_failed_samples`
- `execution.retried_failed_samples`
- `execution.resumed_from_existing_outputs`
- `execution.stage_timings_sec`
- `execution.total_runtime_sec`
- `execution.failed_samples_manifest`
- `execution.sample_status_yaml`
- `execution.sample_status_csv`
- `execution.parallel`

每次运行后，输出目录下还会额外生成一个：

```text
failed_samples.yaml
sample_status.yaml
sample_status.csv
```

里面会集中列出本轮失败样本，方便后续筛查、单独重跑和批处理。

如果你要按“每个样本”查看三阶段状态、开始结束时间、耗时、错误消息和异常堆栈，优先看：

- `sample_status.yaml`
- `sample_status.csv`

## 当前可用基线

当前可用的抽帧算法：

```text
dinov2_similarity
opencv_similarity
fake_selector
```

当前可用的对齐算法：

```text
identity_alignment
phase_correlation_translation
ecc_alignment
coarse_to_flow
global_ecc_homography
feature_match_transform
feature_match_homography
hybrid_feature_flow
mask_aware_alignment
```

当前可用的调色算法：

```text
identity_color_match
mean_std_lab
histogram_match_lab
retinex_color_match
masked_color_transfer
image_adaptive_3d_lut_color_match
low_frequency_joint_appearance_match
learned_retinex_color_match
mask_aware_harmonization_network
diffusion_harmonization
```

## 黄金案例

当前这批真实 smoke 中，作为“可直接复现最终结果”的黄金案例，建议优先看下面两套：

1. `image_adaptive_3d_lut_color_match`
   参考 YAML：`real_smoke/livephoto_color_five_backends/adaptive_3d_lut.yaml`
   真实输出目录：`real_smoke/livephoto_color_five_backends/output_adaptive_3d_lut`
   最终训练集目录：`real_smoke/livephoto_color_five_backends/output_adaptive_3d_lut/final_adaptive_3d_lut_half`
2. `diffusion_harmonization`
   参考 YAML：`real_smoke/livephoto_color_five_backends/diffusion_harmonization.yaml`
   真实输出目录：`real_smoke/livephoto_color_five_backends/output_diffusion_harmonization`
   最终训练集目录：`real_smoke/livephoto_color_five_backends/output_diffusion_harmonization/final_diffusion_harmonization_half`

这两套案例都已经在真实目录 `D:/SR数据集/livePhoto` 上跑通过，能产出最终可用于超分训练的 `LR/HR` 结果。

这两套黄金案例当前都不需要额外下载 `pth / ckpt` 模型权重。它们对应的抽帧、对齐、调色链路目前都是 OpenCV / NumPy / 规则算法路线，可直接离线复现。

直接复现命令：

```bash
livephoto2lrhr --config real_smoke/livephoto_color_five_backends/adaptive_3d_lut.yaml
livephoto2lrhr --config real_smoke/livephoto_color_five_backends/diffusion_harmonization.yaml
```

## 模型依赖说明

当前项目里，真正会涉及“额外模型文件 / 首次联网下载”的，主要是：

- `dinov2_similarity`
  需要 PyTorch 与 torchvision，并且首次运行会通过 `torch.hub` 拉取 DINOv2 权重缓存。

当前两个黄金案例：

- `image_adaptive_3d_lut_color_match`
- `diffusion_harmonization`

都不需要额外 `pth / ckpt`，也不会在运行时自动下载模型。

更完整的按阶段总表，请看：

- [docs/operation_manual.md](/D:/repository/livePhoto2LRHR/docs/operation_manual.md)

当前最终导出（final export）支持“训练用 LR 来源”和“质量筛选来源”分离，质量门槛包括：

```text
align_status
flow_status
align_confidence
gate source-to-HR MAE
gate source-to-HR PSNR
gate source-to-HR SSIM
gate source-to-HR dimension check
gate source-to-HR aspect-ratio check
gate source-to-HR border artifact score
mean_flow_magnitude
file existence
overwrite safety
```

这里的 quality gates 可以理解为“训练集质检门”：它不负责再生成新图，而是负责判断某条 `LR/HR` 样本是否足够适合进入最终训练集，并把拒绝原因记录到 `manifest.csv` 和 `quality_report.csv`。

如果你希望最终训练集里的 `LR` 既继承对齐/调色结果，又保持原始低清尺寸，现在也支持：

```text
final_lr_source = aligned / color_matched
final_lr_resize_mode = match_raw
```

对于 `match_raw / raw`，导出阶段现在会优先走“低清重放链”：

- 如果 metadata 里保存了可重放的对齐变换（例如 translation / ECC），会直接在原始 `raw LR` 网格上重放这些几何变换
- 如果 `final_lr_source = color_matched`，还会继续在低清网格上重放已保存的调色变换
- 只有在当前 metadata 不足以重放时，才会回退到“先得到大图代理结果，再缩回低清尺寸”的兼容逻辑

这条链路的目标是让最终导出的 `LR` 同时满足：

- 保持低清尺寸
- 继承阶段二 / 三的空间与颜色结果
- 尽量减少简单整图缩小时带来的“整体偏移感”

现在 `final_lr_resize_mode` 也支持配置式缩放倍率，方便你控制最终训练 `LR` 的尺寸策略：

```text
copy
raw
match_raw
1.0
0.75
0.5
```

其中：

- `raw` / `match_raw`：回到原始低清 LR 尺寸
- `1.0`：保持当前 HR 空间尺寸
- `0.75`：导出为 HR 空间的 75%
- `0.5`：导出为 HR 空间的 50%

当前默认推荐标准已经切到 `0.5`。也就是如果你不额外指定，最终导出的训练 `LR` 会默认采用“HR 空间 1/2 尺寸”。

经验上，如果你更在意肉眼观察时的空间对齐稳定性，而不是必须严格回到手机原始低清尺寸，`0.5` 往往是目前最稳的默认档位，通常也会比 `raw` 更适合第一版正式训练集。`raw / match_raw` 仍然适合做“真实低清训练输入”导出，但在大幅缩小时会更容易出现整体偏移的体感。

## 文档入口

建议重点阅读 [docs/operation_manual.md](docs/operation_manual.md)，其中包含：

- Windows / Linux 环境准备
- 推荐的阶段化运行方式
- 报告与最终导出说明
- 质量阈值调节建议
- 更强质量门槛的后续方向
- 更强对齐与调色算法的演进方向
