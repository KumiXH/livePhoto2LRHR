# livePhoto2LRHR 使用手册

本文档面向实际落地使用，说明如何用 `livePhoto2LRHR` 从 Live Photo 风格的图片/视频对中，构建超分训练所需的 `LR/HR` 数据集。

工程整体是一个可配置的 Python pipeline：

```text
image + mp4
  -> phase 1: frame selection（抽帧匹配）
  -> phase 2: alignment（对齐）
  -> optional phase 3: color matching（可选调色）
  -> quality report（质量报告）
  -> final dataset export（最终数据集导出）
```

后续阶段不会覆盖原始 `LR` / `HR`，每个阶段都写入自己的目录，因此可以安全试验不同算法和参数。

## 1. 阶段总览

| 阶段 | 目标 | 当前方法 / models | 推荐默认值 | 当前建议 |
| --- | --- | --- | --- | --- |
| Phase 1: Frame Selection | 从 MP4 中找出与静态图最接近的一帧 | `dinov2_similarity`, `opencv_similarity` | `dinov2_similarity` | 真正生成数据集时建议优先用 DINOv2。OpenCV 更适合 CPU smoke 或快速验证。 |
| Phase 2: Alignment | 让 LR 在几何上尽量对齐 HR | `identity_alignment`, `phase_correlation_translation`, `ecc_alignment`, `coarse_to_flow` | `coarse_to_flow` + `phase_correlation_translation` + DIS flow | 这是当前最强 baseline。`identity_alignment` 适合作为 fallback 和调试基线。 |
| Phase 3: Color Matching | 缩小 LR 与 HR 的亮度 / 颜色差距 | `identity_color_match`, `mean_std_lab` | `mean_std_lab` 仅供实验 | 目前建议保持可选，不要默认认为调色后的 LR 一定更适合作为最终训练输入。 |

### 1.1 Phase 1：抽帧匹配

- `dinov2_similarity`
  以 DINOv2 特征相似度为主信号，是当前推荐的生产基线。
- `opencv_similarity`
  轻量级 OpenCV baseline，更适合快速 CPU 验证。

### 1.2 Phase 2：对齐

- `identity_alignment`
  不做几何变换，主要用于阶段验证、fallback 和调试。
- `phase_correlation_translation`
  估计全局平移量（x/y translation）。
- `ecc_alignment`
  用 OpenCV ECC 做 `translation / euclidean / affine / homography` 风格配准。
- `coarse_to_flow`
  先做 coarse align，再在“误差真的变小”的前提下接受 dense flow refinement。

### 1.3 Phase 3：调色

- `identity_color_match`
  不做颜色变化，仅用于基线对照。
- `mean_std_lab`
  在 `LAB` 色彩空间下匹配均值/方差。对部分样本有帮助，对部分样本也可能变差，所以目前仍然视为实验性 baseline。

### 1.4 后续增强方向

- 更强的质量门槛（quality gates）
  例如 `SSIM`、`PSNR`、边缘 / 裁剪伪影检测、尺寸检查、flow 异常值检测。
- 更强的对齐后端（alignment backends）
  例如 `LoFTR`、`LightGlue`、`RAFT`、`GMFlow`、`SAM` 风格 mask、fusion controller。
- 更强的调色后端（color backends）
  例如 histogram matching、masked color transfer、LUT fitting、Retinex、轻量神经网络调色。

## 2. 环境准备

建议使用 Python 3.10 及以上。

### 2.1 CPU / OpenCV baseline

```bash
python -m pip install -e .[dev]
```

### 2.2 DINOv2 抽帧匹配

```bash
python -m pip install -e .[dev,dinov2]
```

### 2.3 运行测试

```bash
python -m pytest -q
```

在 Linux 上也使用同样的命令即可。如果你要使用 CUDA，请先安装与你 CUDA 驱动匹配的 PyTorch，再安装本项目。

## 3. 输入目录组织

把图片和视频放在同一个输入目录下，通过“相同相对 stem 名”进行匹配，例如：

```text
input/
  trip_a/
    IMG_0001.jpg
    IMG_0001.mp4
    IMG_0002.jpg
    IMG_0002.mp4
```

工程会生成镜像结构输出：

```text
output/
  LR/
  HR/
  metadata/
  run_summary.yaml
```

如果输入是 `input/trip_a/IMG_0001.jpg`，那么 HR 输出会是 `output/HR/trip_a/IMG_0001.png`。  
同理，LR 也会保持一致的目录层级。

## 4. 配置文件

推荐从这个模板开始：

```text
configs/full_pipeline_template.yaml
```

把它复制成你自己的配置，然后修改：

```yaml
data:
  input_dir: /path/to/input
  output_dir: /path/to/output
```

### 4.1 Windows 路径写法

建议用正斜杠：

```yaml
input_dir: D:/datasets/livephoto/input
output_dir: D:/datasets/livephoto/output
```

### 4.2 Linux 路径写法

Linux 直接使用正常绝对路径：

```yaml
input_dir: /data/livephoto/input
output_dir: /data/livephoto/output
```

除非是专门的 smoke-test 配置，否则尽量不要把机器私有路径提交到仓库里。

## 5. 推荐首次运行方式

对于一个新数据集，建议先直接跑完整 baseline pipeline：

```bash
livephoto2lrhr --config configs/full_pipeline_template.yaml
```

如果当前环境还没有 editable install，可以先执行：

```bash
python -m pip install -e .[dev,dinov2]
livephoto2lrhr --config configs/full_pipeline_template.yaml
```

默认推荐运行：

```yaml
pipeline:
  stages:
    - frame_select
    - align
```

如果配置里同时开启了 `report.enabled: true` 和 `export.enabled: true`，则还会自动生成报告并导出最终训练数据集。

## 6. Phase 1：抽帧匹配

Phase 1 会从每个 MP4 中选出最接近对应静态图的一帧作为 LR。

推荐默认配置：

```yaml
frame_select:
  algorithm: dinov2_similarity
  device: auto
  sample_fps: 15
  top_k: 5
  batch_size: 16
  resize_short_side: 518
```

输出：

```text
output/
  LR/
  HR/
  metadata/
```

其中：

- `LR/` 中只保存最终选中的最佳帧
- `top_k` 只记录在 metadata 中，不会写进 `LR/`

如果只是想做 CPU 快速 smoke，可以用：

```yaml
frame_select:
  algorithm: opencv_similarity
```

但如果是真实数据集生产，建议优先使用 `dinov2_similarity`。

## 7. Phase 2：对齐

Phase 2 用来让 LR 在几何上更接近 HR。

推荐 baseline：

```yaml
align:
  enabled: true
  algorithm: coarse_to_flow
  output_folder: LR_aligned_flow
  confidence_threshold: 0.3
  fallback_algorithm: identity_alignment
  on_failure: keep_original
  coarse_algorithm: phase_correlation_translation
  optical_flow:
    enabled: true
    algorithm: dis
```

输出：

```text
output/
  LR_aligned_flow/
  artifacts/
  metadata/
```

原始 `LR/` 和 `HR/` 不会被修改。

当前可用对齐算法：

```text
identity_alignment
phase_correlation_translation
ecc_alignment
coarse_to_flow
```

`coarse_to_flow` 的行为是：

1. 先运行 coarse aligner
2. 再尝试 dense flow refinement
3. 只有在实测误差变小时，才接受 flow 结果
4. 如果 flow 后更差，则保留 coarse 结果

## 8. Phase 3：调色

Phase 3 是可选项。它更适合实验，不建议默认直接拿它作为最终训练输入。

```yaml
color_match:
  enabled: true
  algorithm: mean_std_lab
  input_folder: auto
  output_folder: LR_color_matched
```

`input_folder: auto` 的规则：

- 优先读 `LR_aligned`
- 如果没有，则回退到 `LR`

如果你的对齐输出目录是 `LR_aligned_flow`，建议显式写：

```yaml
color_match:
  input_folder: LR_aligned_flow
```

当前可用调色器：

```text
identity_color_match
mean_std_lab
```

`mean_std_lab` 在一些样本上会改善亮度/颜色，但在另一些样本上也可能变差，因此建议继续把它当作实验基线使用。

## 9. 质量报告

开启：

```yaml
report:
  enabled: true
  output_folder: reports_flow
  aligned_folder: LR_aligned_flow
  color_matched_folder: LR_color_matched
```

输出：

```text
output/reports_flow/
  quality_report.csv
  preview_contact_sheet.jpg
```

关键 CSV 字段包括：

```text
sample_id
frame_index
timestamp_sec
frame_select_score
align_status
align_confidence
flow_status
lr_to_hr_mae
aligned_to_hr_mae
color_matched_to_hr_mae
lr_path
aligned_path
color_matched_path
hr_path
```

报告故意保持为普通 CSV，方便你自己接 viewer、dashboard 或人工审核工具。

## 10. 最终数据集导出

最终导出（final export）会读取质量报告，并把通过门槛的样本复制到最终训练数据集中：

```yaml
export:
  enabled: true
  input_report: reports_flow/quality_report.csv
  output_folder: final_flow
  lr_source: aligned
  min_align_confidence: 0.3
  require_align_status: success
  require_flow_status: accepted
  max_source_to_hr_mae: 30.0
```

输出：

```text
output/final_flow/
  LR/
  HR/
  manifest.csv
```

`manifest.csv` 会记录每个样本是 accepted 还是 rejected，以及拒绝原因，例如：

```text
accepted
align_status_mismatch
align_confidence_below_min
flow_status_mismatch
missing_lr_source
missing_hr
source_to_hr_mae_above_max
destination_exists
```

推荐的首次导出策略：

```yaml
lr_source: aligned
require_align_status: success
require_flow_status: accepted
max_source_to_hr_mae: 30.0
```

之后可以根据数据分布再逐步调 `max_source_to_hr_mae`。阈值越低，样本越干净，但通过数也会越少。

## 11. 分阶段单独运行

你也可以把各阶段拆开单独跑。

### 11.1 只跑 Phase 1

```yaml
pipeline:
  stages:
    - frame_select
```

### 11.2 只跑对齐

前提是 Phase 1 已经产出完成：

```yaml
pipeline:
  stages:
    - align
```

### 11.3 只跑最终导出

前提是质量报告已经存在：

```yaml
pipeline:
  stages: []

report:
  enabled: false

export:
  enabled: true
```

这在调 export 阈值时非常有用，因为你不需要每次都重新跑 DINOv2 或 optical flow。

## 12. Linux 使用说明

YAML 中建议统一使用正斜杠绝对路径。

如果 Linux 环境比较精简，可能需要先补一些 OpenCV / Pillow 相关系统库：

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

对于 headless server，本项目通常直接用 `opencv-python` 就够了，因为不会弹 GUI 窗口。  
如果你的环境和 GUI 库有冲突，可以在环境里改用 `opencv-python-headless`。

如果你要用 GPU 版 DINOv2，建议先检查 PyTorch：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

然后在配置里使用：

```yaml
frame_select:
  device: cuda
```

## 13. 常见问题

### 13.1 没有发现任何配对

请确认图片和视频的相对 stem 名一致，例如：

```text
input/a/IMG_0001.jpg
input/a/IMG_0001.mp4
```

### 13.2 DINOv2 首次运行失败

当前 selector 仍然依赖 `torch.hub`。  
请确保机器有网络访问能力，或者本地已经存在可用缓存。

### 13.3 对齐输出缺失

请先确认 `output/LR` 和 `output/HR` 已存在，因为对齐阶段依赖 Phase 1 的输出。

### 13.4 导出时拒绝样本过多

建议先打开 `manifest.csv`，按 `reason` 分组，再针对性调整：

- `max_source_to_hr_mae`
- `min_align_confidence`
- `require_flow_status`

### 13.5 中文路径在终端里显示乱码

优先使用 UTF-8 终端，并在 YAML 中使用正斜杠路径。  
项目内部使用的是 Python `pathlib`，本身支持 Unicode 路径，但终端显示仍然可能因为编码配置出问题。

## 14. 当前工程状态

当前项目已经具备一个可用 baseline：

```text
Phase 1: DINOv2 / OpenCV 抽帧匹配
Phase 2: identity / phase correlation / ECC / coarse-to-flow 对齐
Phase 3: identity / mean-std LAB 调色
Report: CSV + contact sheet
Export: 质量门槛驱动的最终 LR/HR 导出
```

整个 baseline 的核心目标是“可替换、可配置、可逐步增强”。  
算法都通过 registry 管理，pipeline 行为由 YAML 驱动。

## 15. 后续更强质量门槛

当前 export gate 已支持：

```text
align_status
flow_status
align_confidence
source-to-HR MAE
file existence
destination overwrite safety
```

下一步比较值得补的质量指标包括：

```text
SSIM
PSNR
edge similarity
crop / border artifact score
dimension and aspect-ratio checks
flow magnitude outlier checks
local patch error percentiles
face / foreground weighted error
```

推荐扩展方式：

```text
reports/quality.py
  增加新的 metric 列

export/dataset.py
  增加对应可选阈值

configs/*.yaml
  暴露阈值开关
```

建议保持所有门槛都是 optional，因为不同数据集适合的阈值并不一样。

## 16. 更强对齐 roadmap

当前对齐 baseline 是可用的，但不是终局。

推荐后续优先考虑的对齐增强方向：

```text
LoFTR / LightGlue feature matching
RAFT / GMFlow optical flow
SAM-style foreground / object masks
homography + local flow hybrid
fusion controller for per-sample routing
```

但无论换成什么高级模型，都建议继续保持同一 contract：

```text
LR image + HR image + metadata + config -> AlignResult
```

每个高级 aligner 仍应返回：

```text
aligned_lr_rgb
confidence
status
transforms
artifacts
diagnostics
```

不要在 runner 里为高级模型写特殊分支，直接通过 alignment registry 接入即可。

## 17. 更强调色 roadmap

当前 `mean_std_lab` 只是 baseline。

推荐后续增强方向：

```text
histogram matching
masked foreground / background color matching
Retinex-style illumination correction
3D LUT fitting
small neural color transfer model
exposure / white-balance regression
```

在更强质量门槛建立前，调色仍应保持 optional，不建议默认强制进入最终导出链路。

## 18. 推荐生产检查清单

在把一批生成结果认定为“可训练数据集”之前，建议至少完成：

```text
运行全量测试
先在小子集上跑抽帧
检查 metadata 中 top-k 候选
运行对齐并生成报告
查看报告指标分布
用保守阈值导出 final dataset
检查 manifest 中的拒绝原因
人工抽样 accepted / rejected
冻结最终使用的 config
记录 git commit 和 run_summary.yaml
```

最重要的可复现文件包括：

```text
config YAML
run_summary.yaml
metadata/
reports*/quality_report.csv
final*/manifest.csv
git commit SHA
```
