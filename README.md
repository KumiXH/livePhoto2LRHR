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

## 安装

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

无论是 Windows 还是 Linux，YAML 里都建议使用正斜杠路径，例如：

```text
D:/datasets/input
/data/datasets/input
```

## 输入约定

图片和视频通过“相同相对 stem 名”进行配对，例如：

```text
input/trip/IMG_0001.jpg
input/trip/IMG_0001.mp4
```

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
```

当前可用的调色算法：

```text
identity_color_match
mean_std_lab
```

当前最终导出（final export）支持的质量门槛包括：

```text
align_status
flow_status
align_confidence
source-to-HR MAE
file existence
overwrite safety
```

## 文档入口

建议重点阅读 [docs/operation_manual.md](docs/operation_manual.md)，其中包含：

- Windows / Linux 环境准备
- 推荐的阶段化运行方式
- 报告与最终导出说明
- 质量阈值调节建议
- 更强质量门槛的后续方向
- 更强对齐与调色算法的演进方向
