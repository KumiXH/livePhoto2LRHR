# 黄金案例 YAML 当前写死绝对路径，跨电脑无法一条命令直接运行

现在这套仓库在“别的电脑上”运行黄金案例，最稳的方式是这样。

先安装：

```bash
python -m pip install -e .[dev]
```

如果你主要要跑 HEIC，这条就够了，因为黄金案例当前是 `opencv + classical align + classical color`，不依赖额外 `pth/ckpt`。

然后你要先改一份 YAML。

原因很重要：现在仓库里的黄金案例 YAML 里，`input_dir` 和 `output_dir` 还是写死的绝对路径，不能直接跨电脑原样运行。

你至少要改这两个字段：

```yaml
data:
  input_dir: D:/你的数据目录
  output_dir: D:/你的输出目录
```

如果你跑 HEIC 数据，建议用我刚做的这两份：

- `real_smoke/heic_livephoto_golden/adaptive_3d_lut_heic.yaml`
- `real_smoke/heic_livephoto_golden/diffusion_harmonization_heic.yaml`

改完之后，命令就是：

```bash
livephoto2lrhr --config real_smoke/heic_livephoto_golden/adaptive_3d_lut_heic.yaml
```

或者：

```bash
livephoto2lrhr --config real_smoke/heic_livephoto_golden/diffusion_harmonization_heic.yaml
```

如果你的环境里 `livephoto2lrhr` 命令还没生效，就用这个：

```bash
python -m livephoto2lrhr.cli --config real_smoke/heic_livephoto_golden/adaptive_3d_lut_heic.yaml
```

或者不装包，临时直接跑源码：

Windows PowerShell:

```powershell
$env:PYTHONPATH='src'
python -m livephoto2lrhr.cli --config real_smoke/heic_livephoto_golden/adaptive_3d_lut_heic.yaml
```

你如果想要“下载仓库后完全不改 YAML，直接一条命令跑”，那当前版本还不够，因为还不支持命令行覆盖 `input_dir/output_dir`。这块建议后续补上。
