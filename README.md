# 离线物体识别与语音助手

使用摄像头录入身边的物体，通过 ResNet18 特征检索识别已录入的物体，并支持中文语音和文字交互。浏览器界面同时显示摄像头画面、深度热力图和物体库。

视觉推理、特征检索和语音识别在本地运行，支持 Windows + CPU。默认语音播报使用 Edge TTS，**需要联网**；如需完全离线，可配置本地 GPT-SoVITS 或关闭语音播报。模型首次准备也需要联网，除非已有模型文件。

![演示：识别录入的物体](docs/screenshots/demo1.png)

## 功能与原理

- **录入**：通过语音或文字命令为当前物体命名，采集多张图片，提取并保存特征。录入不会重新训练模型。
- **识别**：Depth Anything V2 根据深度生成候选区域；ResNet18 提取 512 维特征，与物体库进行余弦相似度检索。Web 摄像头识别还会检查最佳结果与其他物体的分数差距。
- **拒识**：相似度未达到阈值时，Web 界面返回“未知物体”；相似度接近的多个已录入物体也可能返回“不太确定”。
- **倾斜处理**：匹配不确定时，尝试旋转画面；对有明显长轴的候选区域，还会依据轮廓校正倾斜角度。
- **物体库管理**：浏览已录入物体、查看样本、重命名或删除。语音与文字还支持识别、录入、查找、列出和帮助等指令。
- **语音交互**：Silero VAD + Paraformer 中文语音识别；语音合成可选 Edge TTS、本地 GPT-SoVITS 或测试模式。

ResNet18 使用 ImageNet 预训练权重。相似度是特征空间中的匹配分数，**不是识别正确的概率**；它会受到录入样本、视角、裁剪、光照和物体库内容影响。深度模型负责定位，物体名称来自特征库检索。

## 快速开始

以下命令在项目根目录的 PowerShell 中运行。建议使用 Python 3.10 或 3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install flask
```

`requirements.txt` 当前未列出 Web 界面使用的 Flask，因此需要额外安装。运行视觉模型使用 ONNX Runtime CPU，不需要 CUDA、OpenVINO 或 PyTorch。

### 准备模型

如果模型文件尚未准备，按需运行：

```powershell
# 导出 ResNet18 ONNX；此步骤需要安装 torch、torchvision 和 onnx
python tools/prepare_models.py --resnet18

# 下载深度模型和中文语音识别 / VAD 模型
python tools/prepare_models.py --depth --asr
```

模型准备脚本可能因网络或缺少导出依赖而失败，请检查脚本输出及文件是否存在。ResNet18 导出所需的 PyTorch 只用于准备模型，运行时不需要。主要模型路径由配置文件指定：

```text
models/resnet18/resnet18.onnx
models/depth_anything_v2/depth_anything_v2_vits.onnx
models/sherpa-onnx-paraformer-zh-int8/
models/silero_vad/silero_vad.onnx
```

YOLO 和 MobileSAM 是可选组件，可分别使用 `python tools/prepare_models.py --yolo` 和 `--mobilesam` 准备；导出过程可能需要额外依赖。`config/demo.yaml` 默认关闭两者。`--all` 会尝试准备所有组件，而不只是 Web 演示所需的模型。

### 启动浏览器界面

```powershell
python web/server.py --config config/demo.yaml
```

打开 [http://127.0.0.1:5000](http://127.0.0.1:5000)。首次启动需要加载模型；按 `Ctrl+C` 停止服务。`config/demo.yaml` 是当前 Web 演示配置，默认摄像头索引为 `0`，可在文件中修改 `app.camera_index`。

按需使用以下选项：

```powershell
# 启动时不占用摄像头和麦克风，可稍后在界面中开启
python web/server.py --config config/demo.yaml --no-camera --no-mic

# 不启用语音输入
python web/server.py --config config/demo.yaml --no-voice

# 修改本机监听端口
python web/server.py --config config/demo.yaml --port 5001
```

默认只监听本机地址 `127.0.0.1`。摄像头、麦克风或语音模型不可用时，可使用网页下方的文字输入。

### 其他入口

```powershell
# OpenCV 桌面界面
python main.py --config config/demo.yaml --camera 0

# 不启用语音输入
python main.py --config config/demo.yaml --camera 0 --no-voice

# 批量录入及识别本地图片，不启动摄像头界面
python main.py --config config/demo.yaml --enroll 水杯 --images data/enroll/cup
python main.py --config config/demo.yaml --recognize data/query/test.png
```

`main.py` 默认使用 `config/default.yaml`；Web 示例显式使用 `config/demo.yaml`。现有 `run.bat` 和 `run.ps1` 含机器相关的 Python 路径，跨机器使用前需检查并修改。

## 使用方法

1. 打开页面，确认摄像头、模型和麦克风状态；也可以使用文字输入。
2. 将物体完整举到镜头前，尽量避免遮挡，说或输入“记住这个，这是手机”等带名称的命令。采样期间可缓慢调整角度并保持画面清晰。
3. 在右侧物体库点击缩略图查看录入样本。若物体不完整、被手或背景遮挡，应重新录入。
4. 举起已录入物体，说或输入“这是什么？”，也可点击页面快捷按钮。
5. 使用右上角的按钮暂停或恢复识别；右侧物体库可重命名、删除物体。

常见语音或文字命令包括“这是什么？”“记住这个，这是手机”“帮我找手机”“列出已录入列表”和“帮助”。具体命令由 `nlu/intent.py` 的规则解析。

## 相似度与配置

`config/demo.yaml` 中的相关设置：

| 配置 | 当前值 | 作用 |
| --- | --- | --- |
| `memory.center_features` | `true` | 在满足条件时减去特征库的共同成分，再计算相似度 |
| `memory.match_threshold_centered` | `0.5` | 中心化特征的识别阈值，低于它时拒识 |
| `memory.margin_threshold_centered` | `0.15` | Web 摄像头识别中，第一名与第二名所需的最小分数差 |
| `memory.match_threshold` | `0.7` | 未启用中心化时使用的阈值 |
| `memory.min_samples` / `max_samples` | `3` / `10` | 一次录入最少和最多保存的样本数 |
| `memory.sample_duration_sec` | `2.5` | 正常采样窗口，单位为秒 |
| `memory.sample_timeout_sec` | `15` | 样本不足时的采样总时限，单位为秒 |
| `depth.use_mobilesam` | `false` | 是否启用可选的 MobileSAM 细化 |

中心化至少需要两个不同的已录入物体；未满足条件时使用普通特征及对应阈值。新物体加入或删除后，中心化所用的均值可能变化，分数也可能变化。50% 是当前演示中的拒识设置，不是经过所有物体和场景验证的通用阈值：提高阈值可能减少误识别，也会增加把已录入物体判为未知的情况。

网页摄像头识别、桌面识别和本地图片批量识别的裁剪及判断流程并不完全相同；批量命令的未知结果还会在终端显示最相近候选及分数，便于调试。实际使用应分别验证。

## 数据与模型保存

| 路径 | 内容 |
| --- | --- |
| `data/visual_memory.db` | SQLite 物体名称、样本索引等元数据 |
| `data/resnet18_embeddings.npz` | 特征矩阵缓存 |
| `data/samples/<物体名>/` | 录入样本图片和对应的 `.npy` 特征 |
| `data/debug/` | 启用调试记录时保存的画面、候选区域和识别报告 |
| `models/` | 本地模型文件 |

保留数据库、特征和样本文件，可在重启后继续使用物体库。调试画面可能包含摄像头中的人物与环境，请按需管理；可在配置中设置 `debug.enabled: false` 关闭记录。

## 语音播报与离线使用

`config/demo.yaml` 默认 `tts.mode: edge`，需要网络。若只需本地视觉识别与语音输入，可在配置中设置 `tts.enabled: false`；`mock` 模式用于测试，不能替代正常语音播报。

需要离线语音播报时，可在独立环境中安装并启动本地 GPT-SoVITS 服务，再将配置中的 `tts.mode` 设为 `gpt_sovits`，检查 `tts.server_url`、参考音频和对应的 `prompt_text`。服务入口及参数见 `voice/tts_server.py` 和 `tools/start_tts_server.bat`；后者包含本机路径，使用前需要修改。

## 验证与排查

```powershell
# 不需要摄像头的自动化检查
python -m unittest discover -s tools -p 'test_*.py' -q

# 原有冒烟测试与可选性能基准
python tools/smoke_test.py
python benchmark.py --skip-asr --skip-tts
```

自动化检查包含采样时限、深度候选、纸巾颜色细节、旋转处理和拒识响应等回归场景；它们不能代替真实摄像头下的识别测试。定位失败时，可检查 `data/debug/` 的原图、候选裁剪和报告。`tools/eval_dataset.py` 等工具可用于数据集评测，但历史评测结果及阈值不能直接代表当前中心化配置或新场景的效果。

常见问题：

- **页面无法打开**：确认终端没有启动错误，端口未被占用，并访问对应的本机地址。
- **模型未加载**：检查配置中的模型路径、模型文件和准备脚本输出。
- **摄像头无画面**：检查设备权限、索引，以及是否被其他程序占用。
- **录入不足或样本不完整**：将完整物体放在画面中央，保证光线，适当调整角度后重试。
- **已录入物体显示未知**：查看录入样本及当前裁剪；尝试更接近录入时的角度和光线。倾斜校正不能解决严重遮挡、透视变化或定位不完整。
- **无法语音播报**：检查 `tts.mode`、网络或本地 TTS 服务；视觉识别可以独立使用。

## 项目结构

```text
main.py                    桌面界面及本地图片命令行入口
web/server.py              Flask 浏览器界面服务
web/templates/index.html   浏览器界面
config/                    默认及演示配置
core/                      应用编排与调试记录
vision/                    深度定位、特征提取、检索和摄像头
voice/                     语音识别、端点检测、麦克风和语音合成
nlu/                       中文规则意图解析
tools/                     模型准备、测试及评测工具
models/                    本地模型文件
data/                      物体库、样本和调试数据
```

## 已知局限

当前使用分类模型提取的视觉特征，外观相近的物体仍可能被混淆；少量样本也难以覆盖所有视角、光照和遮挡。深度候选有时会包含手或背景，或截掉物体的一部分。相似度阈值和角度校正只能改善部分场景，不能保证零误识别。界面帧率也不等于每次识别的推理速度。

## License

课程作业项目，仅供学习交流使用。
