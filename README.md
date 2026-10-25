# 基于 ResNet18 与语音交互的物体识别 Demo

> 计算机视觉课程大作业 · **选题一：物体识别**
> 用 ImageNet 预训练的 ResNet18 提取图像特征并建库，通过余弦相似度检索完成「录入 → 识别 → 未知判断」，再用中文语音下达指令、播报结果。

整个系统**运行时完全离线**，不使用任何云端 API、不需要 GPU 显卡，面向 Windows + Intel iGPU/CPU 平台优化。

![演示：语音提问「这是什么」，识别为马克杯（相似度 83%）](docs/screenshots/demo1.png)

---

## 目录

- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [评测与阈值标定](#评测与阈值标定)
- [性能基准](#性能基准)
- [工具脚本](#工具脚本)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [已知局限](#已知局限)

---

## 核心特性

| 能力 | 说明 |
| --- | --- |
| **物体录入** | 语音说「记住，这个叫我的蓝色水杯」，或按 `r` 用当前画面录入；多视角采样，落盘持久化 |
| **特征检索识别** | ResNet18 avgpool 512 维特征 → L2 归一化 → 余弦相似度检索 Top-K |
| **未知判断** | 最高分低于阈值时输出「未知物体」，阈值由真实数据实测标定，非拍脑袋 |
| **语音交互** | Silero VAD + Paraformer 中文 ASR（CPU INT8）识别指令，GPT-SoVITS 语音播报结果 |
| **纯离线** | 一次联网准备模型后，推理全程本地；NLU 为纯规则解析，不调用任何大模型 |
| **无 PyTorch 依赖** | 视觉侧全部走 OpenVINO / ONNX Runtime，运行时不需要 PyTorch |
| **无 CUDA 依赖** | 针对 Intel iGPU + CPU 部署，GPU 不可用时自动回退 CPU |
| **无麦克风也能用** | `--no-voice` 退化为键盘 / 文本指令，UI 内嵌输入框（支持中文 IME） |
| **批量模式** | 不开摄像头不开 UI，直接对本地图片目录做录入或识别，方便评测 |

---

## 系统架构

### 线程模型

`core/app.py` 采用多线程编排，**UI 主线程永不阻塞**在推理或语音上：

```
┌ 主线程      : OpenCV 摄像头 UI 循环 @ 30 FPS（不等待 TTS/ASR/推理）
├ 摄像头线程  : 取帧 → latest() 缓冲区
├ 检测线程    : 每 N 帧跑一次 YOLO（可选 ROI 助手），更新共享检测结果
├ 语音线程    : 麦克风 → VAD → ASR → 意图解析 → 队列
├ 意图工作线程: identify / remember / search（ResNet18 + 余弦检索）
└ TTS 客户端  : 异步工作线程（edge 在线 / GPT-SoVITS 独立进程）；UI 不等待
```

### 数据流

```
录入 enroll   : 裁剪 ROI → ResNet18 512-d → 名称+特征写入特征库
识别 identify : 裁剪 ROI → ResNet18 → 与特征库余弦比对 → 分数 ≥ 阈值 ? 名称 : 未知物体
语音 voice    : ASR 说出物体名称 / TTS 播报识别结果
```

> **关键设计**：ResNet18 **只在 identify / remember / search 时运行**，绝不进入摄像头热路径，
> 因此 30 FPS 的 UI 流畅度不受推理影响。YOLO 同理，默认每 4 帧检测一次（约 7.5 FPS）。

### 模块职责

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| **视觉** | `vision/feature_extractor.py` | ResNet18 特征提取（OpenVINO 优先，ONNX Runtime 回退） |
| | `vision/memory.py` | 特征库：SQLite 元数据 + NumPy 特征矩阵 + 余弦检索 |
| | `vision/detector.py` | YOLOv8n ROI 助手（OpenVINO / OpenCV DNN） |
| | `vision/object_cropper.py` | ROI 裁剪，YOLO 不可靠时回退中心区域 |
| | `vision/camera.py` | 摄像头采集 |
| | `vision/visualizer.py` | 画面叠加层（FPS、字幕、指令栏） |
| | `vision/position.py` | 中文方位描述（画面左侧 / 中间 / 右侧） |
| **语音** | `voice/asr.py` | Paraformer 中文 ASR（CPU INT8），支持 e2e 与 sherpa 两种模型布局 |
| | `voice/vad.py` | Silero VAD，端点检测；不可用时回退能量 VAD |
| | `voice/tts.py` | TTS 客户端，支持 edge（默认）/ gpt_sovits / mock 三种引擎 |
| | `voice/tts_server.py` | 独立 TTS 服务进程（**必须另开终端**） |
| | `voice/mic.py` | 麦克风流式采集（16 kHz） |
| **交互** | `nlu/intent.py` | 纯规则中文意图解析（6 类意图） |
| | `ui/text_input.py` | Tk 文本输入窗口（无麦克风方案） |
| **编排** | `core/app.py` | 多线程主应用 |
| | `core/config.py` | YAML 配置加载 + CLI 覆盖 + 路径解析 |

---

## 技术栈

| 环节 | 方案 | 说明 |
| --- | --- | --- |
| 特征提取 | **ResNet18**（ImageNet 预训练） | 选题一指定模型；取 avgpool 512 维输出 |
| 推理引擎 | **OpenVINO** 2024.4+ | 主后端，Intel iGPU / CPU，FP16 |
| 回退后端 | ONNX Runtime / OpenCV DNN | GPU 不可用或 IR 缺失时自动降级 |
| 目标检测 | **YOLOv8n** | 仅用于提出裁剪框，**身份判定始终来自 ResNet18 检索** |
| 相似度检索 | NumPy 余弦相似度 | 课程规模（几十个物体）无需 FAISS |
| 语音识别 | **Paraformer** 中文（sherpa-onnx, INT8） | 特征管线：kaldi fbank 80-d + LFR7 + CMVN |
| 语音端点 | **Silero VAD** | CPU 离线 |
| 语音合成 | **GPT-SoVITS v3** | 独立进程，经 HTTP 调用 |
| 意图理解 | 纯正则规则 | 无 LLM、无网络 |
| 界面 | OpenCV 窗口 + Tk 输入框 | 作业不要求复杂界面 |
| 语言 | Python 3.10 / 3.11 | — |

### 依赖

```txt
opencv-python>=4.8          numpy>=1.24,<3
openvino>=2024.4.0          onnxruntime>=1.18.0

sherpa-onnx>=1.10.50        kaldi-native-fbank>=1.10.40
sounddevice>=0.4.6          soundfile>=0.12.1

psutil>=5.9                 pyyaml>=6.0                tqdm>=4.66
pywin32>=306  (win32, 用于统计 iGPU 利用率)
```

> **运行时不需要 PyTorch。** GPT-SoVITS 作为独立模块单独安装（见下文）。

---

## 快速开始

### 1. 环境准备

建议使用 conda 虚拟环境（Windows）：

```bash
conda create -n vision_ai python=3.11 -y
conda activate vision_ai
pip install -r requirements.txt
```

> ⚠️ `run.bat` / `run.ps1` 默认调用 `C:\Users\admin\anaconda3\envs\vision_ai\python.exe`，
> 请按自己的实际路径修改，或直接用 `python main.py`。

### 2. 一次性准备模型（需要联网）

```bash
python tools/prepare_models.py --all        # ResNet18 + YOLOv8n + Paraformer + VAD
# 或分步
python tools/prepare_models.py --resnet18
python tools/prepare_models.py --yolo
python tools/prepare_models.py --asr
```

产物落在 `models/`（已被 `.gitignore` 排除，不入库）：

```
models/resnet18/resnet18_openvino_model/resnet18.xml   ← 核心特征模型
models/yolov8n/yolov8n_openvino_model/yolov8n.xml      ← ROI 助手
models/sherpa-onnx-paraformer-zh-int8/                ← ASR
models/silero_vad/silero_vad.onnx                      ← VAD
```

### 3. 准备语音合成

**默认用Edge TTS**（`pip install -r requirements.txt` 已含`edge-tts`），开箱即用、无需额外服务：

```bash
python main.py --camera 0
```

可选用 `config/default.yaml` 里的 `tts.edge_voice` 换音色：

| 音色 | 性别 | 特点 |
|---|---|---|
| `zh-CN-YunxiNeural` | 男 | 年轻、自然，最贴合语音助手（默认） |
| `zh-CN-YunjianNeural` | 男 | 播音腔，正式 |
| `zh-CN-YunyangNeural` | 男 | 解说/新闻腔 |
| `zh-CN-XiaoxiaoNeural` | 女 | 通用自然 |
| `zh-CN-XiaoyiNeural` | 女 | 年轻活泼 |

Edge TTS **需要联网**。若现场无网络或要求完全离线，改用本地 GPT-SoVITS：

GPT-SoVITS **不能**装在 `vision_ai` 环境（缺 `torchaudio`），请用另一个环境，
然后在**另一个终端**启动服务：

```bash
tools\start_tts_server.bat
# 或
python -m voice.tts_server --gpt-sovits-path E:/GPT-SoVITS --port 19880 ^
       --device cpu --ref-audio models/gpt_sovits/ref_audio.wav ^
       --prompt-text "<与参考音频内容逐字一致的文本>" --preload
```

再以 `--tts-mode gpt_sovits` 启动主程序。调试时可用 `--tts-mode mock` 起一个假音频服务。
TTS 不可用也不影响视觉功能，视觉侧会提示并跳过播报。

> 自训练音色请只用你本人或明确授权的语料。克隆可识别的真实人物声音涉及肖像/声音权益，
> 且会带来被冒用生成虚假内容的风险。

### 4. 运行

```bash
# 完整模式（摄像头 + 语音）
python main.py --camera 0

# 无麦克风（键盘 / 文本指令）
python main.py --camera 0 --no-voice

# 无摄像头窗口（CI 冒烟）
python main.py --headless

# Windows 快捷方式（自动切到 vision_ai 环境）
run.bat --camera 2 --no-voice
.\run.ps1 --camera 2 --no-voice
```

### 5. 冒烟测试（无需摄像头）

```bash
python tools/smoke_test.py
```

---

## 使用说明

### 语音指令

对麦克风说话即可，支持 6 类意图：

| 意图 | 示例说法 |
| --- | --- |
| 识别 | 「这是什么？」「你看这是什么」「认识这个吗」 |
| 录入 | 「记住，这个叫我的蓝色水杯」「把它叫做马克杯」「记住这个」 |
| 查找 | 「帮我找我的蓝色水杯」「X 在哪里」 |
| 遗忘 | 「忘记蓝色水杯」「删掉马克杯」 |
| 列出 | 「你记住了什么」「列出记忆」 |
| 帮助 | 「你能做什么」 |

### 键盘快捷键

| 按键 | 功能 |
| --- | --- |
| `q` / `Esc` | 退出 |
| `t` / `/` / `Enter` | 聚焦画面内文本输入框（支持中文 IME） |
| `i` | 识别当前画面（等价于「这是什么」） |
| `r` | 录入当前画面为「测试物品」 |
| `s` | 查找「测试物品」 |
| `l` | 列出已录入物体 |
| `o` | 打开本地图片识别 |
| `h` | 打印快捷键帮助 |

文本框内：`Enter` 发送，`Esc` 取消，`Backspace` 删除，`Tab` 退出焦点。

### 批量模式（不开摄像头、不开 UI）

```bash
# 录入：物体名称 + 图片目录
python main.py --enroll 水杯 --images data/enroll/cup

# 识别：单张本地图片
python main.py --recognize data/query/test.png
```

输出示例：

```text
result: 马克杯  similarity=83.1%  threshold=68%  [known]
result: 未知物体  best_candidate=马克杯  similarity=59.3%  threshold=68%  [unknown]
```

### 主要命令行参数

| 参数 | 说明 |
| --- | --- |
| `--config PATH` | 指定 YAML 配置（默认 `config/default.yaml`） |
| `--camera N` | 摄像头索引 |
| `--no-voice` | 关闭麦克风 / ASR |
| `--text-input` | 强制使用文本输入窗口 |
| `--headless` | 不开 OpenCV 窗口 |
| `--yolo-device GPU\|CPU` | 覆盖 YOLO 推理设备 |
| `--resnet-device GPU\|CPU` | 覆盖 ResNet18 推理设备 |
| `--precision FP32\|FP16\|INT8` | 覆盖推理精度 |
| `--tts-mode edge\|gpt_sovits\|mock` | 覆盖 TTS 模式（edge=微软在线，默认 / gpt_sovits=本地离线 / mock=测试） |
| `--enroll NAME --images DIR` | 批量录入 |
| `--recognize IMAGE` | 批量识别 |
| `-v / --verbose` | 调试日志 |

---

## 评测与阈值标定

### 为什么需要标定

`memory.match_threshold` 直接决定「已知 / 未知」的判定。ImageNet 预训练的 ResNet18
本质是**分类 backbone** 而非细粒度检索模型，真实余弦分布必须实测，不能拍脑袋。

`tools/eval_dataset.py` 用真实图片跑完整链路，分别统计：

- **正样本**：同一物体、不同视角 → 余弦相似度
- **负样本**：未录入物体 → 余弦相似度
- 扫描阈值求 balanced accuracy 最优点，输出正负分布直方图

### 实测结果（2026-10-06）

数据集：6 个录入物体 × 6 张注册图 + 3 张留出测试图 + 4 个未录入物体（香蕉 / 碗 / 雨伞 / 椅子）

| 指标 | 数值 |
| --- | --- |
| 测试集 Top-1 命中率 | **100%（18/18）** |
| 正样本余弦相似度 | min 0.7177 / mean 0.7940 |
| 负样本余弦相似度 | max 0.6361 / mean 0.5646 |
| 正负分布间隙 | 0.0816（完全分离） |
| **采用阈值** | **0.6769**（间隙中点，两侧余量相等） |

复现命令：

```bash
python tools/eval_dataset.py           # 评测 + 建议阈值
python tools/eval_dataset.py --apply   # 直接写回 config
```

> 正负分布完全分离说明该阈值是**有余量**的，不是贴边硬凑。
> 最难的负样本是「碗 → 马克杯」（0.586）、椅子、香蕉 —— 印证了
> 「形状近似但类别不同」正是分类特征的薄弱点。

### 已知数据要求完成情况

选题书要求 vs 实际：

| 要求 | 完成情况 |
| --- | --- |
| ≥5 个物体，每个 ≥5 张图 | ✅ 6 个物体 × 6 张 |
| 注册图与测试图不完全相同 | ✅ 每物体留出 3 张测试图 |
| ≥3 个未录入物体测未知判断 | ✅ 4 个（香蕉 / 碗 / 雨伞 / 椅子） |

---

## 性能基准

```bash
python benchmark.py                          # 完整基准
python benchmark.py --skip-asr --skip-tts    # 只测视觉
python benchmark.py --runs 50 --out benchmark_report.json
```

测量项：ResNet18 推理延迟（avg / p50 / p95）、YOLO 延迟、ASR 延迟、TTS RTF、
进程与系统内存、CPU 占用、iGPU 设备与利用率（best-effort，需 pywin32）。

已记录数据（CPU / FP32）：

| 指标 | 数值 |
| --- | --- |
| **Edge-TTS**（默认）平均延迟 | p50 1.79s（含联网），RTF 0.35–0.65 |
| 首次 TTS 合成 RTF（含懒加载，GPT-SoVITS） | 4.90 |
| 热态平均 TTS RTF（GPT-SoVITS） | 1.29（交互可接受） |
| TTS 引擎常驻内存（GPT-SoVITS） | ≈ 2 GB（独立进程，视觉侧 30 FPS 不受影响） |
| UI 帧率 | 约 29.7 FPS（1280×720） |
| YOLO 检测频率 | 每 4 帧一次 @ 30 FPS（约 7.5 FPS，满足 ≥5 FPS 要求） |

> Edge-TTS 延迟含网络往返，断网或网络慢时会变长；对延迟敏感或现场无网络
> 请切 `--tts-mode gpt_sovits`（本地离线，需另开终端启动服务）。

---

## 工具脚本

| 脚本 | 用途 |
| --- | --- |
| `prepare_models.py` | 一次性下载 / 转换模型（ResNet18、YOLOv8n、Paraformer、VAD） |
| `fetch_dataset.py` | 从 Wikimedia Commons 构建课程数据集；抓取时即用 YOLO 校验并裁剪 ROI，与在线链路一致 |
| `fetch_test_scenes.py` | 抓**完整场景测试图**（物体只占画面一角，含杂乱背景），用于端到端验证；与 `fetch_dataset` 的裁剪图互补 |
| `run_feature_tests.py` | 场景图端到端功能测试：逐图断言「检测 / 裁剪 / 识别 / 未知判断 / 位置播报」5 项，另测未录入类拒绝 |
| `eval_dataset.py` | 离线评测 + 阈值标定，扫描阈值求 balanced accuracy 最优点 |
| `calibrate_threshold.py` | 从真实相似度数据标定阈值（正样本 / 负样本 / 光照变体） |
| `benchmark.py`（根目录） | 性能基准测试 |
| `smoke_test.py` | 离线冒烟测试，无需摄像头 / 麦克风 |
| `export_infer_weights.py` | 把 GPT-SoVITS 训练断点导出为纯推理权重（剥离 optimizer） |
| `verify_export_pipeline.py` | 端到端验证「训练权重 → 导出 → 引擎加载 → 实际合成」 |
| `trim_gpt_sovits.py` | 裁剪 GPT-SoVITS 仓库，只保留 v3 推理链路必需文件 |
| `trace_tts_deps.py` | 用 audit hook 追踪推理真实读取的仓库文件（权威依赖清单） |
| `probe_tts_deps.py` | 删除法验证依赖：逐个移走候选文件实测引擎能否启动 |
| `probe_tts_deps_nondestructive.py` | 非破坏性依赖验证（屏蔽模块 / 模拟文件不存在） |
| `retest_tts_deps.py` | 用长文本 / 数字 / 中英混排复测短句测不出的依赖 |
| `verify_tts_server_config_isolation.py` | 验证 `--vits-weights` 不会污染 `tts_infer.yaml` |
| `start_tts_server.bat` | 一键启动独立 TTS 服务 |

---

## 项目结构

```
.
├── main.py                    # 入口：UI / 批量模式、CLI 参数
├── benchmark.py               # 性能基准
├── config/
│   └── default.yaml           # 唯一配置源，含实测阈值标定注释
├── core/
│   ├── app.py                 # 多线程主应用编排
│   └── config.py              # YAML 加载 + CLI 覆盖 + 路径解析
├── vision/
│   ├── feature_extractor.py   # ResNet18 特征提取（OpenVINO / ONNX Runtime）
│   ├── memory.py              # 特征库：SQLite + NumPy 余弦检索
│   ├── detector.py            # YOLOv8n ROI 助手
│   ├── object_cropper.py      # ROI 裁剪 + 中心区域回退
│   ├── camera.py              # 摄像头采集
│   ├── visualizer.py          # 画面叠加层
│   └── position.py            # 中文方位描述
├── voice/
│   ├── asr.py                 # Paraformer 中文 ASR
│   ├── vad.py                 # Silero VAD
│   ├── tts.py                 # GPT-SoVITS 客户端
│   ├── tts_server.py          # 独立 TTS 服务（HTTP）
│   └── mic.py                 # 麦克风采集
├── nlu/
│   └── intent.py              # 纯规则中文意图解析
├── ui/
│   └── text_input.py          # Tk 文本输入窗口
├── tools/                     # 见上表
├── models/                    # 模型（gitignored，需 prepare_models 生成）
├── data/                      # 特征库、样本、数据集（gitignored）
└── docs/screenshots/          # 演示截图
```

### 数据落盘

| 路径 | 内容 |
| --- | --- |
| `data/visual_memory.db` | SQLite：物体名称、时间戳、样本索引 |
| `data/resnet18_embeddings.npz` | 压缩特征矩阵 + object_ids，启动时秒级加载 |
| `data/samples/<name>/*.png` | 录入时保存的样本图 |
| `data/samples/<name>/*.npy` | 对应的 512 维特征 |

> 特征库与元数据分离保存，重启后可直接继续使用，无需重新录入。

---

## 配置说明

所有配置集中在 `config/default.yaml`，`core/config.py` 负责加载、CLI 覆盖与相对路径解析。

| 配置块 | 关键项 |
| --- | --- |
| `app` | 摄像头索引、分辨率、UI 帧率目标、文本输入开关 |
| `yolo` | 后端、设备、精度、置信度、NMS、检测间隔帧数 |
| `resnet18` | 后端、设备、精度、输入尺寸、`embed_dim: 512` |
| `memory` | **`match_threshold: 0.6769`（实测标定）**、`topk`、`min/max_samples`、`embed_dim` |
| `asr` | 模型路径、采样率 16 kHz、线程数、静音判定阈值 |
| `tts` | 模式、服务地址、参考音频、`prompt_text`、超时 |
| `nlu` | `min_confidence: 0.6` |

### 两个必须注意的配置项

**1. `tts.prompt_text` 必须与 `ref_audio` 的实际说话内容逐字一致**，否则音色与韵律会明显变差。

**2. INT8 精度需要量化后的模型文件**，直接设 `--precision INT8` 而没有对应的量化
`.xml` 会失败。

---

## 已知局限

- **类别粒度受限**：ResNet18 是分类 backbone，对「形状近似但类别不同」的物体
  （碗 vs 马克杯）区分力有限。细粒度识别需换用度量学习模型（ArcFace / DINOv2 等）。
- **依赖光照与视角**：录入与识别时的光照、角度差异过大会影响余弦相似度，
  建议同一物体从多个角度录入。
- **半双工语音**：为避免识别到自己的播报，识别指令期间会短暂静音麦克风，
  存在少量重叠语音的边界情况。
- **单机单进程**：特征库用 SQLite + 内存矩阵，适合几十到几百个物体量级，
  上万级需要换 FAISS / 向量数据库。
- **Windows 优先**：部分便捷脚本（`run.bat`、`start_tts_server.bat`）为 Windows 编写。

---

## 说明

本项目为课程大作业成果。仓库内使用了大量中文注释，重点解释**为什么这样设计**
（例如阈值的实测依据、线程模型的取舍、依赖验证方法），便于答辩时说明代码与流程。

**数据与隐私**：仅使用公开数据集（Wikimedia Commons）与自摄图片；
人脸数据不涉及本项目。如需在自己的数据集上测试，请遵守对应数据的许可协议。

## License

课程作业项目，仅供学习交流使用。
