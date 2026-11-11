# demo.yaml 配置说明

本文根据当前代码解释 `config/demo.yaml` 中的配置。部分字段为历史或预留配置，当前修改后不会影响运行，下面会明确标注。

## 基本约定

- `true` / `false` 表示开启 / 关闭，`null` 通常表示使用系统默认设备。
- `ms` 为毫秒，`sec` 或 `s` 为秒；1000 毫秒等于 1 秒。
- `models/...`、`data/...` 等路径相对于项目目录，加载配置时会转换成本地路径。
- 修改配置后通常需要重启程序。摄像头和麦克风也可以在网页设备设置中临时切换。

## app：摄像头和界面

| 配置项 | 当前值 | 含义 |
|---|---|---|
| `camera_index` | `0` | 默认摄像头索引。多摄像头时可尝试 `1`、`2` 等，也可在网页切换。 |
| `width` | `1920` | 向摄像头请求的画面宽度，单位像素。设备不支持时实际值可能不同。 |
| `height` | `1080` | 向摄像头请求的画面高度，单位像素。 |
| `locale` | `zh` | 预留语言标识；当前代码没有使用它切换语言。 |
| `text_input` | `false` | 预留项；当前网页文字输入不由此字段控制。 |
| `text_input_always` | `false` | 预留项；当前没有实际控制作用。 |
| `ui_fps_target` | `30` | 预留界面帧率目标；当前代码没有用这个字段限制网页帧率。 |

## asr：语音识别和麦克风

ASR 将语音转换成文字；VAD 用来判断什么时候有人说话。

| 配置项 | 当前值 | 含义 |
|---|---|---|
| `mic_device` | `null` | 麦克风设备。`null` 表示系统默认输入，也可填写设备索引或名称。 |
| `sample_rate` | `16000` | 录音和识别采样率，每秒 16000 个采样点。 |
| `vad_model` | `models/silero_vad/silero_vad.onnx` | Silero VAD 模型路径。加载失败时尝试使用基于音量的简单检测。 |
| `max_end_silence_ms` | `2500` | 一段语音之后允许的最长静音时间，单位毫秒。较大时更容忍停顿，但可能增加响应延迟。 |
| `min_speech_ms` | `250` | VAD 配置的最短语音时长，单位毫秒，用于过滤很短的声音。 |
| `e2e_model` | `models/sherpa-onnx-paraformer-zh-int8/model.int8.onnx` | 当前端到端 Paraformer 模型路径，优先用于当前模型布局。 |
| `model_dir` | `models/sherpa-onnx-paraformer-zh-int8` | ASR 模型目录；代码也会在这里查找模型、词表和 `am.mvn`。 |
| `tokens` | `models/sherpa-onnx-paraformer-zh-int8/tokens.txt` | ASR 词表，用于将模型输出转换为文字。 |
| `encoder` | `models/sherpa-onnx-paraformer-zh-int8/model.int8.onnx` | 兼容不同 Paraformer 模型布局的编码器路径；当前配置实际使用端到端模型。 |
| `decoder` | `models/sherpa-onnx-paraformer-zh-int8/decoder.int8.onnx` | 分离式 encoder/decoder 模型的解码器路径；当前端到端模型不依赖它。 |
| `num_threads` | `4` | ASR 推理使用的 CPU 线程数。 |
| `cmvn` | `models/sherpa-onnx-paraformer-zh-int8/am.mvn` | 本意是语音特征归一化文件路径，但当前代码不直接读取此字段，而是自动从模型目录查找 `am.mvn`。 |
| `device` | `cpu` | 当前实现固定使用 CPU；修改此字段不会切换到 GPU。 |
| `enabled` | `true` | 当前代码没有使用此字段作为语音识别开关；麦克风可通过网页设备开关或启动参数控制。 |

最长静音设为 2.5 秒，并不意味着每次都要等满 2.5 秒：程序对能提前识别的指令会尝试在较短停顿后处理。

## debug：识别过程调试记录

| 配置项 | 当前值 | 含义 |
|---|---|---|
| `enabled` | `false` | 是否保存识别过程的调试资料。开启后可能保存摄像头画面、候选区域和匹配信息。 |
| `dir` | `data/debug` | 调试资料保存目录。 |

画面可能包含人物，日常使用可保持关闭；排查定位或识别问题时可以临时开启。

## depth：深度定位

Depth Anything V2 估计相对深度，用于寻找可能的物体区域。深度表示相对远近，不是以米为单位的真实距离。

| 配置项 | 当前值 | 含义 |
|---|---|---|
| `enabled` | `true` | 是否启用深度定位。不可用时，识别流程会回退到画面中心区域。 |
| `model_path` | `models/depth_anything_v2/depth_anything_v2_vits.onnx` | 深度模型文件路径。 |
| `input_size` | `518` | 深度模型推理输入尺寸。通常越大细节越多，但计算更慢；应结合模型输入要求调整。 |
| `onnx_threads` | `4` | 深度模型使用的 CPU 推理线程数。 |
| `min_area_frac` | `0.005` | 部分候选区域筛选步骤中目标区域占画面面积的最小比例，约 0.5%。 |
| `max_area_frac` | `0.45` | 候选区域允许的最大面积比例，约 45%，用于过滤过大的区域。 |
| `expand_ratio` | `0.06` | 裁剪目标时按目标框宽高向外扩展的比例，约为每侧扩展 6%。 |
| `refresh_interval_sec` | `0.5` | 后台深度画面刷新的目标间隔，单位秒；实际速度也受推理耗时影响。 |
| `min_idle_sec` | `0.12` | 后台深度推理之间至少留出的休息时间，减少持续占用 CPU。 |
| `near_percentile` | `80.0` | 预留的近处深度百分位参数；当前虽读取该值，但定位逻辑没有使用它。 |

## memory：物体库、录入和匹配

物体录入时保存图像和特征；识别时将当前特征与物体库比较。

### 存储位置

| 配置项 | 当前值 | 含义 |
|---|---|---|
| `db_path` | `data/visual_memory.db` | SQLite 数据库路径，保存物体名称、样本索引等。 |
| `embeddings_path` | `data/resnet18_embeddings.npz` | ResNet18 特征向量文件。 |
| `samples_dir` | `data/samples` | 录入样本图像的保存目录。 |

### 匹配阈值

| 配置项 | 当前值 | 含义 |
|---|---|---|
| `center_features` | `true` | 对特征做中心化：减去物体库特征的共同成分，尝试增强不同物体之间的区别。 |
| `match_threshold` | `0.7` | 未中心化时的最低相似度；低于此值通常视为未知物体。 |
| `margin_threshold` | `0.05` | 未中心化时，第一名和第二名所需的最小分差。 |
| `match_threshold_centered` | `0.5` | 中心化匹配的最低相似度。 |
| `margin_threshold_centered` | `0.15` | 中心化匹配时，第一名和第二名所需的最小分差。 |
| `topk` | `3` | 检索接口默认返回的候选物体数量；具体调用也可显式指定。 |

中心化通常需要至少两个不同物体；只有一个物体时会回退到普通匹配阈值。例如中心化模式下第一名得分 `0.80`、第二名 `0.70`，虽然第一名高于 `0.5`，但差距 `0.10` 小于 `0.15`，系统仍可能回答“不太确定”。

相似度是特征匹配分数，**不是识别正确的概率**。降低阈值可能让更多角度通过，也可能增加认错。

### 录入和裁剪

| 配置项 | 当前值 | 含义 |
|---|---|---|
| `min_samples` | `3` | 成功录入一个物体至少需要的有效样本数。 |
| `max_samples` | `10` | 一次录入最多保存的样本数。 |
| `sample_duration_sec` | `2.5` | 正常采样窗口，单位秒。时间到了且达到最少样本数时，录入可以结束。 |
| `sample_timeout_sec` | `15.0` | 未达到最少样本时允许继续等待的总时限；不保证一定采样这么久。 |
| `sample_interval_ms` | `250` | 两次尝试采样之间的最小间隔，单位毫秒；实际间隔也受深度推理耗时影响。 |
| `min_crop_px` | `64` | 有效裁剪图像的最小边长，单位像素；太小的区域会被拒绝或尝试扩展。 |
| `center_roi_frac` | `0.45` | 深度定位不可用时，取画面中心的宽和高各 45% 作为候选区域。 |

若希望系统认识物体的不同侧面，录入期间应缓慢展示不同视角。仅增大样本数不保证视角多样，因为画面也可能一直相同。

## nlu：文字指令理解

| 配置项 | 当前值 | 含义 |
|---|---|---|
| `min_confidence` | `0.6` | 预留的意图识别置信度阈值；当前规则式意图解析没有使用此配置。 |

当前“这是什么”“记住这个叫……”等主要由规则解析，修改此值不会改变指令识别灵敏度。

## resnet18：物体特征提取

ResNet18 将裁剪出的物体图像转换为特征向量，供物体库进行相似度匹配。

| 配置项 | 当前值 | 含义 |
|---|---|---|
| `onnx_path` | `models/resnet18/resnet18.onnx` | 当前优先使用的 ONNX 模型路径。 |
| `model_path` | `models/resnet18/resnet18.onnx` | `onnx_path` 为空时使用的备用路径。 |
| `input_size` | `224` | 输入模型前将裁剪图像缩放到 `224 × 224`。 |
| `embed_dim` | `512` | 期望的特征向量维度，必须与模型和物体库兼容。 |
| `backend` | `onnxruntime` | 后端标识；当前实现实际固定使用 ONNX Runtime。 |
| `device` | `cpu` | 设备偏好；当前实现固定使用 CPU，修改此项不会启用 GPU。 |
| `precision` | `FP32` | 精度标识；当前实现不会根据此值切换模型精度。 |
| `allow_cpu_fallback` | `true` | 预留项；当前实现只有 CPU 推理路径，不通过此字段控制回退。 |
| `enabled` | `true` | 当前代码没有用此字段跳过 ResNet18 加载。 |

通常不建议单独修改 `embed_dim` 或 `input_size`：它们需要与实际模型及已保存特征保持兼容。

## tts：文字转语音

当前默认使用本地中文语音；`edge` 是可选在线语音模式。

### 通用与离线设置

| 配置项 | 当前值 | 含义 |
|---|---|---|
| `enabled` | `true` | 是否启用语音播报；设为 `false` 后仍可使用文字交互。 |
| `mode` | `offline` | `offline` 为本地中文语音；`edge` 为微软在线语音，失败后回退离线语音；`mock` 为测试用提示音。 |
| `play_audio` | `true` | 是否播放合成结果；设为 `false` 时仍可合成，但不通过扬声器播放。 |
| `offline_model_dir` | `models/sherpa-onnx-vits-zh-ll` | 离线中文语音模型目录。 |
| `offline_speaker` | `0` | 离线模型中的说话人编号，决定使用哪个声音；可用编号取决于模型。 |
| `offline_speed` | `1.0` | 离线语速倍率。通常大于 `1.0` 更快，小于 `1.0` 更慢。 |
| `offline_threads` | `2` | 离线语音合成使用的 CPU 线程数。 |
| `mock_wav` | `data/tts_beep.wav` | `mock` 模式下生成提示音的位置；默认 `offline` 模式不使用。 |
| `timeout_s` | `30` | 主要用于同步测试接口 `speak_blocking` 的等待期限；不会强制终止正常异步播报，也不是 Edge 网络请求超时。 |

### Edge 在线语音

以下字段只在 `mode: edge` 时使用：

| 配置项 | 当前值 | 含义 |
|---|---|---|
| `edge_voice` | `zh-CN-YunxiNeural` | 在线语音音色，当前为中文男声“云希”。 |
| `edge_rate` | `+0%` | 在线语速调整；`+0%` 表示默认速度。 |
| `edge_pitch` | `+0Hz` | 在线音高调整；`+0Hz` 表示默认音高。 |

常用的调整项包括摄像头 `camera_index`、物体识别的 `memory` 阈值，以及语音的 `tts.mode` 和 `offline_speed`。标为预留项的配置需要修改对应代码才能生效。
