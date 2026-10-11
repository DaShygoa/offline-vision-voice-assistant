# 架构文档

> 基于 ResNet18 与语音交互的物体识别 Demo · 计算机视觉课程大作业选题一
>
> 本文说明**系统为什么这么设计**。使用说明见 [README.md](README.md)，实测数据见 [实测结论](#十实测结论)。

---

## 目录

- [一、设计目标与约束](#一设计目标与约束)
- [二、整体架构](#二整体架构)
- [三、线程模型](#三线程模型)
- [四、视觉链路](#四视觉链路)
- [五、特征库设计](#五特征库设计)
- [六、语音链路](#六语音链路)
- [七、意图解析](#七意图解析)
- [八、配置体系](#八配置体系)
- [九、降级策略](#九降级策略)
- [十、实测结论](#十实测结论)
- [十一、关键设计决策](#十一关键设计决策)
- [十二、扩展方向](#十二扩展方向)

---

## 一、设计目标与约束

### 选题书要求

| 要求 | 设计响应 |
| --- | --- |
| 用 ImageNet 预训练 ResNet18 | 取 avgpool 512 维特征，不微调、不训练 |
| 物体录入 + 特征保存 | 多视角采样 → SQLite + NumPy 特征库 |
| 识别返回名称 + 相似度 | 余弦检索 Top-K，附带相似度百分比 |
| 相似度低时显示未知物体 | 阈值判定，阈值**实测标定** |
| 语音说出物体名称 + 语音播报 | Paraformer ASR + GPT-SoVITS TTS |
| 简单界面 | OpenCV 窗口 + 内嵌文本框 |
| ≥5 物体 ×≥5 图，≥3 未录入物体 | `tools/fetch_dataset.py` 一键构建，自检 PASS |

### 部署环境约束（决定了全部技术选型）

> 目标机器：**Windows + Intel iGPU / CPU，无 NVIDIA 显卡**

| 约束 | 影响 |
| --- | --- |
| 无 CUDA | 不使用 TensorRT / PyTorch GPU 路径 |
| 无 PyTorch 依赖 | 视觉推理走 OpenVINO / ONNX Runtime；torch 仅用于一次性导出 |
| 需完全离线 | 不用任何云端 API；NLU 用规则而非 LLM |
| 单次会话较长 | 特征库落盘，重启可继续用 |

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    AssistantApp (core/app.py)                │
│                    编排层：只做调度，不含算法                  │
└─────────────────────────────────────────────────────────────┘
        │            │             │              │
        ▼            ▼             ▼              ▼
   ┌─────────┐  ┌──────────┐  ┌─────────┐   ┌──────────┐
   │ 视觉侧   │  │  语音侧   │  │  意图层  │   │  UI 层    │
   │ vision/ │  │  voice/  │  │  nlu/   │   │   ui/    │
   └─────────┘  └──────────┘  └─────────┘   └──────────┘
        │             │             │
        ▼             ▼             ▼
   ┌─────────┐  ┌──────────┐  ┌──────────────┐
   │OpenVINO │  │sherpa-onnx│  │  纯正则规则   │
   │ ONNX RT │  │GPT-SoVITS │  │  无 LLM/网络 │
   └─────────┘  └──────────┘  └──────────────┘
        │
        ▼
   ┌──────────────────────────────────────┐
   │  特征库 (vision/memory.py)          │
   │  SQLite 元数据 + NumPy 512-d 矩阵    │
   └──────────────────────────────────────┘
```

### 分层原则

| 层 | 职责 | 不做什么 |
| --- | --- | --- |
| `core/` | 线程编排、生命周期、组件装配 | 不含视觉/语音算法 |
| `vision/` `voice/` | 算法实现 | 不感知 UI，不直接控制线程 |
| `nlu/` | 文本 → 意图 | 不访问摄像头或特征库 |
| `ui/` | 输入呈现 | 不做业务判断 |

**好处**：`vision/` 和 `voice/` 完全可独立测试。`tools/` 里的脚本就是直接 import
这些模块做批量评测，不启动 UI、不需要摄像头。

---

## 三、线程模型

### 线程清单

`core/app.py` 最多启动 5 个线程：

| 线程 | 目标函数 | 频率 | 阻塞风险 |
| --- | --- | --- | --- |
| **主线程** | `run_ui()` | 30 FPS | 不等待任何推理/语音 |
| camera | `Camera._loop` | 尽可能快 | 写帧到 `latest` 缓冲区 |
| detect | `_detect_loop` | 每 N 帧（默认 4） | YOLO 推理约 0.5–1 s/次 |
| intent | `_intent_loop` | 队列驱动 | ResNet18 推理 + 检索 |
| voice | `_voice_loop` | VAD 端点触发 | ASR 解码 |

### 核心约束：UI 永不阻塞

```python
# core/app.py run_ui()
frame, _ = self.camera.latest()      # 只读最新帧，不等采集
vis = self._viz.draw(frame, overlay) # 纯渲染
cv2.imshow(win, vis)
cv2.waitKey(1)                      # 非阻塞轮询键盘
```

UI 只做三件事：取最新帧 → 叠加渲染 → 轮询按键。**任何耗时操作都不在主线程**。

### 帧缓冲区与背压

`Camera.latest()` 语义是「拿最新的那一帧」，不是「排队取帧」：

```python
def latest(self):
    with self._lock:
        return self._frame.copy(), self._frame_id   # 覆盖式，非队列
```

**为什么**：若用队列积压，UI 会越来越延迟。覆盖式保证 UI 永远显示**当前**画面，
代价是可能丢帧——对交互式 UI 丢帧可接受，延迟不可接受。

### 组件降级不影响启动

```python
# core/app.py start()
if self.detector is not None:
    try:
        self.detector.load()
    except Exception as e:
        log.warning("YOLO load failed (ROI falls back to center crop): %s", e)
        self.detector = None
        self.yolo_enabled = False    # ← 继续启动，不中断
```

YOLO 加载失败只降级 ROI 来源，**不阻止启动**。ResNet18 加载失败才真正影响功能，
因为它是识别主体。

### 半双工语音

```python
self._mic_muted_until = time.time() + min(8.0, 1.0 + 0.18 * len(reply))
```

播报后按文本长度短暂静音麦克风，避免 TTS 声音被自己识别回去。
这是简单方案，成本是边界情况下可能漏掉快速连说的指令。

---

## 四、视觉链路

### 职责划分：YOLO 提框，ResNet18 定身份

这是本项目最关键的设计取舍：

```
YOLO  ──→ 提出候选框「剪哪一块」
ResNet18 ──→ 判断「这是什么东西」
```

`config/default.yaml` 原文：

> Only proposes crop boxes; identity always comes from ResNet18 retrieval.

**为什么这样切分**：

1. **YOLO 的类别知识用不上**。用户自定义物体（"我的蓝色水杯"）不在 COCO 80 类里，
   YOLO 对它只能给出 "bottle"，无法区分用户的多个瓶子。
2. **符合选题书要求**。题目明确要求用 ResNet18 特征检索，YOLO 只是辅助。
3. **可插拔**。YOLO 关闭后系统仍能工作（见[降级策略](#九降级策略)）。

### ROI 选取优先级

`vision/object_cropper.py`：

```
crop_for_learning(frame, dets, generic_class)
  │
  ├─ 若指定了 generic_class，先按COCO 类别过滤
  │    best_center_detection(dets, class_id=杯子的 COCO id)
  │      └─ 打分 = 0.6 × 与中心 ROI 的 IoU + 0.4 × 离画面中心的接近度
  │
  ├─ 命中 → expand_box(×1.06) → crop_box → source = "yolo"
  │
  └─ 未命中 → center_roi(画面中心 45%) → crop_box → source = "center_roi"
```

`best_center_detection` 的打分用「与中心 ROI 的重叠」而非「置信度最高」——
因为用户是对着镜头提问，**画面中央的物体才是他要问的那个**。

### 质量门控

裁剪后、embedding 前先检查，拒绝无效样本：

```python
def quality_ok(self, crop):
    if min(crop.shape[:2]) < self.min_crop_px: return False  # 太小
    mean = gray.mean(); std = gray.std()
    if mean < 12 or mean > 245:  return False   # 全黑 / 过曝
    if std < 8:                  return False   # 纯色（多半是拍空）
    return True
```

### 录入时的多样性过滤

静态画面下连续采样会得到几乎相同的特征，浪费配额：

```python
if embeds and not force:                      # force: 保证 min_samples
    sims = [float(emb @ s) for s in embeds]
    if max(sims) > 0.995:                     # 与已有样本几乎一样
        continue                              # 丢弃这一帧
```

`force = len(samples) < min_samples` 保证即使画面完全静止，
也能凑够最少样本数（只是多样性差），不会无限等待。

### 预处理与后处理

`vision/feature_extractor.py`：

| 步骤 | 参数 |
| --- | --- |
| 缩放 | 224×224，`INTER_AREA` 缩小时质量更好 |
| BGR → RGB | OpenCV 是 BGR，ImageNet 权重按 RGB 训练 |
| 归一化 | `/255.0` 后减均值除标准差（ImageNet 统计量） |
| NCHW | `transpose(2,0,1)[None]` |
| 输出取 avgpool | 512 维，**在 fc 之前** |
| L2 归一化 | 余弦相似度 = 点积，此后无需再除范数 |

**为什么取 fc 之前**：fc 层输出 1000 维 ImageNet 类别分布，做检索时维度冗余且
类别相关性强；avgpool 的 512 维是通用语义特征，更适合做特征检索。

---

## 五、特征库设计

### 存储分层

```
data/visual_memory.db          SQLite：物体元数据 + 样本索引（可查询、可删）
data/resnet18_embeddings.npz   NumPy：压缩特征矩阵，启动时秒级加载
data/samples/<name>/*.png     样本图（人工核对用）
data/samples/<name>/*.npy     对应特征（重建索引的数据源）
```

**为什么不只用数据库**：余弦检索要遍历全库，SQLite 逐行读太慢。
把特征抽出来存成连续矩阵，一次 `np.load` 全量载入内存，检索变成一次矩阵乘。

**为什么还保留 SQLite**：需要按名称查对象、列出全部物体、级联删除样本——
这些是关系操作，且元数据量小（几十条），SQLite 天然合适。

### 内存索引

```python
scores = self._matrix @ qq      # 形状 (N,512) @ (512,) → (N,) 一次算完
```

同物体多视角取最高分：

```python
best = {}
for i, s in enumerate(scores):
    oid = self._row_object_id[i]
    if oid not in best or s > best[oid]:
        best[oid] = s           # 同物体多个视角 → 取最优视角
```

### 维度自愈

特征库换模型后会残留旧维度的脏数据，代码做了三重防御：

1. 载入时校验 `mat.shape[1] == self.embed_dim`，不匹配则整体重建
2. 重建时跳过维度不符的 `.npy`（注释：*skip stale DINO/old features*）
3. `_purge_featureless_objects()` 清理一个特征都没有的残留物体

> 这是 DINOv2 → ResNet18 迁移时踩过的坑：旧库里有 768 维 DINO 特征，
> 新代码是 512 维，直接查会静默出错。

### 为什么用 NumPy 不用 FAISS

课程规模是**几十个物体**，即几百到几千个 512 维向量。矩阵乘在这个量级是微秒级，
引入 FAISS 反而增加依赖和维护成本。代码注释也明确写了这一判断。

---

## 六、语音链路

### 进程隔离

```
视觉进程 (vision_ai 环境)          TTS 进程 (GPTSoVits 环境)
┌──────────────────────┐          ┌──────────────────────┐
│ AssistantApp         │  HTTP    │ voice/tts_server.py  │
│  └ TtsClient ────────┼─────────→│  └ GPT-SoVITS 引擎    │
│      异步，不阻塞 UI │  POST    │  常驻 ~2GB           │
└──────────────────────┘          └──────────────────────┘
```

**为什么要独立进程**：两个原因，都是硬约束。

1. **依赖冲突**：`vision_ai` 环境没有 `torchaudio`，GPT-SoVITS 必需。
   两者装在同一环境会互相拖垮。
2. **资源隔离**：GPT-SoVITS 常驻约 2GB 且 CPU 推理密集（RTF 1.29）。
   同进程会直接抢走视觉侧的 CPU，导致 UI 掉帧。

### HTTP 接口

```python
GET  /health                      → {"ok": true, "mode": "gpt_sovits"}
POST /tts  {"text": ..., "ref_audio": ..., "prompt_text": ...}
                                   → audio/wav bytes（或 JSON 路径）
```

用 `ThreadingHTTPServer`，合成期间仍能响应 `/health`。

### 必须绕过系统代理

```python
# voice/tts.py
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
```

本机设了 `http_proxy` 环境变量，`urllib` 会把 `127.0.0.1` 也塞进隧道，
导致 `502 Bad Gateway`。这是实测踩过的坑，代码里留了注释。

### 半双工策略

```python
if not self._mic_muted_until or time.time() > self._mic_muted_until:
    ...  # 送去 VAD
```

播放 TTS 期间丢弃麦克风数据。代价是 TTS 播放期间的语音指令会丢失。

### ASR 双模型布局兼容

`voice/asr.py` 支持两种 Paraformer 导出格式：

| 布局 | 文件 | 特征管线 |
| --- | --- | --- |
| A. 端到端 FunASR | `model.int8.onnx` | kaldi fbank 80-d + LFR7 + CMVN |
| B. sherpa-onnx | `encoder/decoder.int8.onnx` | sherpa 内部处理 |

自动检测 `decoder.int8.onnx` 是否存在来决定走哪条路。
CMVN 统计量的解析抄自 sherpa-onnx 官方测试脚本（`am.mvn`）：

```python
if not line.startswith("<LearnRateCoef>"): continue
neg_mean = np.array(t)   # 第一行
inv_std  = np.array(t)   # 第二行
```

---

## 七、意图解析

### 纯规则，不用 LLM

`nlu/intent.py` 用有序正则匹配 6 类意图。这样做的原因很实际：
**离线要求** + 中文指令句式高度固定，规则匹配准确率足够，且零延迟、零成本、
可解释（答辩时能直接说每个 pattern 对应什么说法）。

### 顺序敏感

```python
# LIST 必须先于 REMEMBER ——否则「你记住了什么」会匹配「记住…」
if _try(_LIST_PATTERNS, t): return LIST

# REMEMBER 必须先于 SEARCH ——「记住，这个叫X」含「叫」
if _try(_REMEMBER_PATTERNS, t): return REMEMBER

if _try(_SEARCH_PATTERNS, t): return SEARCH
```

这是模式匹配设计的典型坑：**更具体的模式必须先试**。
代码里用大写注释标出了这两处排序理由。

### 抽取物体名

```python
_LAB = r"[一-龥A-Za-z0-9_·\-]{1,32}"
```

字符集同时支持中文、英文、数字，允许 `·`（名字里的分隔符）和 `-`。
长度上限 32 防止正则回溯。

---

## 八、配置体系

### 单一配置源 + 三层覆盖

```yaml
# config/default.yaml —— 唯一配置源，带详细中文注释
memory:
  match_threshold: 0.6769    # 实测标定值，见下
```

覆盖优先级：**CLI 参数 > 环境变量 > YAML 默认值**

```python
cfg = load_config(args.config, overrides)      # core/config.py
```

`load_config` 还负责把相对路径解析为绝对路径（相对项目根），
这样从任意工作目录启动都不会找错文件。

### 阈值的标定

`match_threshold: 0.6769` **不是拍脑袋的**。`tools/eval_dataset.py` 用真实数据实测：

```
正样本（同物体不同视角）：min 0.7177   mean 0.7940
负样本（未录入物体）    ：max 0.6361   mean 0.5646
                        ↓
                    中间 0.0816 完全分离 → 取中点 0.6769，两侧余量相等
```

⚠️ **阈值强依赖数据集**。换物体后必须重跑：

```bash
python tools/eval_dataset.py           # 看建议值
python tools/eval_dataset.py --apply   # 写回配置
```

本机在新数据集上实测：`separable: False`（正样本 min 0.5831 < 负样本 max 0.7582），
建议值 0.6323。详见[实测结论](#十实测结论)。

---

## 九、降级策略

系统的设计原则：**任何单个组件失效都不应让整体不可用**。

| 失效组件 | 表现 | 降级行为 |
| --- | --- | --- |
| YOLO 模型缺失 | 无候选框 | 回退中心 ROI，功能完整 |
| YOLO 推理异常 | 同上 | `detector = None`，继续运行 |
| ResNet18 缺失 | 无法识别/录入 | 明确提示运行 `prepare_models.py --resnet18` |
| 麦克风缺失 | 无法语音输入 | 退化为键盘/文本输入 |
| Tkinter 缺失 | 无输入窗口 | 退化为画面内OpenCV 输入框（按 `t`） |
| GPT-SoVITS 未启动 | 无播报 | 记warning，其余功能不受影响 |
| TTS 引擎缺失 | 无法播报 | `tts_mode: mock` 生成提示音 |
| VAD 不可用 | 端点检测失效 | 回退能量 VAD |

关键实现：

```python
# Tkinter 缺失 → 画面内输入框
except ImportError:
    log.warning("Falling back to in-frame OpenCV text input (press t)")
```

```yaml
tts:
  mode: mock      # 无 GPT-SoVITS 时仍可跑通全链路
```

---

## 十、实测结论

> 本机环境：Windows，Python 3.13，OpenVINO 2026.4，**无 GPU**，两个模型均跑 CPU/FP16。
> 数据集：6 物体×6 图 + 18 测试 + 16 未知（`fetch_dataset.py` 自检 PASS）。

### YOLO 能否被替代

这是本机实测回答的问题，结论**修正**了最初的直觉判断。

方法：把物体贴到 1280×720 杂乱背景，分「居中」「偏置」两种摆法，共 104 张场景图，
用真实 `_do_identify` 取框逻辑跑 A/B。

| 指标 | YOLO 开 | YOLO 关 | 差异 |
| --- | --- | --- | --- |
| 排名 Top-1 | 86.1% | 77.8% | **−8.3pp** |
| └物体居中 | 88.9% | 83.3% | −5.6pp |
| └ 物体偏置 | 83.3% | 72.2% | **−11.1pp** |
| 未知拒绝率 | 75.0% | 90.6% | +15.6pp |
| 单次识别耗时 | 262 ms | 259 ms | −3 ms |

**结论**：

- ✅ **可以关闭**：改 `yolo.enabled: false` 即可，代码原生支持，功能不崩
  （已验证 `detector=None` 正常走中心 ROI）。
- ❌ **不能说「不影响准确率」**：实测损失约 8pp。
- 📌 **偏置场景损失最大**（−11.1pp）——物体没摆到画面中央时，YOLO 价值最高。
- ⚠️ 关掉后未知拒绝率反而更高，但代价是已知物体也被误拒
  （决策准确率 76.4% → 52.8%）。

### 端到端验证

通过 `tools/capture_live_ui.py` 驱动真实 `AssistantApp` 走完整链路
（`inject_text` → 意图队列 → `_dispatch` → `_do_identify` → `Visualizer.draw`）：

| 输入 | 识别结果 | YOLO 框 |
| --- | --- | --- |
| bottle | 你的 bottle，相似度 91% | 1 |
| cup | 你的 cup，相似度 76% | 1 |
| bowl（未录入） | **未知物体**，最相似 bottle 仅 63% < 阈值 68% | 1 |

未知判断与语音播报文案均正常。

### 与README 数字的差异

README 记录 `eval_dataset.py` 的 100%（36/36），本机同口径复现一致。
但需注意：

- **100% 是用「已裁紧的导出图」直接检索**测得的
- 换成真实场景（物体占画面一部分 + 背景干扰）后是 **86.1%**
- 阈值 0.6769 在新数据集上正负样本**不可分**，建议 0.6323

答辩时建议说「特征检索在受控数据集上 top-1 100%，真实场景约 86%」，
比单报100% 更站得住。

---

## 十一、关键设计决策

### 记录几个「为什么不」

| 决策 | 理由 |
| --- | --- |
| ResNet18 不微调 | 选题书要求；且 6 个物体的数据量微调必过拟合 |
| 特征取 avgpool 而非 fc 输出 | fc 是 1000 类分类头，类别相关；avgpool 是通用语义特征 |
| NumPy 而非 FAISS | 几十个物体，微秒级；引入依赖不划算 |
| 特征矩阵与元数据分离存储 | 元数据要SQL 查询，检索要连续内存，两者需求不同 |
| YOLO 每N 帧而非每帧 | 检测只用于提框，不需要高帧率；省 CPU 给 ResNet18 |
| NLU 用正则不用 LLM | 离线要求 + 句式固定 + 可解释 |
| 覆盖式帧缓冲不用队列 | UI 要「当前画面」而非「不丢帧」，延迟比丢帧更影响体验 |
| 深度覆盖用 center ROI 而非报错 | 课堂演示机不该因少一个模型就跑不起来 |

### 踩过的坑（代码里留了注释）

1. **`Infer Request is busy`**：同一 OpenVINO compiled model 并发 `infer()` 会抛异常。
   截图脚本里改为让 app 自己的检测线程独占推理。
2. **`prompt_text` 必须与 `ref_audio` 逐字一致**，否则音色韵律明显变差。
3. **GPT-SoVITS 的 `TTS_Config.__init__` 会就地重写 `tts_infer.yaml`**：
   用 `--vits-weights` 指定临时路径后，下次裸跑会因文件已删而 `FileNotFoundError`。
   `voice/tts_server.py` 里做了还原，配套验证脚本 `verify_tts_server_config_isolation.py`。
4. **特征库跨模型迁移**：DINOv2（768 维）残留到 ResNet18（512 维）会静默出错，
   已加维度校验与自动清理。

---

## 十二、扩展方向

按投入产出排序，供后续改进参考。

| 方向 | 预期收益 | 成本 | 说明 |
| --- | --- | --- | --- |
| **细粒度检索模型**（DINOv2 / ArcFace） | **最大** | 中 | 解决「碗 vs 马克杯」这类形状近似问题，是当前准确率的主要瓶颈 |
| ROI 改用分割（MobileSAM） | 较大 | 中 | 检测框含大量背景，分割更贴合「剪哪一块」的需求。代码注释已预留方向 |
| 特征库换 FAISS | 小（当前规模） | 中 | 超过约 1 万向量才需要 |
| 录入时自动多角度提示 | 中 | 低 | 引导用户转动物体，天然拉开视角差异 |
| 阈值自适应 | 中 | 中 | 按特征库内最近邻分布动态定阈值，而非固定值 |
| ASR 热词增强 | 小 | 低 | 物体名称作为热词喂给 Paraformer，提升专有名词识别率 |

### 已知局限

- **类别粒度受限**：ResNet18 是分类 backbone，形状近似但类别不同的物体区分力有限。
  实测最难负样本是「碗 → 马克杯」，得分 0.6999–0.7582，**全部超过阈值 0.6769**，
  即未知判断在这里失效。
- **依赖光照与视角**：录入与识别的光照角度差异过大会影响余弦相似度。
- **单机单进程**：SQLite + 内存矩阵适合几十到几百物体量级。

---

## 附：模块速查

| 文件 | 行数量级 | 核心职责 |
| --- | --- | --- |
| `core/app.py` | ~900 | 多线程编排、事件分发、UI 循环、按键处理 |
| `core/config.py` | ~50 | YAML 加载、CLI 覆盖、相对路径解析 |
| `vision/memory.py` | ~390 | 特征库：SQLite + NumPy 索引 + 余弦检索 |
| `vision/detector.py` | ~400 | YOLOv8n，OpenVINO 优先，OpenCV DNN 回退 |
| `vision/feature_extractor.py` | ~170 | ResNet18 提取、L2 归一化 |
| `vision/object_cropper.py` | ~160 | ROI 选择、质量门控 |
| `vision/visualizer.py` | ~250 | 叠加层渲染（框、字幕、状态栏、输入栏） |
| `vision/camera.py` | ~95 | 摄像头线程、最新帧缓冲 |
| `voice/tts_server.py` | ~290 | 独立 TTS HTTP 服务 |
| `voice/tts.py` | ~230 | 异步 TTS 客户端、mock 模式 |
| `voice/asr.py` | ~270 | Paraformer，兼容两种模型布局 |
| `nlu/intent.py` | ~140 | 6 类意图正则解析 |
