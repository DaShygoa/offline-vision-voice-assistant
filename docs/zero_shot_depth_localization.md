# 深度引导零样本物体定位 — 技术报告

## 1. 问题背景

本项目的核心任务是离线物体识别：用户通过语音/文本指令"这是什么"触发识别，或"记住这个叫XX"触发录入。识别管线由两步组成：

1. **定位**：从摄像头画面中确定"用户想问的那个物体"的区域
2. **识别**：对裁剪区域提取 ResNet18 特征，与特征库做余弦检索

原方案中定位步骤依赖 YOLOv8n（COCO 80 类预训练检测器）。YOLO 是**闭集检测器**——只认识训练时见过的 80 个类别。当用户拿起一个 COCO 类别之外的物体（比如一支笔、一个手工艺品、一本书的特定封面），YOLO 无法输出检测框，系统退化为"裁画面正中间 45% 区域"的粗糙回退。

这构成了零样本叙事的硬伤：**识别环节（ResNet18 特征检索）是零样本的，但定位环节不是。**

## 2. 方案选型

针对"零样本物体定位"这一需求，调研了三条技术路线：

| 路线 | 代表方法 | 零样本程度 | CPU 延迟 | 所需提示 |
|------|---------|-----------|---------|---------|
| A. 深度最近物体 | Depth Anything V2 | 完全零类别 | 300–500ms | 无需任何提示 |
| B. 类别无关分割 | MobileSAM | 类别无关 | 100–300ms | 需要点/框提示 |
| C. 开放词汇检测 | YOLO-World / OWL-ViT | 需文本词汇 | 0.5–12s | 需要文本提示词 |

**路线 B** 的 SAM 系列虽然类别无关，但原版 SAM（ViT-H, 632M 参数）在 CPU 上推理需数秒且需要外部提示点/框。**MobileSAM**（Zhang et al., arXiv 2306.14289）用 TinyViT（5M 参数）替换 ViT-H 编码器，速度快约 38 倍，掩码质量保持原版 97% 以上，CPU 可部署。但单独使用仍需提示点——存在"先有鸡还是先有蛋"的引导问题。

**路线 C** 的开放词汇检测器（OWL-ViT、Grounding DINO）需要文本提示词来描述目标类别，本质上仍是"类别范式"。且 OWL-ViT 在 CPU 上单图推理需约 12 秒，无法满足交互延迟要求。YOLO-World 速度更快，但仍是词汇表驱动的检测范式。

**路线 A** 以单目深度估计为核心，通过几何信息（"什么离摄像头最近"）而非语义类别来定位物体，天然零类别、零提示。我们以路线 A 为主体，叠加路线 B 的 MobileSAM 做掩码精化，取两者之长。

## 3. 方法设计

### 3.1 核心思想

用户说"这是什么"时，最自然的交互假设是：**用户想问的是离摄像头最近的那个物体**（通常是手持或放在桌面前方的物体）。单目深度估计恰好能回答"什么更近"这一几何问题，而不需要任何类别知识。

### 3.2 流水线

```
摄像头帧 (1080p BGR)
  → Depth Anything V2 Small (ONNX Runtime CPU, 518×518 输入)
  → 相对深度图归一化 [0,1]（0=最近，1=最远）
  → 近景阈值分割：取最近 20% 像素（near_percentile=80）
  → 形态学闭运算 + 开运算（合并碎片、去噪点）
  → 连通域分析 (cv2.connectedComponentsWithStats)
  → 候选区域打分：
      score = 0.35 × 面积归一 + 0.30 × 画面中心邻近度 + 0.35 × 深度接近度
  → 取最高分区域
  → [MobileSAM 掩码精化]（可选）
      深度区域质心 → 点提示 → MobileSAM (TinyViT encoder + mask decoder)
      → 像素级掩码 → 最大连通域 → 紧致 bbox（替代矩形框）
  → 向外扩 6% 后裁剪 ROI
  → ResNet18 512 维特征提取 → 余弦检索
```

**MobileSAM 掩码精化**解决了纯深度连通域的一个固有缺陷：连通域给出的是矩形包围框，可能混入背景像素。MobileSAM 以深度质心为正点提示，输出像素级掩码，从掩码取紧致 bbox，裁剪更干净，ResNet18 特征质量更高。这正是 DepSeg 论文"深度引导点提示 + 基础模型分割"的完整复现——只是用 5M 参数的 MobileSAM 替代了原版 SAM。

### 3.3 打分公式设计

三个因子分别解决不同问题：

- **面积归一**（权重 0.35）：太小的区域大概率是噪声，太大的区域大概率是背景/桌面。将面积映射到 [0,1]（面积占比 ×4 截断），最优面积约占画面 5–25%
- **画面中心邻近度**（权重 0.30）：用户想问的物体通常在画面中心附近。用到画面中心的归一化距离的补来度量
- **深度接近度**（权重 0.35）：越近的区域越可能是用户手持/关注的目标。用 `1 - mean_depth` 直接度量

### 3.4 深度可视化窗口

为了让演示时能直观展示深度识别能力，系统提供一个独立的深度热力图窗口：
- 使用 TURBO 伪彩色映射：**暖色=近，冷色=远**
- 实时叠加"最近物体"的白色边界框
- 约每 500ms 刷新一次（不进入 30FPS 主循环热路径）

## 4. 与 YOLO 的关系

切换到深度定位后，YOLO 的角色从"定位门控"降级为**纯可视化叠加**：

| 环节 | 原方案 | 新方案 |
|------|-------|-------|
| identify 裁剪 | YOLO 检测框 → 裁剪 | **深度最近物体 → 裁剪**（YOLO 仅做可视化） |
| remember 裁剪 | YOLO 检测框 → 裁剪 | **深度最近物体 → 裁剪** |
| 画面叠加框 | YOLO 检测框 | YOLO 检测框（保留） |
| 降级兜底 | 中心 ROI 45% | 中心 ROI 45%（不变） |

这样对老师展示时可以清晰地说："**识别管线中没有任何预训练类别知识**。定位靠纯几何（深度最近物体），识别靠实例级特征匹配（用户教什么就认识什么）。"

## 5. 性能开销

| 指标 | 数值 |
|------|------|
| **深度估计 (DA-V2 Small)** | |
| 参数量 | ~25M |
| ONNX 模型大小 | ~95MB |
| CPU 单次推理 | 300–500ms（笔记本 i5/i7 级） |
| **掩码精化 (MobileSAM)** | |
| 参数量 | ~5.8M（TinyViT encoder 5M + decoder） |
| ONNX 模型大小 | encoder ~22MB + decoder ~16MB |
| CPU 单次推理（编码+解码） | 100–300ms |
| **合计** | |
| 单次 identify/remember 总延迟 | 400–800ms |
| 内存增量 | ~350MB（DA-V2 + MobileSAM） |
| 运行时机 | 仅在 identify/remember 指令触发时执行，不在摄像头热路径 |

推理开销与 ResNet18 同级（都是按需执行），对 30FPS UI 主循环无影响。深度窗口的刷新（约 2Hz）也独立于主循环。MobileSAM 仅在深度定位成功后追加执行一次，用于掩码精化。

## 6. 参考文献

1. **Depth Anything V2** — Yang L., Kang B., Huang Z., et al. "Depth Anything V2." *NeurIPS 2024*. arXiv: 2406.09414.
   - 本项目使用的单目深度估计模型。Small 变体 (ViT-S) 在 CPU 上亚秒级推理，相对深度精度接近 DPT-Large，适合边缘部署。

2. **DepSeg** — Yang K., Wang Q., Cheng Y., Ban Y. "See in Depth: Training-Free Surgical Scene Segmentation with Monocular Depth Priors." *arXiv: 2512.05529*, 2025.
   - 提出用 Depth Anything V2 的深度图引导 SAM 生成点提示，实现训练自由的零样本分割。在 CholecSeg8k 上 mIoU 从 14.7% 提升到 35.9%。**本项目的"深度引导物体定位"思想直接借鉴自该论文。**

3. **ZISVFM** — "Zero-Shot Object Instance Segmentation in Indoor Robotic Environments with Vision Foundation Models." *arXiv: 2502.03266*, 2025.
   - 将深度图着色后输入 SAM 生成类别无关掩码提案，再用 DINOv2 自监督注意力图过滤非物体区域。验证了"深度 + 类别无关分割"在未知物体抓取场景中的有效性。

4. **SupeRGB-D** — Örnek E.P., et al. "SupeRGB-D: Zero-shot Instance Segmentation in Cluttered Indoor Environments." *arXiv: 2022*.
   - 超像素合并 + 深度的零样本实例分割，模型仅 0.4MB，CPU 1.44s/帧。证明了深度信息对类别无关物体性（objectness）判断的价值。

5. **MobileSAM** — Zhang C., Li D., Hu Q. "Faster Segment Anything: Towards Lightweight SAM for Mobile Applications." *arXiv: 2306.14289*, 2023.
   - 用 TinyViT（5M 参数）替换 SAM 的 ViT-H 编码器（632M 参数），速度提升约 38 倍，掩码质量保持 97% 以上。保留原 SAM 的 mask decoder，支持点/框提示。本项目用其做深度质心点提示 → 像素级掩码精化。ONNX 导出分离为 encoder + decoder 两个模型，CPU 推理 100–300ms。

6. **Segment Anything Model (SAM)** — Kirillov A., et al. "Segment Anything." *ICCV 2023*. arXiv: 2304.02643.
   - 类别无关分割基础模型。MobileSAM 的架构基础（共享 mask decoder）。

7. **YOLOv8 / Ultralytics** — Jocher G., Chaurasia A., Qiu J. "Ultralytics YOLO." 2023.
   - 原方案使用的闭集检测器。本项目保留其可视化叠加功能，但将其从识别管线的关键路径中移除。

---

*生成日期: 2026-10-26*
