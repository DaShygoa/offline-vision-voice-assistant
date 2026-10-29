"""Offline Chinese rule-based intent parser.

No LLM, no network. Patterns cover the demo dialogues from the spec:

  identify : 这是什么？ / 你看这是什么 / 认识这个吗
  remember : 记住，这个叫我的蓝色水杯 / 把它叫做X / 记住X
  search   : 帮我找我的蓝色水杯 / X在哪里 / 找一下X
  forget   : 忘记X / 删掉X
  list     : 你记住了什么 / 列出记忆
  help     : 你能做什么
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class IntentType(str, Enum):
    IDENTIFY = "identify"
    REMEMBER = "remember"
    SEARCH = "search"
    FORGET = "forget"
    LIST = "list"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    type: IntentType
    label: str = ""
    raw_text: str = ""
    confidence: float = 0.0

    @property
    def is_known(self) -> bool:
        return self.type != IntentType.UNKNOWN and self.confidence >= 0.6


# order matters: more specific patterns first
_LAB = r"[一-龥A-Za-z0-9_·\-]{1,32}"

_REMEMBER_PATTERNS = [
    # 记住这个 / 记住它 / 记住这东西（无名字，触发"问名字"流程）
    re.compile(r"^记住\s*[，,、]?\s*(?:这个|它|这东西|这个东西)$"),
    # 记住这个是/叫 XXX · 记住这是XXX · 记住是/叫 XXX · 记住，这个叫XXX
    # (口语里常用「是」而非「叫」：记住这个是充电宝)
    re.compile(rf"记住[，,、\s]*(?:这个|它|这东西|这个东西|这)?\s*(?:是|叫|就是|称为|叫做|名字是|名字叫)\s*(?P<label>{_LAB})"),
    # 把这个/这/它 叫做 XXX
    re.compile(rf"把(?:这个|这|它)\s*(?:叫做|叫|称为|命名为|起名为|是)\s*(?P<label>{_LAB})"),
    # 这是 XXX / 这叫 XXX —— 陈述句式录入（对着物体说出名字）
    # lookahead 排除疑问词，让「这是什么」落到 IDENTIFY
    re.compile(rf"^这\s*(?:是|就是|就叫|叫|叫做)\s*(?P<label>(?!什么|啥|哪个|哪)[一-龥A-Za-z0-9_·\-]{{1,32}})$"),
    # 记住 XXX（无「叫」）— 排除「记住了什么/哪些」和「这个/它/这是」
    re.compile(rf"记住\s*[，,、]?\s*(?P<label>(?!这是|这个|它|这东西|什么|哪些)[一-龥A-Za-z0-9_·\-]{{1,32}})\s*$"),
    # 这个是/叫 XXX，记住
    re.compile(rf"(?:这个|它)\s*(?:叫|是|叫做)\s*(?P<label>{_LAB})\s*[，,]?\s*记住"),
    # 名字叫 XXX
    re.compile(rf"名字(?:叫|是)\s*(?P<label>{_LAB})"),
]

_SEARCH_PATTERNS = [
    re.compile(r"(?:请|帮我|麻烦)?\s*(?:找一?下?|寻找|查找|搜索|搜一下)\s*(?P<label>[一-龥A-Za-z0-9_·\-]{1,40})"),
    re.compile(r"(?P<label>[一-龥A-Za-z0-9_·\-]{1,40}?)(?:在)?哪里\s*$"),
    re.compile(r"哪里(?:有|可以找到)\s*(?P<label>[一-龥A-Za-z0-9_·\-]{1,40})"),
    re.compile(r"(?:帮我)?(?:找找|找到)\s*(?P<label>[一-龥A-Za-z0-9_·\-]{1,40})"),
]

_IDENTIFY_PATTERNS = [
    re.compile(r"这是(?:什么|啥|什么东西|啥东西|啥玩意)"),
    re.compile(r"你看(?:看)?这是(?:什么|啥)"),
    re.compile(r"认识(?:这个|它)吗"),
    re.compile(r"这是谁"),
    re.compile(r"(?:识别|看一下|帮我看看|你看看)(?:这个|它|一下)?\s*$"),
    re.compile(r"什么(?:东西|物品)"),
]

_FORGET_PATTERNS = [
    re.compile(r"(?:请)?(?:忘记|忘掉|删除|删掉|移除)\s*(?P<label>[一-龥A-Za-z0-9_·\-]{1,40})"),
]

_LIST_PATTERNS = [
    re.compile(r"(?:你)?(?:记住了|记住的|记忆里)(?:什么|哪些|啥)"),
    re.compile(r"(?:列出|显示|查看)(?:你的)?(?:记忆|已记住|物品)"),
    re.compile(r"记忆列表|物品列表"),
]

_HELP_PATTERNS = [
    re.compile(r"你能做什么|会做什么|有什么功能|怎么用|帮助|help"),
]


def _clean_label(label: str) -> str:
    """Strip filler prefixes so '这是牛奶瓶' / '这个是充电宝' yield the bare name."""
    label = (label or "").strip("，,、。！？!? \t")
    label = re.sub(r"^(这是|这个是|那个是|它是|它就是|是个|就是|是)\s*", "", label)
    return label.strip("，,、。！？!? \t")


def _try(patterns, text: str) -> Optional[re.Match]:
    for p in patterns:
        m = p.search(text)
        if m:
            return m
    return None


# Common ASR mis-transcriptions that flip the intent word. Paraformer sometimes
# hears 记住 as 技术/记住/即是 — normalise before pattern matching.
_ASR_FIX = [
    (re.compile(r"^技术"), "记住"),
    (re.compile(r"^即是"), "记住"),
    (re.compile(r"技术(?=这个|它|是|叫)"), "记住"),
    # redundant filler words: "记住这个东西这个是手柄" → "记住这个是手柄"
    (re.compile(r"这个东西这个"), "这个"),
    (re.compile(r"这个东西(?=是|叫)"), "这个"),
    (re.compile(r"那个东西那个"), "那个"),
    (re.compile(r"这东西这个"), "这个"),
    # repeated demonstratives: "记住这个这个是牛奶瓶" → "记住这个是牛奶瓶"
    # (+) collapses any run of repeats in one pass
    (re.compile(r"(这个)+"), "这个"),
    (re.compile(r"(那个)+"), "那个"),
    (re.compile(r"(它)+"), "它"),
    (re.compile(r"(是)+"), "是"),
    (re.compile(r"(叫)+"), "叫"),
    # duplicated single 「这」 before a verb: "记住这个这是手机" → "记住这个是手机"
    (re.compile(r"(这个)这(?=是|叫)"), "这个"),
    (re.compile(r"(这)这(?=是|叫)"), "这"),
    (re.compile(r"(这)是(?=这)"), "这"),
]

def parse_intent(text: str) -> Intent:
    raw = (text or "").strip()
    # 1) remove punctuation noise
    t = re.sub(r"[。！？!?，,、；;：:…\-_\"'“”‘’()（）\[\]【】]+", "", raw)
    # 2) remove ALL whitespace — Paraformer often inserts spaces between CJK chars
    #    e.g. "这 是 什 么" → "这是什么" (patterns expect no spaces)
    t = re.sub(r"\s+", "", t).strip()
    if not t:
        return Intent(IntentType.UNKNOWN, raw_text=raw, confidence=0.0)
    # fix ASR mis-hearings before matching (记住 ← 技术/即是). Run to a fixed
    # point: a phrase like "记住这个这这是手机" needs several passes to collapse.
    for _ in range(4):
        before = t
        for pat, repl in _ASR_FIX:
            t = pat.sub(repl, t)
        if t == before:
            break

    # LIST before REMEMBER — 「你记住了什么」 would otherwise match 记住…
    m = _try(_LIST_PATTERNS, t)
    if m:
        return Intent(IntentType.LIST, raw_text=raw, confidence=0.9)

    # REMEMBER — check before SEARCH (「记住，这个叫…」 contains 叫)
    m = _try(_REMEMBER_PATTERNS, t)
    if m:
        label = _clean_label(m.groupdict().get("label") or "")
        if label:
            return Intent(IntentType.REMEMBER, label=label, raw_text=raw, confidence=0.95)
        # "记住这个" without a name → ask for name handled by app with IDENTIFY-like teaching
        return Intent(IntentType.REMEMBER, label="", raw_text=raw, confidence=0.7)

    m = _try(_SEARCH_PATTERNS, t)
    if m:
        label = _clean_label(m.groupdict().get("label") or "")
        if label.endswith("在") and not label.endswith("还在"):
            label = label[:-1].strip()
        if label and not re.fullmatch(r"这是什么|什么|哪", label):
            return Intent(IntentType.SEARCH, label=label, raw_text=raw, confidence=0.9)

    m = _try(_FORGET_PATTERNS, t)
    if m:
        label = _clean_label(m.groupdict().get("label") or "")
        return Intent(IntentType.FORGET, label=label, raw_text=raw, confidence=0.9)

    m = _try(_IDENTIFY_PATTERNS, t)
    if m:
        return Intent(IntentType.IDENTIFY, raw_text=raw, confidence=0.92)

    m = _try(_HELP_PATTERNS, t)
    if m:
        return Intent(IntentType.HELP, raw_text=raw, confidence=0.9)

    return Intent(IntentType.UNKNOWN, raw_text=raw, confidence=0.3)
