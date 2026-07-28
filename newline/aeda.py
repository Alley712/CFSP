"""
AEDA (An Easier Data Augmentation) — 随机标点插入。

对中文句子随机插入标点符号，增强文本表面多样性，同时保持
frame / 论元边界 / 角色标签不变。插入位置排除目标词和论元
内部，防止标注被破坏。
"""

import random

# 中文标点优先，兼容英文标点
AEDAP_CHARS = ["。", "；", "？", "：", "！", "，"]


def aeda_augment(text, target_start, target_end, spans,
                 num_insert=3, chars=None):
    """
    对句子进行 AEDA 增强，返回增强文本和修正后的坐标。

    Args:
        text: 原始句子（字符串）
        target_start: 目标词起始位置（字符级，闭区间）
        target_end: 目标词结束位置（字符级，闭区间）
        spans: 论元 span 列表 [(start, end), ...]（字符级，闭区间）
        num_insert: 插入标点数量
        chars: 标点候选池，默认使用中文标点池

    Returns:
        new_text: 增强后的文本
        new_target: [start, end] 修正后的目标词位置
        new_spans: [(start, end), ...] 修正后的 span 列表
    """
    if chars is None:
        chars = AEDAP_CHARS

    # 1. 计算禁区（目标词内部 + 所有论元内部）
    forbidden = set()
    # 阻止在目标词内部插入：多字词的字间位置不可拆分
    for i in range(target_start + 1, target_end + 1):
        forbidden.add(i)
    # 论元内部：同理，阻止在论元字间插入
    for s, e in spans:
        for i in range(s + 1, e + 1):
            forbidden.add(i)

    # 2. 从安全位置中随机选择（不含位置 0=句首，含 len(text)=句尾）
    safe_positions = [i for i in range(1, len(text) + 1)
                      if i not in forbidden]
    if len(safe_positions) == 0:
        return text, [target_start, target_end], spans

    n = min(num_insert, len(safe_positions))
    positions = sorted(random.sample(safe_positions, n))

    # 3. 插入标点，同时跟踪累计偏移
    new_text_chars = list(text)
    for i, pos in enumerate(positions):
        punct = random.choice(chars)
        new_text_chars.insert(pos + i, punct)  # +i 是因为前面已插入的字符

    new_text = ''.join(new_text_chars)

    # 4. 计算坐标偏移：对每个插入位置，统计在 target/span 之前的数量
    def offset(pos):
        return sum(1 for p in positions if p <= pos)

    new_target = [target_start + offset(target_start),
                  target_end + offset(target_end)]

    new_spans = [[s + offset(s), e + offset(e)] for s, e in spans]

    return new_text, new_target, new_spans
