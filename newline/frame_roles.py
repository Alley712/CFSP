"""
frame_info.json → 框架-角色约束映射。

用途:
  - predict_task3.py: 解码时构建合法角色 mask
  - 验证脚本: 统计 top-K 覆盖率
"""

import json
import torch


def load_frame_info(frame_info_path):
    """加载 frame_info.json"""
    with open(frame_info_path, 'r', encoding='utf8') as f:
        return json.load(f)


def build_frame2roles(frame_info_path, role2idx):
    """
    构建 frame_id → 合法角色 id 集合的映射。

    Args:
        frame_info_path: frame_info.json 路径
        role2idx: {fe_name: idx} 角色名→索引 (与 dataset_task3.py label2idx 一致)

    Returns:
        frame2roles: dict, frame_id → set of legal role_ids
        frame2idx: dict, frame_name → frame_id
        idx2frame: dict, frame_id → frame_name
    """
    frame_info = load_frame_info(frame_info_path)

    frame2idx = {}
    idx2frame = {}
    for i, item in enumerate(frame_info):
        frame2idx[item['frame_name']] = i
        idx2frame[i] = item['frame_name']

    frame2roles = {}
    for i, item in enumerate(frame_info):
        legal_roles = set()
        for fe in item['fes']:
            fe_name = fe['fe_name']
            if fe_name in role2idx:
                legal_roles.add(role2idx[fe_name])
        frame2roles[i] = legal_roles

    return frame2roles, frame2idx, idx2frame


def build_legal_mask(frame_id, frame2roles, num_labels, illegal_val=float('-inf')):
    """
    为指定 frame 构建合法角色 mask (合法=0, 非法=illegal_val)。

    用法: logits = logits + mask  # 非法角色被压低
    """
    mask = torch.full((num_labels,), illegal_val)
    legal_ids = frame2roles.get(frame_id, set())
    for rid in legal_ids:
        mask[rid] = 0.0
    return mask


def build_legal_mask_batch(frame_ids, frame2roles, num_labels, illegal_val=float('-inf')):
    """batch 版本: 返回 (batch_size, num_labels) 的 mask tensor"""
    masks = []
    for fid in frame_ids:
        fid_int = fid.item() if isinstance(fid, torch.Tensor) else fid
        masks.append(build_legal_mask(fid_int, frame2roles, num_labels, illegal_val))
    return torch.stack(masks, dim=0)
