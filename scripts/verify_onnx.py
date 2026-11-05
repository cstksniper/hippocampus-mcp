#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验 ONNX 嵌入器与 torch（sentence-transformers）版向量是否一致。

这是 H2 的**准入门槛**：cos 达不到 0.9999 就不能拿 ONNX 上生产，
否则等于悄悄换了一套向量——历史记忆的向量与新写入的向量不在一个空间里，
检索会静默变差，而且没有任何报错。

用法::

    python scripts/verify_onnx.py
    python scripts/verify_onnx.py --model /path/to/bge-base-zh-v1.5

验收标准：
  * 逐句 cos >= 0.9999（含空串、超长截断、中英混排、emoji）
  * 查询侧（带 BGE 前缀）同样 >= 0.9999
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

COS_THRESHOLD = 0.9999

# 覆盖：空串 / 短句 / 长句 / 超长截断 / 中英混排 / 数字 / emoji / 重复
SAMPLES = [
    "",
    "你好",
    "记忆系统的向量一致性校验",
    "这是一段用于校验的中等长度中文文本，包含多个从句，用来观察句向量是否稳定。",
    "量子计算与机器学习交叉方向的最新进展综述",
    "The quick brown fox jumps over the lazy dog.",
    "2026年9月26日 23:59 记录：H2 ONNX 替换验收",
    "😀 emoji 与特殊字符 !@#$%^&*() 混排测试",
    "深度学习" * 200,          # 远超 512 token → 触发截断路径
    "A" * 3000,                 # 纯长文本，同样触发截断
    "重复重复重复重复重复重复重复重复重复重复",
]


def cos(a, b) -> float:
    import numpy as np

    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0 if na < 1e-12 and nb < 1e-12 else 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    from hippocampus_mcp.engine.embeddings import SentenceTransformerEmbedder
    from hippocampus_mcp.engine.onnx_embedder import OnnxEmbedder

    if args.model:
        model_dir = os.path.abspath(args.model)
        model_name = model_dir
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        model_dir = os.path.join(
            os.path.dirname(root), "libs", "models", "bge-base-zh-v1.5"
        )
        model_name = "BAAI/bge-base-zh-v1.5"

    if not os.path.isdir(model_dir):
        print(f"[FAIL] 模型目录不存在: {model_dir}", file=sys.stderr)
        return 1

    print(f"模型目录: {model_dir}")
    print("[1/3] 加载 torch(sentence-transformers) 版 …")
    # 🔴 反假绿（实测踩中）：``SentenceTransformerEmbedder`` 在
    # ``auto`` 后端下会自动挂上 ONNX ⇒ 所谓"torch 版"其实是 ONNX 版，
    # 门槛退化成 **ONNX vs ONNX**，cos 恒等于 1.000000 而**什么都没验到**
    # （项目史：int8 假绿就是同款形态）。所以这里必须**强制** torch 后端，
    # 并且下面用 ``_onnx is None`` 再钉一道——拿不到真 torch 就报错，不许放行。
    _prev_backend = os.environ.get("HIPPOCAMPUS_EMB_BACKEND")
    os.environ["HIPPOCAMPUS_EMB_BACKEND"] = "torch"
    t0 = time.time()
    try:
        torch_emb = SentenceTransformerEmbedder(model_name=model_name)
        torch_emb.embed("预热")
    finally:
        if _prev_backend is None:
            os.environ.pop("HIPPOCAMPUS_EMB_BACKEND", None)
        else:
            os.environ["HIPPOCAMPUS_EMB_BACKEND"] = _prev_backend
    if getattr(torch_emb, "_onnx", None) is not None:
        print("[FAIL] torch 侧被解析成了 ONNX 后端：本门槛将退化为 ONNX vs ONNX，"
              "cos 恒为 1.0 而毫无验证意义。请检查 HIPPOCAMPUS_EMB_BACKEND 是否被覆盖。",
              file=sys.stderr)
        return 1
    print(f"      耗时 {time.time() - t0:.1f}s  dim={torch_emb.dim}")

    print("[2/3] 加载 ONNX 版 …")
    t0 = time.time()
    onnx_emb = OnnxEmbedder(model_dir=model_dir, model_name=model_name)
    onnx_emb.embed("预热")
    print(f"      耗时 {time.time() - t0:.1f}s  dim={onnx_emb.dim}")

    if torch_emb.dim != onnx_emb.dim:
        print(f"[FAIL] 维度不一致: torch={torch_emb.dim} onnx={onnx_emb.dim}", file=sys.stderr)
        return 1

    print("[3/3] 逐句比对余弦相似度")
    worst = 1.0
    worst_text = ""
    fails = []
    for s in SAMPLES:
        a = torch_emb.embed(s)
        b = onnx_emb.embed(s)
        c = cos(a, b)
        label = (s[:24] + "…") if len(s) > 24 else (s or "<空串>")
        flag = "OK " if c >= COS_THRESHOLD else "LOW"
        print(f"  [{flag}] cos={c:.6f}  {label}")
        if c < COS_THRESHOLD:
            fails.append((label, c))
        if c < worst:
            worst, worst_text = c, label

    # 查询侧（带 BGE 前缀）也要比
    q = "记忆系统如何做多跳检索"
    cq = cos(torch_emb.embed_query(q), onnx_emb.embed_query(q))
    print(f"  [{'OK ' if cq >= COS_THRESHOLD else 'LOW'}] cos={cq:.6f}  查询侧(带前缀) {q!r}")
    if cq < COS_THRESHOLD:
        fails.append(("查询侧", cq))
    worst = min(worst, cq)

    # 批量路径
    tb = torch_emb.embed_batch(SAMPLES[:6])
    ob = onnx_emb.embed_batch(SAMPLES[:6])
    cb, cb_i = min(
        ((cos(tb[i], ob[i]), i) for i in range(len(SAMPLES[:6]))),
        key=lambda x: x[0],
    )
    cb_text = (SAMPLES[cb_i][:24] or "<空串>")
    print(f"  [{'OK ' if cb >= COS_THRESHOLD else 'LOW'}] cos={cb:.6f}  批量 embed_batch 最差值: {cb_text}")
    if cb < COS_THRESHOLD:
        fails.append((f"批量:{cb_text}", cb))
    if cb < worst:
        worst, worst_text = cb, cb_text

    print()
    if fails:
        print(f"[FAIL] {len(fails)} 项低于 {COS_THRESHOLD}，最低={worst:.6f} ({worst_text})")
        return 1
    print(f"[PASS] 全部 >= {COS_THRESHOLD}，最低={worst:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
