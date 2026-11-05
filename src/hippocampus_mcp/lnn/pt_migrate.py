# -*- coding: utf-8 -*-
"""legacy ``lnn_state.pt`` → npz 一次性迁移（在**独立子进程**里跑）.

为什么是子进程：读 ``.pt`` 必须有 torch；主进程（MCP server）的目标是
**零 torch 常驻**。本模块只被 ``StateCheckpoint._load_legacy_pt`` 以
``python -m hippocampus_mcp.lnn.pt_migrate <src.pt> <dst.npz>`` 方式拉起，
读完即退，torch 永不进入主进程。

迁移语义与内存格式无关：``hx`` 会随权重漂移（历史库权重每次重启随机），
本迁移只是**保底把状态搬过来**；读不出来就安全重置，不影响任何记忆数据。
"""

from __future__ import annotations

import faulthandler
import sys

# 子进程若在父进程（server）特定上下文里意外卡住，30s 后把自身线程栈
# 打到 stderr（父进程 capture_output 会收到），现场自暴露，不再黑盒猜。
faulthandler.dump_traceback_later(30, exit=False)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("用法: python -m hippocampus_mcp.lnn.pt_migrate <src.pt> <dst.npz>",
              file=sys.stderr)
        return 2
    src, dst = argv
    try:
        import torch

        data = torch.load(src, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"pt_migrate: 读取失败 ({src}): {exc}", file=sys.stderr)
        return 1

    from .checkpoint import payload_from_state

    state = dict(data or {})
    # get_state 之外的键（step_count / reference_time 等）payload_from_state
    # 会收进 scalars_json；hx/last_output/window 转数组。
    try:
        payload = payload_from_state(state)
        import numpy as np

        np.savez(dst, **payload)
    except Exception as exc:
        print(f"pt_migrate: 写入失败 ({dst}): {exc}", file=sys.stderr)
        return 1
    print(f"pt_migrate: {src} → {dst} OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
