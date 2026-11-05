# -*- coding: utf-8 -*-
"""通用 LLM judge 客户端 —— 评测层与推理服务商之间的**唯一边界**。

## 为什么有这个文件

评测需要一个大模型当裁判（LLM-as-judge）。但"用哪家的模型"是**可替换的实现细节**，
不该渗进项目本体。所以这里把评测能力抽象成三件环境变量：

    JUDGE_BASE_URL    必填，OpenAI 兼容的 /chat/completions 端点
    JUDGE_API_KEY     必填
    JUDGE_MODEL       必填，模型标识

只要提供 OpenAI 兼容接口，自建推理、本地部署、任何商用服务都能当 judge。
**换 judge 只改环境变量，不改一行代码。**

## 硬约束

1. **本模块不出现任何推理服务商名称**，也不内置任何默认端点或默认模型。
   ——没有"某厂商专用分支"这种后门。
2. **缺配置时显式报错**，不静默回退到某个内置服务。静默用错默认值比缺省报错危险得多。
3. `src/` 核心包**完全不依赖本模块**——记忆服务本体不需要 LLM 也能跑。

## 用法

    from judge_client import JudgeClient

    judge = JudgeClient.from_env()
    text = judge.chat([{"role": "user", "content": "回答问题：..."}])

零第三方依赖（标准库 urllib），clone 下来即可用。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

try:
    # 可选：同目录存在 _load_env 时，顺带加载仓库根 .env。
    # 公开导出时本文件可能被单独带走，缺 _load_env 就用纯环境变量，不影响。
    import _load_env  # noqa: F401
except ImportError:
    pass

__all__ = ["JudgeClient", "JudgeConfigError", "JudgeRequestError"]

# 环境变量名 —— 全项目评测侧的唯一真相源，别在别处再拼字面量
ENV_BASE_URL = "JUDGE_BASE_URL"
ENV_API_KEY = "JUDGE_API_KEY"
ENV_MODEL = "JUDGE_MODEL"


class JudgeConfigError(RuntimeError):
    """judge 配置缺失或不完整。故意不做兜底——缺什么就报什么。"""


class JudgeRequestError(RuntimeError):
    """judge 调用失败（网络、HTTP 错误、返回体格式不符）。"""


class JudgeClient:
    """一个极小的 OpenAI 兼容 chat 客户端。

    刻意不引入 SDK：评测应当能被任何人用任何服务复现，
    而 SDK 会把"用哪家"这件事重新写回代码里。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_backoff: float = 2.0,
    ) -> None:
        if not base_url:
            raise JudgeConfigError(f"{ENV_BASE_URL} 为空——评测需要 judge 端点")
        if not api_key:
            raise JudgeConfigError(f"{ENV_API_KEY} 为空——评测需要 judge 凭据")
        if not model:
            raise JudgeConfigError(f"{ENV_MODEL} 为空——评测需要指定 judge 模型")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    # ---------------------------------------------------------------- 构造

    @classmethod
    def from_env(cls, *, prefix: str = "") -> "JudgeClient":
        """从环境变量构造。

        Args:
            prefix: 可选的变量名前缀，用于在同一台机器上并存多套 judge 配置，
                例如 prefix="JUDGE_ALT_" 会读 JUDGE_ALT_BASE_URL 等。
        """
        return cls(
            base_url=os.environ.get(prefix + ENV_BASE_URL, ""),
            api_key=os.environ.get(prefix + ENV_API_KEY, ""),
            model=os.environ.get(prefix + ENV_MODEL, ""),
        )

    @classmethod
    def is_configured(cls) -> bool:
        """三件套是否齐备——供评测脚本做前置检查、或 SKIP 需要 judge 的用例。"""
        return all(
            os.environ.get(name)
            for name in (ENV_BASE_URL, ENV_API_KEY, ENV_MODEL)
        )

    # ---------------------------------------------------------------- 调用

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """单轮 chat 调用，返回 assistant 文本。

        评测一律 temperature=0：判定要可复现，不要随机性。
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra:
            payload.update(extra)

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                return self._extract_text(raw)
            except urllib.error.HTTPError as e:  # 4xx/5xx：读回包体便于定位
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:400]
                except Exception:
                    pass
                last_err = JudgeRequestError(
                    f"judge HTTP {e.code}（第 {attempt + 1} 次）：{detail}"
                )
                # 4xx 是请求本身的问题，重试无意义
                if 400 <= e.code < 500:
                    raise last_err from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = JudgeRequestError(f"judge 网络失败（第 {attempt + 1} 次）：{e}")
            except ValueError as e:  # 返回体不是预期结构
                raise JudgeRequestError(f"judge 返回体解析失败：{e}") from e

            if attempt < self.max_retries:
                time.sleep(self.retry_backoff * (attempt + 1))

        raise last_err or JudgeRequestError("judge 调用失败：原因未知")

    # ---------------------------------------------------------------- 内部

    @staticmethod
    def _extract_text(raw: str) -> str:
        """从 OpenAI 兼容返回体里取 assistant 文本。

        刻意**不做多格式猜测**——拿不到就报错并带上原始片段，
        免得评测在"静默返回空串"的情况下给出假分数。
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise JudgeRequestError(f"judge 返回的不是 JSON：{raw[:200]}") from e

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise JudgeRequestError(
                f"judge 返回体缺少 choices[0].message.content：{raw[:200]}"
            ) from e


# ------------------------------------------------------------------ 自检

if __name__ == "__main__":
    if not JudgeClient.is_configured():
        missing = [
            n
            for n in (ENV_BASE_URL, ENV_API_KEY, ENV_MODEL)
            if not os.environ.get(n)
        ]
        raise SystemExit(
            "judge 未配置，缺：" + ", ".join(missing) + "\n"
            "设置这三个环境变量即可（见 .env.example）。\n"
            "注意：核心记忆服务不需要 judge，只有评测脚本需要。"
        )

    client = JudgeClient.from_env()
    print(f"judge 已配置：model={client.model} endpoint={client.base_url}")
    print("连通性测试 ->", client.chat([{"role": "user", "content": "回复两个字：正常"}]))
