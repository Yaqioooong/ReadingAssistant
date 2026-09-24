"""模型自由文本的出口清洗（工具参数字符串是唯一没有护栏的通道）。

## 为什么要单独一个模块

``final_answer`` 的 ``answer`` 与 ``request_clarification`` 的 ``missing_info``
都是**模型自由生成的字符串**，取出来直接进 API 响应、写进 ``chat_messages`` 表、
写进 QA 缓存。它既没有 schema 校验也没有格式约束 —— 于是模型侧的任何格式残渣
都会原样落到用户眼前，并且**存进库里**。

## 实测（2026-09-24，用户贴回来的回答）

回答正文末尾带着 ``</answer>\\n</invoke>\\n``。全仓 grep（含 ``config/prompt.yml``）为空
→ **不是我们写进去的**，是模型自己附加在工具参数末尾的。
库里同一条形态共 3 条（``id=2874`` 是回答正文、``id=2787/2861`` 是「信息不足」说明文本）
—— **两个工具都中招**，所以这不是某个工具的偶发问题，而是「工具参数 = 不可信自由文本」这一类问题。

## 危害不止是难看：它会自我强化

带标签的文本写进 ``chat_messages`` 之后，此后每一轮的 prompt 历史里都带着它，
模型继续模仿自己上一轮的格式（``logs/qa_20260921.log`` 的 ``rendered history``
里能直接看到带标签的历史被回放）。所以清洗必须**双向**：
出口（写库/返回）拦住新的，入口（组 prompt 时）中和存量。

## 为什么只剥首尾、只认已知标签

回答正文里会出现 ``[1]``、引号、破折号，也可能引用含尖括号的原文。
在正文中间做「去掉所有 <...>」的正则会误伤正文。所以判据收得很紧：
**只在首尾**、**只剥已知的调用外壳标签名**。宁可漏，不可误伤。
"""
from __future__ import annotations

import re

from reading_assistant.utils.logger_handler import get_logger

logger = get_logger('answer_text')

# 调用外壳的标签名。只收「明显是外层包裹」的名字 —— 收得越宽越容易误伤正文。
_HARNESS_TAGS: tuple[str, ...] = (
    'answer',
    'invoke',
    'tool_call',
    'tool_response',
    'function_call',
    'function',
    'parameter',
    'result',
    'output',
    'response',
)

# 允许命名空间前缀（真实外壳里常见形如 ``antml:invoke``）。
# 前缀形状写死成 ``[A-Za-z0-9_.:-]*``，这样 ``</随便什么>`` 不会被误当成外壳标签。
_TAG = r'</?(?:[A-Za-z0-9_.:-]*)(?:%s)>' % '|'.join(_HARNESS_TAGS)
_TRAILING = re.compile(r'(?:\s*%s)+\s*$' % _TAG)
_LEADING = re.compile(r'^\s*(?:%s\s*)+' % _TAG)


def sanitize_model_text(text: str | None) -> str | None:
    """剥掉模型附加在文本首尾的调用外壳标签；没有则**原样返回**。

    只动首尾、反复剥（模型可能叠好几层，实测形态就是 ``</answer>`` + ``</invoke>`` 两层）。
    正文中间的同类标签**不碰** —— 那是内容，不是残渣。

    两种输入**零改动**返回，保证这个函数可以无脑挂在所有出口上：
    - 不含 ``>`` 的文本（绝大多数回答）：连正则都不跑；
    - 一次都没剥到东西时返回**原对象**（不重新拼字符串，也就不会顺手改掉空白）。

    剥完为空时**回退原文**并告警：本函数只管格式，不做「这条回答是不是没内容」的判断，
    更不能把一个非空的回答变成空字符串（那会把判断权从调用方手里拿走）。
    """
    if not isinstance(text, str) or not text or '>' not in text:
        return text
    removed: list[str] = []
    body = text
    while True:
        match = _TRAILING.search(body)
        if match:
            removed.append(match.group(0).strip())
            body = body[: match.start()]
            continue
        match = _LEADING.search(body)
        if match:
            removed.append(match.group(0).strip())
            body = body[match.end():]
            continue
        break
    if not removed:
        return text
    cleaned = body.strip()
    if not cleaned:
        logger.warning('问答[出口清洗] 整条文本都是外壳标签，回退原文以免产生空回答：%s', removed)
        return text
    logger.info(
        '问答[出口清洗] 剥掉模型附加的调用外壳标签 %s（%d → %d 字）',
        removed,
        len(text),
        len(cleaned),
    )
    return cleaned
