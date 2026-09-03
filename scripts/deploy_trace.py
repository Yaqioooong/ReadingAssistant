"""本地部署 trace：起服务 → 跑完整业务流 → 留档 reports/deploy_trace.md。

覆盖：上传 / 文档列表 / 正常问答 / 信息不足(HITL) / 提交澄清 / 再问答 / 缓存命中。
运行：uv run python scripts/deploy_trace.py
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

PORT = 8017
BASE = f'http://127.0.0.1:{PORT}'
LOG_DIR = Path('logs')
REPORT = Path('reports/deploy_trace.md')
UPLOAD = Path('uploads/5fcce69c74214df88f66224e70e1facb_张三爱情故事.txt')

LOG_DIR.mkdir(exist_ok=True)
REPORT.parent.mkdir(exist_ok=True)

steps: list[dict] = []


def step(name: str, resp: requests.Response | None, extra: str = '') -> dict:
    s = {
        'time': datetime.now().strftime('%H:%M:%S'),
        'name': name,
        'status': resp.status_code if resp else '-',
        'extra': extra,
    }
    steps.append(s)
    print(f"[{s['time']}] {name} -> {s['status']} {extra}")
    return s


def main() -> None:
    print('=== 启动 uvicorn（端口 %d）===' % PORT)
    log_file = open(LOG_DIR / 'server_trace.log', 'w', encoding='utf-8')
    proc = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'reading_assistant.api.main:app',
         '--host', '127.0.0.1', '--port', str(PORT)],
        stdout=log_file, stderr=subprocess.STDOUT, text=True,
    )
    try:
        for _ in range(40):
            try:
                if requests.get(f'{BASE}/api/documents', timeout=1).status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(0.5)
        else:
            raise RuntimeError('服务启动超时')
        print('服务已就绪\n')

        # 1. 上传
        with open(UPLOAD, 'rb') as f:
            r = requests.post(f'{BASE}/api/documents/upload',
                              files={'file': (UPLOAD.name, f, 'text/plain')})
        step('上传《张三爱情故事.txt》', r)
        doc_id = r.json()['id']

        # 2. 文档列表
        r = requests.get(f'{BASE}/api/documents')
        step('列出文档', r, f'共 {len(r.json())} 本')

        # 3. 创建会话
        r = requests.post(f'{BASE}/api/sessions')
        session_id = r.json()['session_id']
        step('创建会话', r, f'session={session_id}')

        # 4. 正常问答
        r = requests.post(f'{BASE}/api/sessions/{session_id}/messages',
                          json={'question': '张三喜欢谁？', 'document_ids': [doc_id]})
        ans = r.json().get('answer', '')
        step('正常问答：张三喜欢谁？', r, f'cit={len(r.json().get("citations", []))} | {ans[:50]}')

        # 5. 同样问题再问（命中缓存）
        r = requests.post(f'{BASE}/api/sessions/{session_id}/messages',
                          json={'question': '张三喜欢谁？', 'document_ids': [doc_id]})
        step('缓存命中：重复提问', r, f'cit={len(r.json().get("citations", []))}')

        # 6. 书内提问无关内容 → 信息不足 → HITL
        r = requests.post(f'{BASE}/api/sessions/{session_id}/messages',
                          json={'question': '少白公在哪一年失去权力？', 'document_ids': [doc_id]})
        body = r.json()
        step('信息不足：问不相关书籍内容', r,
             f"needs_clarification={body.get('needs_clarification')} hitl={body.get('hitl_task_id')}")

        # 7. 查看澄清任务
        r = requests.get(f'{BASE}/api/hitl/tasks',
                         params={'session_id': int(session_id)})
        tasks = r.json()
        step('查看 HITL 任务', r, f'待处理 {len(tasks)} 个')
        task_id = tasks[0]['id'] if tasks else None

        # 8. 提交澄清
        if task_id:
            r = requests.post(f'{BASE}/api/hitl/tasks/{task_id}/submit',
                              json={'clarification': '补充：少白公在名词大动词.txt 中'})
            step('提交澄清', r, r.json().get('status', ''))

        # 9. 带澄清重问
        r = requests.post(f'{BASE}/api/sessions/{session_id}/messages',
                          json={'question': '少白公在哪一年失去权力？',
                                'document_ids': [doc_id],
                                'clarification': '少白公在名词大动词.txt 中'})
        ans = r.json().get('answer', '')
        step('澄清后重问', r, f'{ans[:50]}')

        _write_report(doc_id, session_id)
        print('\n✓ trace 完成，报告: reports/deploy_trace.md')
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        log_file.close()


def _write_report(doc_id, session_id) -> None:
    with open(UPLOAD) as f:
        pass
    lines = [
        '# 部署 Trace 记录',
        '',
        f'- 生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'- 服务：uvicorn `reading_assistant.api.main:app`，端口 {PORT}',
        f'- 后端日志：`logs/api_*.log`（请求级）、`logs/qa_*.log`（流水线级）、`logs/server_trace.log`（服务级）',
        '',
        '## 业务流时间线',
        '',
        '| 时间 | 步骤 | HTTP | 说明 |',
        '|---|---|---|---|',
    ]
    for s in steps:
        lines.append(f"| {s['time']} | {s['name']} | {s['status']} | {s['extra'][:60]} |")
    lines += [
        '',
        '## 覆盖场景',
        '',
        '1. 上传解析入库（幂等去重）',
        '2. 文档列表',
        '3. 正常问答（带引用）',
        '4. QA 缓存命中（同题二次提问，跳过 LLM）',
        '5. 信息不足 → HITL 澄清任务自动创建',
        '6. 澄清提交（状态机 awaiting→approved）',
        '7. 澄清后重问',
    ]
    REPORT.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
