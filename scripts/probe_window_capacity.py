"""窗口容量对照：真实长度消息下，能带进 prompt 的**原文**有多少。

旧方案：HISTORY_TURNS=6 条，单条 _RENDER_MSG_CAP=240 字符。
新方案：history_token_budget=4000 token，单条 history_msg_token_cap=800 token。
"""
from reading_assistant.config import get_settings
from reading_assistant.graph.qa import _count_tokens, _render_history, _select_window

# 真实形态：用户短问、助手长答（含引用标注），交替
USER = '小雷音寺那一回里，黄眉大王到底是什么来历？'
ASSISTANT = (
    '黄眉大王本是弥勒佛座下司磬的童儿，偷了人种袋和金铙下界为妖[1]。'
    '他在小雷音寺假变佛祖，设下瓜田诱唐僧师徒入彀，把孙悟空扣在金铙里[2]。'
    '后来孙悟空请来弥勒佛，用瓜田之计将他收服，人种袋与金铙也一并收了回去[3]。'
    '原文写他“本是弥勒佛面前司磬的一个黄眉童儿”，来历交代得很清楚[4]。'
    '这一段的关键在于：黄眉的“法宝”其实都是弥勒佛的随身之物，'
    '所以悟空请遍天兵天将都奈何不得，最后只能去求弥勒佛本人出面[5]。'
    '收服之后弥勒佛并未严惩，只叫他随自己回去，这也照应了“司磬童儿”的身份[6]。'
    '另需注意：此难与前面黄风岭、平顶山两处不同，妖怪的来历直接点名了佛门[7]。'
    '原文以“弥勒”二字点破，是很典型的《西游记》“妖魔有主”写法[8]。'
    '从叙事结构看，这一回也在为后文狮驼岭等情节铺垫佛门内部的纵容主题[9]。'
    '因此回答“来历”时，务必落到“弥勒佛座下司磬童儿”这一句上，不要泛泛而谈[10]。'
)
print(f'单条消息长度：用户 {len(USER)} 字符 / 助手 {len(ASSISTANT)} 字符')
print(f'单条 token 估算：用户 {_count_tokens(USER)} / 助手 {_count_tokens(ASSISTANT)}')

rows = []
for i in range(20):
    rows.append(('user', f'{USER}（第{i+1}轮）'))
    rows.append(('assistant', f'{ASSISTANT}（第{i+1}轮）'))

settings = get_settings()
print(f'\n配置：budget={settings.history_token_budget} token, '
      f'msg_cap={settings.history_msg_token_cap} token')

# 新方案
picked = _select_window(rows, settings.history_token_budget)
rendered = _render_history(picked)
print(f'\n【新方案】保留 {len(picked)} 条消息 / {len(rows)} 条候选')
print(f'        注入 prompt 的原文 {len(rendered)} 字符 ≈ {_count_tokens(rendered)} token')

# 旧方案（复刻其语义：最近 6 条，单条截到 240 字符）
old_rows = rows[-6:]
old_rendered = '\n'.join(
    f"{'用户' if r == 'user' else '助手'}：{(c[:240] + '…') if len(c) > 240 else c}"
    for r, c in old_rows
)
print(f'\n【旧方案】保留 {len(old_rows)} 条消息（且窗口内**全部**助手长答被截断到 240 字符）')
print(f'        注入 prompt 的原文 {len(old_rendered)} 字符 ≈ {_count_tokens(old_rendered)} token')
print(f'\n→ 新方案带进 prompt 的原文量是旧方案的 '
      f'{len(rendered) / len(old_rendered):.1f} 倍，且**零额外 LLM 调用**')
