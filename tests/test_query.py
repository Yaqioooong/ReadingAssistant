from reading_assistant.graph.query import analyze_question, split_question


def test_split_numbered_questions() -> None:
    assert split_question('1. 张三是谁？ 2. 李四发生了什么？') == [
        '张三是谁？',
        '李四发生了什么？',
    ]


def test_split_semicolon_questions() -> None:
    assert split_question('张三是谁；李四去了哪里') == ['张三是谁', '李四去了哪里']


def test_single_question_is_not_over_split() -> None:
    assert split_question('请说明张三和李四的关系') == ['请说明张三和李四的关系']


def test_analyze_question_assigns_stable_ids_and_status() -> None:
    is_multi, items = analyze_question('张三是谁？李四是谁？')
    assert is_multi is True
    assert [item['id'] for item in items] == ['q1', 'q2']
    assert [item['status'] for item in items] == ['pending', 'pending']
