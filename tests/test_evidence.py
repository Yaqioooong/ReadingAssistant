from reading_assistant.graph.evidence import (
    judge_evidence,
    remove_invalid_citations,
    verify_answer_claims,
)


def test_judge_marks_missing_evidence() -> None:
    result = judge_evidence('张三是谁？', [])
    assert result['status'] == 'insufficient'
    assert result['evidence_count'] == 0


def test_judge_keeps_semantic_hits_as_ready() -> None:
    result = judge_evidence('张三是谁？', [{'text': '原文中写道张三在第一章出场。'}])
    assert result['status'] == 'evidence_ready'
    assert result['evidence_count'] == 1


def test_invalid_citation_is_removed() -> None:
    answer, invalid = remove_invalid_citations(
        '张三在第一章出场。[1] 李四不在这里。[9]',
        [{'index': 1, 'excerpt': '张三在第一章出场。'}],
    )
    assert answer == '张三在第一章出场。[1] 李四不在这里。'
    assert invalid == [9]


def test_claim_verifier_passes_supported_claim() -> None:
    result = verify_answer_claims(
        '张三在第一章出场。[1]',
        [{'index': 1, 'excerpt': '张三在第一章出场。'}],
    )
    assert result['status'] == 'pass'
    assert result['unsupported_claims'] == []


def test_claim_verifier_marks_uncited_claim_for_review() -> None:
    result = verify_answer_claims(
        '张三后来成为将军。',
        [{'index': 1, 'excerpt': '张三在第一章出场。'}],
    )
    assert result['status'] == 'needs_review'
    assert result['unsupported_claims'] == ['张三后来成为将军。']
