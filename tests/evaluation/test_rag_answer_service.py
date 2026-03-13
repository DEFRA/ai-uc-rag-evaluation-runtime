from pytest_mock import MockerFixture

from app.evaluation import rag_answer_service


async def test_answer_with_rag_formats_context_and_returns_output(
    mocker: MockerFixture,
) -> None:
    documents = [{"content": "doc one"}, {"content": "doc two"}]
    mocker.patch(
        "app.evaluation.rag_answer_service.rag_service.query_snapshot",
        new=mocker.AsyncMock(return_value=documents),
    )
    mock_result = mocker.MagicMock()
    mock_result.output = "The answer is 42."
    mock_run = mocker.patch.object(
        rag_answer_service._agent,
        "run",
        new=mocker.AsyncMock(return_value=mock_result),
    )

    result = await rag_answer_service.answer_with_rag("What is the answer?", "group-1")

    assert result == "The answer is 42."
    call_args = mock_run.call_args[0][0]
    assert "doc one" in call_args
    assert "doc two" in call_args
    assert "What is the answer?" in call_args


async def test_answer_with_rag_handles_empty_documents(
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "app.evaluation.rag_answer_service.rag_service.query_snapshot",
        new=mocker.AsyncMock(return_value=[]),
    )
    mock_result = mocker.MagicMock()
    mock_result.output = "I don't have enough context."
    mocker.patch.object(
        rag_answer_service._agent,
        "run",
        new=mocker.AsyncMock(return_value=mock_result),
    )

    result = await rag_answer_service.answer_with_rag("What is the answer?", "group-1")

    assert result == "I don't have enough context."


async def test_answer_with_rag_passes_max_results_to_snapshot(
    mocker: MockerFixture,
) -> None:
    mock_snapshot = mocker.patch(
        "app.evaluation.rag_answer_service.rag_service.query_snapshot",
        new=mocker.AsyncMock(return_value=[]),
    )
    mock_result = mocker.MagicMock()
    mock_result.output = "answer"
    mocker.patch.object(
        rag_answer_service._agent,
        "run",
        new=mocker.AsyncMock(return_value=mock_result),
    )

    await rag_answer_service.answer_with_rag("query", "group-1", max_results=10)

    mock_snapshot.assert_called_once_with("group-1", "query", 10, None)
