from typing import Any

from pytest_mock import MockerFixture

from app.truth import models, repository


def _make_source(**kwargs: object) -> models.TruthDataSource:
    defaults: dict = {
        "id": "source-1",
        "dataset_id": "dataset-1",
        "question_answers": [models.QuestionAnswer(question="q1", answer="a1")],
    }
    return models.TruthDataSource(**{**defaults, **kwargs})


async def _mock_collection(mocker: MockerFixture) -> Any:
    col = mocker.AsyncMock()
    mocker.patch(
        "app.truth.repository._collection",
        new=mocker.AsyncMock(return_value=col),
    )
    return col


async def test_list_sources_returns_summaries(mocker: MockerFixture) -> None:
    col = await _mock_collection(mocker)
    mock_cursor = mocker.MagicMock()
    mock_cursor.to_list = mocker.AsyncMock(
        return_value=[
            {"id": "source-1", "dataset_id": "dataset-1"},
            {"id": "source-2", "dataset_id": "dataset-2"},
        ]
    )
    col.find = mocker.MagicMock(return_value=mock_cursor)

    result = await repository.list_sources()

    assert result == [
        models.TruthDataSourceSummary(id="source-1", dataset_id="dataset-1"),
        models.TruthDataSourceSummary(id="source-2", dataset_id="dataset-2"),
    ]
    col.find.assert_called_once_with({}, {"_id": 0, "id": 1, "dataset_id": 1})


async def test_create_inserts_document(mocker: MockerFixture) -> None:
    col = await _mock_collection(mocker)
    source = _make_source()

    await repository.create(source)

    col.insert_one.assert_awaited_once_with(source.model_dump())


async def test_update_question_answers_returns_true_when_found(
    mocker: MockerFixture,
) -> None:
    col = await _mock_collection(mocker)
    col.update_one.return_value = mocker.MagicMock(matched_count=1)
    qas = [models.QuestionAnswer(question="q2", answer="a2")]

    result = await repository.update_question_answers("source-1", qas)

    assert result is True
    col.update_one.assert_awaited_once_with(
        {"id": "source-1"},
        {"$set": {"question_answers": [qa.model_dump() for qa in qas]}},
    )


async def test_update_question_answers_returns_false_when_not_found(
    mocker: MockerFixture,
) -> None:
    col = await _mock_collection(mocker)
    col.update_one.return_value = mocker.MagicMock(matched_count=0)

    result = await repository.update_question_answers("missing", [])

    assert result is False


async def test_get_returns_validated_source(mocker: MockerFixture) -> None:
    source = _make_source()
    col = await _mock_collection(mocker)
    col.find_one = mocker.AsyncMock(return_value=source.model_dump())

    result = await repository.get("source-1")

    assert result == source
    col.find_one.assert_awaited_once_with({"id": "source-1"}, {"_id": 0})


async def test_get_returns_none_when_not_found(mocker: MockerFixture) -> None:
    col = await _mock_collection(mocker)
    col.find_one = mocker.AsyncMock(return_value=None)

    result = await repository.get("missing")

    assert result is None
