from pytest_mock import MockerFixture

from app.config import LlmAsAJudgeConfig


def test_build_models_populates_dict(mocker: MockerFixture) -> None:
    mocker.patch.dict(
        "os.environ",
        {
            "LLM_AS_A_JUDGE_MODEL_ID_SONNET": "anthropic.claude-3-sonnet-20240229-v1:0",
            "LLM_AS_A_JUDGE_INFERENCE_PROFILE_ARN_SONNET": "arn:aws:bedrock:eu-west-2:111:application-inference-profile/sonnet",
            "LLM_AS_A_JUDGE_MODEL_ID_HAIKU": "anthropic.claude-3-haiku-20240307-v1:0",
            "LLM_AS_A_JUDGE_INFERENCE_PROFILE_ARN_HAIKU": "arn:aws:bedrock:eu-west-2:222:application-inference-profile/haiku",
        },
    )

    cfg = LlmAsAJudgeConfig()

    assert cfg.models["sonnet"] == {
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "arn": "arn:aws:bedrock:eu-west-2:111:application-inference-profile/sonnet",
    }
    assert cfg.models["haiku"] == {
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "arn": "arn:aws:bedrock:eu-west-2:222:application-inference-profile/haiku",
    }


def test_build_models_empty_when_no_env_vars_set(mocker: MockerFixture) -> None:
    mocker.patch(
        "os.environ",
        {
            k: v
            for k, v in __import__("os").environ.items()
            if not k.startswith("LLM_AS_A_JUDGE_MODEL_ID_")
        },
    )

    cfg = LlmAsAJudgeConfig()

    assert cfg.models == {}


def test_build_models_arn_defaults_to_empty_string_when_missing(
    mocker: MockerFixture,
) -> None:
    mocker.patch.dict(
        "os.environ",
        {"LLM_AS_A_JUDGE_MODEL_ID_SONNET": "anthropic.claude-3-sonnet-20240229-v1:0"},
    )

    cfg = LlmAsAJudgeConfig()

    assert cfg.models["sonnet"] == {
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "arn": "",
    }
