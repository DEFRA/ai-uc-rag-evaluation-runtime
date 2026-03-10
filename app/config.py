from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlmAsAJudgeConfig(BaseSettings):
    model_config = SettingsConfigDict()
    model_id: str = Field(..., alias="LLM_AS_A_JUDGE_MODEL_ID")
    temperature: float = Field(0, alias="LLM_AS_A_JUDGE_TEMPERATURE")
    threshold: float = Field(0.5, alias="LLM_AS_A_JUDGE_THRESHOLD")
    n_prompts: int = Field(10, alias="LLM_AS_A_JUDGE_N_PROMPTS")


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict()
    python_env: str | None = None
    host: str = "127.0.0.1"
    port: int = 8086
    log_config: str | None = None
    mongo_uri: str | None = None
    mongo_database: str = "ai-uc-rag-evaluation-runtime"
    mongo_truststore: str = "TRUSTSTORE_CDP_ROOT_CA"
    aws_endpoint_url: str | None = None
    http_proxy: HttpUrl | None = None
    enable_metrics: bool = False
    tracing_header: str = "x-cdp-request-id"
    evaluation_data_service_url: HttpUrl | None = None
    llm_as_a_judge_config: LlmAsAJudgeConfig = Field(
        default_factory=lambda: LlmAsAJudgeConfig()
    )


config = AppConfig()
