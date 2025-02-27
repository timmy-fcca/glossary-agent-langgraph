from typing import Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    model: str
    api_key: str
    openai_api_base: str
    temperature: Optional[float] = 0


class AppConfig(BaseModel):
    model_args: LLMConfig
    data_path: str
    workflow: str
    max_case: Optional[int] = 1
    max_concurrency: Optional[int] = 10
    preprocess_fn: Optional[str] = "sentence_split"
    del_keys: Optional[list[str]] = Field(default_factory=list)


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    return AppConfig(**config).model_dump()


if __name__ == "__main__":
    config = load_config("./configs/test.yaml")
    print(config)
    print(config["model_args"])
