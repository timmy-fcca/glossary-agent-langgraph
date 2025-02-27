import argparse
import asyncio
import json
import os
from datetime import datetime

from tqdm import tqdm

from configs import load_config
from data.dataloader import DataEngine
from helpers.workflow_helper import (
    load_workflow,
    parse_responses,
    update_batch_with_external_args,
)


def parse_args():
    parser = argparse.ArgumentParser("Inference code for KV langgraph pipeline")
    parser.add_argument("--cfg", help="path to yaml file for config", required=True)

    args = parser.parse_args()

    if not os.path.isfile(args.cfg):
        raise FileNotFoundError(f"{args.cfg} not found")

    return args


def create_save_dir(name: str) -> str:
    save_dir = f"./responses/{name.upper()}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_responses(response_dict: dict, save_dir: str, file_id: int) -> None:
    with open(
        os.path.join(save_dir, f"response_{str(file_id).zfill(3)}.json"), "w"
    ) as js:
        json.dump(response_dict, js, indent=4)


async def main(engine: DataEngine, config: dict, save_dir: str):
    for file_id, file, generator in tqdm(engine()):
        response_dict = {}
        for batch_id, batch in generator():
            update_batch_with_external_args(batch, config, engine.glossary_path)
            responses = await workflow.abatch(
                batch, config={"max_concurrency": config["max_concurrency"]}
            )
            response_dict.update(parse_responses(responses, batch_id, config))

        save_responses(response_dict, save_dir, file_id)

        if file_id + 1 == config["max_case"]:
            print(f"Reached max # of cases: {file_id + 1}")
            break


if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.cfg)

    save_dir = create_save_dir(config["workflow"])

    workflow = load_workflow(config)

    engine = DataEngine(
        config["data_path"], config["max_concurrency"], config["preprocess_fn"]
    )

    asyncio.run(main(engine, config, save_dir))
