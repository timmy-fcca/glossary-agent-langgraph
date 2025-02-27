import argparse
import asyncio
import json
import os
from datetime import datetime

from tqdm import tqdm

from agents import workflow_map
from configs import load_config
from data.dataloader import DataEngine


def parsing(response, del_keys):
    # output cleaning
    for del_k in del_keys:
        response[del_k] = ""  # safety
        del response[del_k]

    return response


def parse_args():
    parser = argparse.ArgumentParser("Inference code for KV langgraph pipeline")
    parser.add_argument("--cfg", help="path to yaml file for config")
    return parser.parse_args()


def create_save_dir(name: str):
    save_dir = f"./responses/{name.upper()}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def load_workflow(config: dict):
    try:
        workflow = workflow_map[config["workflow"]]()
    except TypeError:
        workflow = workflow_map[config["workflow"]](config["model_args"])

    return workflow


def update_batch_with_external_args(batch, config, glossary_path):
    for b in batch:
        b.update(
            {
                "external_args": {
                    "model_args": config["model_args"],
                    "glossary_path": glossary_path,
                },
            }
        )


async def process_batch(workflow, batch, config):
    responses = await workflow.abatch(
        batch, config={"max_concurrency": config["max_concurrency"]}
    )
    return responses


def parse_responses(responses, batch_id, config):
    response_dict = {}
    for idx, response in enumerate(responses):
        response_dict[batch_id * config["max_concurrency"] + idx] = parsing(
            response, config["del_keys"]
        )
    return response_dict


def save_responses(response_dict, save_dir, file_id):
    with open(
        os.path.join(save_dir, f"response_{str(file_id).zfill(3)}.json"), "w"
    ) as js:
        json.dump(response_dict, js, indent=4)


async def main(workflow, engine: DataEngine, config: dict, save_dir: str):
    for file_id, file, generator in tqdm(engine()):
        response_dict = {}
        for batch_id, batch in generator():
            update_batch_with_external_args(batch, config, engine.glossary_path)
            responses = await process_batch(workflow, batch, config)
            response_dict.update(parse_responses(responses, batch_id, config))

        save_responses(response_dict, save_dir, file_id)

        if file_id + 1 == config["max_case"]:
            print(f"Reached max # of cases: {file_id + 1}")
            break


if __name__ == "__main__":
    args = parse_args()
    assert os.path.isfile(args.cfg), "Config file not found"

    config = load_config(args.cfg)

    save_dir = create_save_dir(config["workflow"])

    workflow = load_workflow(config)

    engine = DataEngine(
        config["data_path"], config["max_concurrency"], config["preprocess_fn"]
    )

    asyncio.run(main(workflow, engine, config, save_dir))
