from agents import workflow_map


def parsing(response, del_keys):
    # output cleaning
    for del_k in del_keys:
        response[del_k] = ""  # safety
        del response[del_k]

    return response


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
