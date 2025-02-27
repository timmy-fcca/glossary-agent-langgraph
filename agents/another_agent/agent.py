# flake8: noqa:E501
from langgraph.graph import END, START, StateGraph

from agents.another_agent import nodes, states


def build_workflow(config: dict):
    all_nodes = nodes.Nodes(config)
    workflow = StateGraph(
        states.InternalState, input=states.ValidateState, output=states.OutputState
    )
    workflow.add_node("validate_node", all_nodes.validate_node)
    workflow.add_edge(START, "validate_node")
    workflow.add_edge("validate_node", END)

    return workflow.compile()


if __name__ == "__main__":
    import yaml

    test_input = {
        "SRC": "NLP",
        "TGT": "Natural language processing",
        "DESC": "A field of artificial intelligence that focuses on the interaction between computers and humans through natural language, enabling machines to understand, interpret, and generate human language.",
        "CONTEXT": "There is no scientific evidence supporting the claims made by NLP advocates, and it has been called a pseudoscience. Scientific reviews have shown that NLP is based on outdated metaphors of the brain's inner workings that are inconsistent with current neurological theory, and that NLP contains numerous factual errors.Reviews also found that research that favored NLP contained significant methodological flaws, and that there were three times as many studies of a much higher quality that failed to reproduce the claims made by Bandler, Grinder, and other NLP practitioners.",
    }
    config = yaml.safe_load(open("./configs/kv-kv.yaml", "r"))
    graph = build_workflow(config.get("model_args"))
    print(graph.invoke(test_input))
