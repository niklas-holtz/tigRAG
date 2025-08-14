from tigrag import TemporalInfluenceGraph, TigParam
from tigrag.dataset_provider.ultradomain_dataset_provider import UltraDomainDatasetProvider
import pandas as pd
import logging

# Logging-Grundkonfiguration einstellen
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] --> %(message)s",
    datefmt="%H:%M:%S"
)

if __name__ == '__main__':
    local_params = TigParam(
        embedding_model_name="local",
        llm_name="llama",
        working_dir="./data"
    )

    aws_params = TigParam(
        embedding_model_name="local",
        llm_name="claude3_7_bedrock",
        working_dir="./working_dir/reduced_datasets/tig/cooking_dataset",
        llm_worker_nodes=10
    )

    params = aws_params
    tig = TemporalInfluenceGraph(
        query_param=params
    )

    # load dataset
    dataset_dir = params.working_dir
    dataset_dir = "./working_dir/reduced_datasets/data"
    provider = UltraDomainDatasetProvider(save_dir=dataset_dir)
    provider.load(3)
    cooking_text = ' '.join(provider.get("cooking", column="context"))
    text = cooking_text

    # insert into tig
    #tig.insert(text)

    # retrieve relevant entries from tig
    answer = tig.retrieve('How does the technical vocabulary differ between professional culinary training materials and consumer-oriented cookbooks?')
    #print('#'*20 )
    #print(answer)