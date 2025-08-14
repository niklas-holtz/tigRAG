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
        working_dir="./working_dir/tig/cooking_dataset",
        llm_worker_nodes=10
    )

    params = aws_params

    tig = TemporalInfluenceGraph(
        query_param=params
    )

    # load dataset
    provider = UltraDomainDatasetProvider(save_dir=params.working_dir)
    provider.load()
    cooking_text = ' '.join(provider.get("cooking", column="context"))
    print(len(cooking_text))
    #text = cooking_text[:7000000]
    text = cooking_text
    # insert into tig
    tig.insert(text)

    # retrieve relevant entries from tig
    #tig.retrieve('How do traditional cooking methods compare with modern approaches in the various texts?')