from tigrag import TemporalInfluenceGraph, TigParam
from tigrag.dataset_provider.ultradomain_dataset_provider import UltraDomainDatasetProvider
import logging
from pathlib import Path
from typing import List
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def process_queries(queries_file: str, output_json_path: str, tig):
    qfile = Path(queries_file)
    queries = [line.strip() for line in qfile.read_text(encoding="utf-8").splitlines() if line.strip()]

    outp = Path(output_json_path)
    if outp.is_dir() or outp.suffix.lower() != ".json":
        outp = outp / "answers.json"
    outp.parent.mkdir(parents=True, exist_ok=True)

    results = []
    if outp.exists():
        try:
            results = json.loads(outp.read_text(encoding="utf-8"))
        except Exception:
            pass

    answered = {r["query"]: r["answer"] for r in results if "query" in r and "answer" in r}

    for q in tqdm(queries, desc="Processing queries", unit="query"):
        if q in answered:
            continue
        answer = tig.retrieve(q)
        logging.info(f"Query: {q}")
        logging.info(f"Answer: {answer}")
        results.append({"query": q, "answer": answer})
        tmp_path = outp.with_suffix(outp.suffix + ".tmp")
        tmp_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(outp)
        logging.info(f"Saved progress to: {outp}")

    logging.info(f"Done. Results saved to: {outp}")

    print(f"Done. Results saved to: {outp}")


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
        llm_worker_nodes=10,
        keyword_extraction_method='none'
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


    # insert into tig once
    #tig.insert(text)

    process_queries(
        "./working_dir/reduced_datasets/queries/cooking_queries.txt",
        "./working_dir/reduced_datasets/answers/tigrag_cooking_answers.json",
        tig
    )

# run with  python3 -m tigrag.evaluation.tigrag_answer_generator