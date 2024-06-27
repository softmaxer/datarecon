import logging
import multiprocessing
import os
import time
from multiprocessing import Manager
from typing import List

import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer, pipeline


logging.basicConfig(
    filename=os.environ.get("LOG_FILE"),
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
    force=True,
)


tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33")
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33",
    tokenizer=tokenizer,
    device=0,
)
target_labels = [
    "Wait times and scheduling",
    "Bedside manner and communication",
    "Facility and staff professionalism",
    "Trust and confidence",
    "Accessibility and follow ups",
    "Effectiveness of treatment",
]


def divide_chunks(l: List, n: int):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def get_chunk_labels(data_chunk):
    results = classifier(data_chunk, target_labels, multi_label=True)
    return results


if __name__ == "__main__":
    load_dotenv()

    multiprocessing.set_start_method("spawn")
    manager = Manager()
    final_labels = manager.dict()
    df = pd.read_parquet(os.environ.get("INPUT_DATA"))
    logging.info("Reading from parquet file")
    sequences = df["reviewDescription"].tolist()
    sequences_chunks = list(divide_chunks(sequences, int(os.environ.get("CHUNK_SIZE"))))
    counter = 0
    num_chunks = len(sequences_chunks)
    logging.info(f"Divided reviews into {num_chunks} chunks")
    total_time = 0
    for idx, chunk in enumerate(sequences_chunks):
        logging.info(f"Processing chunk {idx}")
        start_time = time.time()
        with multiprocessing.Pool(processes=5) as pool:
            for result in pool.map(get_chunk_labels, chunk):
                final_labels[counter] = result
                counter += 1

        end_time = time.time()
        chunk_time = end_time - start_time
        total_time += chunk_time
        logging.info(f"Time taken for chunk {idx}: {chunk_time}")

    logging.info(f"All {num_chunks} finished in {total_time} time.")
    res_list = list(final_labels.values())
    res = pd.DataFrame(res_list)
    res["expertId"] = df["expertId"].tolist()
    logging.info("Saving results to output folder")
    res.to_parquet(os.environ.get("OUTPUT_PATH"))
    logging.info("Done.")
