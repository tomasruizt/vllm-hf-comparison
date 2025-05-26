import pandas as pd
from tqdm import tqdm
import requests
import concurrent.futures

from utils import dump_answer

PORT = 8001
MODEL = "Qwen/Qwen3-14B"
MAX_NEW_TOKENS = 5000
TEMPERATURE = 0.7
N_THREADS = 1000


def loop_over_dataset(n_examples: int):
    dataset = pd.read_json("CSC_train.json", orient="index")
    dataset = dataset.head(n_examples)
    for text in tqdm(dataset["text"]):
        prompt = "Rate the sarcasm level in the response from 1 to 6: " + str(text)
        ans = answer(prompt)
        dump_answer(ans, file="answers/vllm.jsonl")


def answer(prompt: str):
    url = f"http://localhost:{PORT}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    return prompt, message["content"], message["reasoning_content"]


def loop_over_dataset_parallel(n_examples: int):
    dataset = pd.read_json("CSC_train.json", orient="index")
    dataset = dataset.head(n_examples)
    prompts = [
        "Rate the sarcasm level in the response from 1 to 6: " + str(text)
        for text in dataset["text"]
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = [executor.submit(answer, prompt) for prompt in prompts]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(prompts)
        ):
            ans = future.result()
            dump_answer(ans, file="answers/vllm-parallel.jsonl")


if __name__ == "__main__":
    loop_over_dataset(n_examples=10)
    # loop_over_dataset_parallel(n_examples=1000)
