import json
import pandas as pd
from tqdm import tqdm
import requests
import multiprocessing as mp

PORT = 8001
MODEL = "Qwen/Qwen3-0.6B"


def loop_over_dataset():
    dataset = pd.read_json("CSC_train.json", orient="index").head(100)
    for text in tqdm(dataset["text"]):
        print("calling vllm")
        prompt = "Rate the sarcasm level in the response from 1 to 6: " + str(text)
        ans = answer(prompt)
        dump_answer(ans)


def answer(question: str):
    url = f"http://localhost:{PORT}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 5000,
        "temperature": 0.7,
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    return question, message["content"], message["reasoning_content"]


def dump_answer(ans: tuple) -> None:
    question, output, reasoning = ans
    data = {"question": question, "output": output, "reasoning": reasoning}
    with open("answers.jsonl", "at") as f:
        f.write(json.dumps(data) + "\n")


def loop_over_dataset_parallel():
    dataset = pd.read_json("CSC_train.json", orient="index").head(1000)
    prompts = [
        "Rate the sarcasm level in the response from 1 to 6: " + str(text)
        for text in dataset["text"]
    ]
    with mp.Pool(mp.cpu_count()) as pool:
        for ans in pool.imap_unordered(answer, tqdm(prompts)):
            dump_answer(ans)


if __name__ == "__main__":
    loop_over_dataset()
    # loop_over_dataset_parallel()
