"""
Inspired by: https://huggingface.co/Qwen/Qwen3-14B
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm

from utils import dump_answer


MAX_NEW_TOKENS = 5000
TEMPERATURE = 0.7


def answer(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion

    generated_ids = model.generate(
        **model_inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return prompt, content, thinking_content


def loop_over_dataset(model, tokenizer, n_examples: int):
    dataset = pd.read_json("CSC_train.json", orient="index")
    dataset = dataset.head(n_examples)
    for text in tqdm(dataset["text"]):
        prompt = "Rate the sarcasm level in the response from 1 to 6: " + str(text)
        ans = answer(model, tokenizer, prompt)
        dump_answer(ans, file="answers/hf.jsonl")


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-14B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    loop_over_dataset(model, tokenizer, n_examples=10)

    # prompt = "What is the capital of France?"
    # ans = answer(model, tokenizer, prompt)
    # dump_answer(ans, file="hf_answers.jsonl")
