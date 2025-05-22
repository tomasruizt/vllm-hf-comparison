from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import time


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
    generated_ids = model.generate(**model_inputs, max_new_tokens=5000)
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
    return thinking_content, content


def loop_over_dataset(model, tokenizer):
    dataset = pd.read_json("CSC_train.json", orient="index").head(100)
    start = time.time()
    for text in tqdm(dataset["text"]):
        prompt = "Rate the sarcasm level in the response from 1 to 6: " + str(text)
        _ = answer(model, tokenizer, prompt)
    end = time.time()
    print(f"Time taken: {end - start} seconds")


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    loop_over_dataset(model, tokenizer)

    # # prepare the model input
    # prompt = "What is the capital of France?"
    # thinking_content, content = answer(model, tokenizer, prompt)

    # print("thinking content:", thinking_content)
    # print("content:", content)
