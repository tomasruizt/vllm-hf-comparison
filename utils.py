import json


def dump_answer(ans: tuple, file: str) -> None:
    prompt, output, reasoning = ans
    data = {"prompt": prompt, "output": output, "reasoning": reasoning}
    with open(file, "at") as f:
        f.write(json.dumps(data) + "\n")
