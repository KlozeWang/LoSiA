def preprocess_function(example):
    instruction_prefix = example["text"].split("### Response:")[0].strip()
    return {
        "input": f"<s>{instruction_prefix}\n\n### Response:\n",
        "output": f"{example['output'].strip()}</s>"
    }