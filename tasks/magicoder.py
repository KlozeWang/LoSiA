def preprocess_function(example):
    return {
        "input": f"<s>[INST]\nQuestion: {example['instruction'].strip()}\nAnswer: [/INST]",
        "output": f"{example['response'].strip()}</s>\n"
    }