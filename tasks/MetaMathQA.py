def preprocess_function(example):
    return {
        "input": f"<s>[INST][/INST] Question: {example['query'].strip()}\n",
        "output": f"Answer: {example['response'].strip()}\n</s>\n"
    }