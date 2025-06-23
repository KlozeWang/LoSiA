def preprocess_function(example):
    answer_text = "yes" if example["answer"] else "no"
    
    output_text = (
        f"<s>{example['passage'].strip()}\n"
        f"Question: {example['question'].strip()}?\n"
        f"Answer:\n"
        f"{answer_text}\n"
        f"</s>\n"
    )
    
    return {
        "input": "",
        "output": output_text
    }