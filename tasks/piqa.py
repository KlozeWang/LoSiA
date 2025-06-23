def preprocess_function(example, **kwargs):
    correct_solution = example[f"sol{int(example['label'])+1}"]
    output_text = f"<s>Question: {example['goal']}\nAnswer: {correct_solution}</s>"
    
    return {
        "input": "",
        "output": output_text
    }