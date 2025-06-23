def preprocess_function(example):
    answer_key = f"answer{['A', 'B', 'C'][int(example['label'])-1]}"
    correct_answer = example[answer_key]
    full_question = f"{example['context']} {example['question']}".strip()
    
    return {
        "input": "",
        "output": f"<s>Q: {full_question}\nA: {correct_answer}</s>"
    }