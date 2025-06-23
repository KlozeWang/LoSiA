def preprocess_function(example):
    for ith, item in enumerate(example["choices"]["label"]):
        if item == example["answerKey"]:
            answer_index = ith
            break
    correct_answer = example["choices"]["text"][answer_index]
    output_text = f"<s>Question: {example['question'].strip()}\nAnswer: {correct_answer.strip()}</s>\n"
    
    return {
        "input": "",
        "output": output_text
    }