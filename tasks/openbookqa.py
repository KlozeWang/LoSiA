def preprocess_function(example, **kwargs):
    answer_index = ord(example["answerKey"]) - ord("A")
    correct_answer = example["choices"]["text"][answer_index]
    
    input_text = f"<s>Given that: {example['fact1']}\nQuestion: {example['question_stem']}?\nAnswer: "
    
    output_text = f"{correct_answer}</s>"
    
    return {
        "input": input_text,
        "output": output_text
    }