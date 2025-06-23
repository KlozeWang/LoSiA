def preprocess_function(example, **kwargs):
    print(example)
    answer_key = example["answer"]
    print(answer_key)
    correct_option = example[f"option{answer_key}"]
    
    sentence = example["sentence"]
    
    before_blank, after_blank = sentence.split("_", 1)
    
    return {
        "input": f"<s>{before_blank}{correct_option}",
        "output": f"{after_blank}</s>"
    }