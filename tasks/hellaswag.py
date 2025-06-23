def preprocess_function(example):
    try:
        label = int(example["label"])
        correct_answer = example["endings"][label]
        
        return {
            "input": f"<s>{example['ctx'].strip()} ",
            "output": f"{correct_answer.strip()}</s>"
        }
    
    except Exception as e:
        print(f"Error processing example {example.get('ind', '')}: {str(e)}")
        return {
            "input": "<s>",
            "output": "</s>"
        }