def test_zero_shot_inference(model, tokenizer, dataset, device):
    """
    Tests the base model's ability to summarize a dialogue without any examples.
    """
    print("\n🧪 Running Zero-Shot Inference Test...")
    
    # Select a random index (like in the lab)
    index = 200
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    # Create the prompt mimicking our future review summarization
    prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
    """

    # 1. Tokenize the input and move to the correct hardware device
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    # 2. Generate the output using the model
    output_tokens = model.generate(
        inputs["input_ids"], 
        max_new_tokens=200
    )[0]
    
    # 3. Decode the output back to human-readable text
    output = tokenizer.decode(output_tokens, skip_special_tokens=True)

    # Print the results beautifully
    dash_line = '-' * 80
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}')
    print(dash_line)