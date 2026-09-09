import torch

def generate_response(
    model,
    tokenizer,
    messages,
    do_sample=False
):


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }


    with torch.no_grad():

        generation_kwargs = {
            "max_new_tokens":256,
            "do_sample":do_sample,
            "repetition_penalty":1.2,
            "eos_token_id":tokenizer.eos_token_id,
            "pad_token_id":tokenizer.pad_token_id,
        }

        if do_sample:
            generation_kwargs.update(
            {
                "temperature":0.7,
                "top_p":0.9,
            }
        )

        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )


    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return response

