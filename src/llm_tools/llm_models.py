from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

# Dictionary of available models
MODEL_DICT = {
    "llama3-8b": "meta-llama/Meta-Llama-3.1-8B",  # Base model
    "llama3-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # Instruction-tuned

    "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",

    "gemma-2b": "google/gemma-2b",  # Small but capable
    "gemma-2b-instruct": "google/gemma-2b-it",  # Instruction-tuned variant

    "gemma-7b": "google/gemma-7b",  # Larger base model
    "gemma-7b-instruct": "google/gemma-7b-it",  # Instruction-tuned variant

    "phi-4": "microsoft/phi-4",  # Strong reasoning capabilities
    "phi-4-coder": "microsoft/phi-4-coder",  # Coder variant,

    # Smaller models
    "phi-1": "microsoft/phi-1",
    "minilm": "microsoft/MiniLM-L12-H384-uncased",
    "tiny-llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "bloomz-1b1": "bigscience/bloomz-1b1",

    # Multilingual models
    "nllb-600M": "facebook/nllb-200-distilled-600M",
    "Qwen-72B-AWQ": "Qwen/Qwen2.5-72B-Instruct-AWQ",

    "TinyLLama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


# Custom stopping criteria to ensure text ends with a period
class EndWithPeriodCriteria(StoppingCriteria):
    def __init__(self, tokenizer, max_new_tokens):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the sequence ends with a period
        last_token = input_ids[0][-1]
        period_tokens = [self.tokenizer.encode('.', add_special_tokens=False)[-1]]

        # Can add more period-like tokens if needed (!, ?, etc.)
        # period_tokens.extend([
        #     self.tokenizer.encode('!', add_special_tokens=False)[-1],
        #     self.tokenizer.encode('?', add_special_tokens=False)[-1]
        # ])

        if last_token in period_tokens:
            return True

        # If we're at max tokens but don't have a period, we'll handle this
        # in post-processing
        return False


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[model_name])
    return tokenizer


def get_model(model_name, quant_type="4bit"):
    # Set up quantization configuration for memory efficiency
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }

    if model_name != "Qwen-72B-AWQ" and model_name != "TinyLLama-1.1B" and torch.cuda.is_available():
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(quant_type == "4bit"),
            load_in_8bit=(quant_type == "8bit"),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"  # if quant_type == "4bit" else None  # (nf4 or fp4)
        )

    # Set the model to use
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[model_name])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DICT[model_name],
        **model_kwargs
    )

    # Compile the model for faster inference
    # model = torch.compile(model, backend="inductor")

    if tokenizer.pad_token is None:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_instruct_responses(model, tokenizer,
                           system_prompt, user_prompt,
                           temperature=0.2,
                           max_new_tokens=1,
                           return_raw_response=False,
                           debug=False,):

    assert isinstance(system_prompt, str), "system_prompt must be a string (batch size = 1)"
    assert isinstance(user_prompt, str), "user_prompt must be a string (batch size = 1)"
    assert tokenizer.chat_template is not None

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
                ]

    # Apply template (also adds <|begin_of_text|> to input)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp_tokens = tokenizer.apply_chat_template(messages, tokenize=True)

    system_tokens = tokenizer.apply_chat_template([messages[0]], tokenize=True)
    user_begin_idx = len(system_tokens) + 4  # the +4 is for <|eot_id|><|start_header_id|>user<|end_header_id|>

    sequences, hidden_states = generate_responses(model, tokenizer, input_text,
                                                  temperature=temperature,
                                                  max_new_tokens=max_new_tokens,
                                                  return_cpu=True,
                                                  debug=debug)

    sequences = sequences[0]
    hidden_states = hidden_states[0]  # hidden_states.shape = (batch_size, num_layers, inp_seq_len, hidden_dim)
    layer_reps = hidden_states[:, user_begin_idx:]  # (num_layers, user_seq_length, hidden_dim)

    # num_layers = len(layer_reps)  # Number of layers in the model
    # inp_length = len(inp_tokens)  # Number of input tokens
    # num_tokens = hidden_states.shape[1]  # Number of token representations
    # assert inp_length + 1 == num_tokens, f"num_inputs ({inp_length}+1) must match seq_len {num_tokens}"

    raw_response = tokenizer.decode(sequences, skip_special_tokens=False)
    response = tokenizer.decode(sequences[user_begin_idx:], skip_special_tokens=False).strip('\n')

    try:
        input, generated = response.split('<|eot_id|>')[:2]
        generated = ''.join(generated.split('\n'))
    except Exception as e:
        input, generated = None, None
        print('here')
        print(e)
        print('here')
        print(response)
        print('here')
        print(response.split('<|eot_id|>'))
        print('here')
        print(user_prompt)
        print('here')
        print(layer_reps.shape)

    if debug:
        print(input)
        print(tokenizer.decode(inp_tokens[user_begin_idx:], skip_special_tokens=False))
        print(tokenizer.decode(inp_tokens, skip_special_tokens=False))
        print(generated)
        print(layer_reps.shape)

    if return_raw_response:
        return raw_response, layer_reps
    else:
        return generated, layer_reps


def get_base_responses(model, tokenizer, input_text,
                       temperature=0.2,
                       max_new_tokens=1,
                       return_raw_response=False,
                       debug=False):

    assert isinstance(input_text, str), "input_text must be a string (batch size = 1)"

    inp_tokens = tokenizer.encode(input_text, add_special_tokens=False)
    user_begin_idx = 0

    sequences, hidden_states = generate_responses(model, tokenizer, input_text,
                                                  temperature=temperature,
                                                  max_new_tokens=max_new_tokens,
                                                  return_cpu=True,
                                                  debug=debug)

    sequences = sequences[0]
    hidden_states = hidden_states[0]  # List of layer representations of the input tokens.
    layer_reps = hidden_states[:, user_begin_idx:]  # (num_layers, user_seq_length, hidden_dim)

    # num_layers = len(layer_reps)  # Number of layers in the model
    inp_length = len(inp_tokens)  # Number of input tokens
    num_tokens = hidden_states.shape[1]  # Number of token representations
    assert inp_length + 1 == num_tokens, f"num_inputs ({inp_length}+1) must match seq_len {num_tokens}"
    raw_response = tokenizer.decode(sequences, skip_special_tokens=False)

    generated = tokenizer.decode(sequences[len(inp_tokens)+1:], skip_special_tokens=False)

    if debug:
        print(input_text)
        print(tokenizer.decode(inp_tokens[:], skip_special_tokens=False))
        print(generated)
        print(layer_reps.shape)

    if return_raw_response:
        return raw_response, layer_reps
    else:
        return generated, layer_reps


def generate_responses(model, tokenizer, input_text,
                       batch_size=300,
                       temperature=0.2, max_new_tokens=1,
                       return_cpu=True,
                       debug=False):

    # Create stopping criteria instance
    stopping_criteria = StoppingCriteriaList(
        [EndWithPeriodCriteria(tokenizer, max_new_tokens=max_new_tokens)]
    )

    if isinstance(input_text, str):
        input_text = [input_text]

    inputs = tokenizer(input_text,  # This adds <|begin_of_text|> at the beginning of the input
                       padding=True,  # Adds padding tokens to make all sequences in a batch the same length
                       padding_side='left',
                       max_length=512,  # Limits input sequence length - 512, 1024, 2048, or 4096 by default
                       truncation=True,  # Truncates sequences longer than max_length instead of raising an error
                       return_tensors="pt",
                       ).to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    num_sentences = len(input_ids)
    num_input_tokens = len(input_ids[0])

    assert len(input_ids) == len(attention_mask)

    input_ids_batched = []
    attention_mask_batched = []
    for i in range(len(input_ids)//batch_size+1):
        input_ids_batched.append(input_ids[i*batch_size:(i+1)*batch_size])
        attention_mask_batched.append(attention_mask[i*batch_size:(i+1)*batch_size])

    outputs = []
    for input_ids, attention_mask in zip(input_ids_batched, attention_mask_batched):
        print("batches:", input_ids.shape, attention_mask.shape)
        outputs.append(
            generate_responses_batched(model, input_ids, attention_mask,
                                       stopping_criteria=stopping_criteria,
                                       pad_token_id=tokenizer.pad_token_id,
                                       temperature=temperature,
                                       max_new_tokens=max_new_tokens,
                                       return_cpu=return_cpu)
        )

    sequences = []
    hidden_states = []
    for output in outputs:
        sequences += list(output.sequences)
        hidden_states.append(output.hidden_states[0])  # Take only hidden states for input tokens
    hidden_states = torch.cat(hidden_states, dim=1)  # (num_layers, batch_size, num_input_tokens, hidden_dim)
    hidden_states = hidden_states.swapaxes(0, 1)  # (batch_size, num_layers, num_input_tokens, hidden_dim)

    for output in outputs:
        seq_length = len(output.sequences[0])
        num_new_tokens = len(output.hidden_states)  # Last token is EOS token (128009)
        assert num_input_tokens + num_new_tokens == seq_length, "sequence length do not match with input length"

    if debug:
        num_layers = len(hidden_states[0])  # Embedding layer (first layer) + hidden layers
        # eos_token_id = sequences[0][-1].item()
        print(num_sentences, num_layers, num_input_tokens)

    return sequences, hidden_states


def generate_responses_batched(model, input_ids, attention_mask,
                               stopping_criteria, pad_token_id,
                               temperature=0.2, max_new_tokens=1,
                               return_cpu=True):

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 max_new_tokens=max_new_tokens,
                                 temperature=temperature,
                                 pad_token_id=pad_token_id,  # Add this line
                                 output_hidden_states=True,
                                 return_dict_in_generate=True,
                                 use_cache=True,
                                 stopping_criteria=stopping_criteria,
                                 #  attn_implementation="flash_attention_2",
                                 )

    # outputs.sequences: Tensor of shape (batch_size, num_total_tokens)
    # num_total_tokens = num_initial_tokens + max_new_tokens (includes EOS token)

    # outputs.hidden_states[0] has shape (num_layers, batch_size, num_input_tokens, hidden_dim)
    # outputs.hidden_states[i] has shape (num_layers, batch_size, 1, hidden_dim) for i > 0
    # len(outputs.hidden_states) = 1 + max_new_tokens - 1 (last token is EOS token)

    device = outputs.sequences.device
    if return_cpu:
        device = 'cpu'

    # Move all outputs to CPU
    outputs.sequences = outputs.sequences.to(device)

    # Move hidden states to CPU if they exist
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        if isinstance(outputs.hidden_states, tuple):
            outputs.hidden_states = [
                torch.stack([h.to(device) for h in hidden_state], dim=0)  # stack layers
                for hidden_state in outputs.hidden_states
            ]

    if hasattr(outputs, 'scores') and outputs.scores is not None:
        outputs.scores = [score.to(device) for score in outputs.scores]

    # Move hidden states to CPU if they exist
    if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
        if isinstance(outputs.past_key_values, tuple):
            outputs.past_key_values = tuple(
                tuple(h.to(device) for h in hidden_state)
                for hidden_state in outputs.past_key_values)

    # Release CUDA memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return outputs
