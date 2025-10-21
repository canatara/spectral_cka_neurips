import os
import torch

DATA_ROOT = os.environ.get('DATA_ROOT', './')


def which_language_prompt(sentences):

    system_prompt = (
        'You are a helpful assistant that detects the language of the text and '
        'returns only the language name in "English".'
    )

    if isinstance(sentences, str):
        sentences = [sentences]

    messages = []
    for text in sentences:

        user_prompt = f"{text}"
        messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    return messages


def which_topic_prompt(sentences):

    system_prompt = (
        'You are a helpful assistant that detects the topic of the text and '
        'returns only the topic name in "English".'
    )

    if isinstance(sentences, str):
        sentences = [sentences]

    messages = []
    for text in sentences:

        user_prompt = f"{text}"
        messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    return messages


def summary_prompt(sentences):

    system_prompt = "You are a helpful assistant that summarizes text in at most 10 words"

    if isinstance(sentences, str):
        sentences = [sentences]

    messages = []
    for text in sentences:

        user_prompt = f"{text}"
        messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    return messages


def english_summary_prompt(sentences):

    system_prompt = (
        "You are a helpful assistant that summarizes text in at most 10 words "
        "then translates the summary to English."
    )
    system_prompt = (
        "Translate the input text to English, summarize it in at most 10 words "
        "and return only the summary without including the original text."
    )
    # system_prompt = (
    #     "Translate the input text to English, summarize it in at most 10 words and return only the summary."
    # )

    if isinstance(sentences, str):
        sentences = [sentences]

    messages = []
    for text in sentences:

        user_prompt = f"{text}"
        messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    return messages


def empty_system_prompt(sentences):

    system_prompt = ''

    if isinstance(sentences, str):
        sentences = [sentences]

    messages = []
    for text in sentences:

        user_prompt = f"{text}"
        messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    return messages


def reduce_token_reps(hidden_state, attention_mask, input_ids, reduce_mode, dtype=torch.float64, device='cuda'):

    num_samples, num_tokens, num_features = hidden_state.shape

    assert attention_mask.shape == (num_samples, num_tokens)
    assert input_ids.shape == (num_samples, num_tokens)

    if isinstance(reduce_mode, tuple):
        reduce_mode, token_idx = reduce_mode
        assert isinstance(reduce_mode, str) and isinstance(token_idx, int)

    hidden_state = hidden_state.to(dtype).to(device)
    attention_mask = attention_mask.to(dtype).to(device).view(num_samples, num_tokens, 1)

    sentence_lengths = attention_mask.sum(dim=1)
    min_seq_length = sentence_lengths.min()

    if reduce_mode == "mean":
        hidden_state = hidden_state.mean(dim=1)

    elif reduce_mode == "std":
        hidden_state = hidden_state.std(dim=1)

    elif reduce_mode == "norm":
        hidden_state = hidden_state.norm(dim=1)

    elif reduce_mode == "squared_mean":
        hidden_state = (hidden_state**2).mean(dim=1)

    elif reduce_mode == "masked_mean":
        hidden_state = hidden_state * attention_mask
        hidden_state = hidden_state.sum(dim=1) / sentence_lengths

    elif reduce_mode == "single_token":
        hidden_state = hidden_state[:, token_idx, :]

    elif reduce_mode == "flattened":
        hidden_state = hidden_state[:, -token_idx:, :].reshape(num_samples, -1)

    elif reduce_mode == "rand_proj":
        hidden_state = hidden_state * attention_mask
        hidden_state = hidden_state[:, -min_seq_length:, :]
        hidden_state = RandomProjection(hidden_state)

    else:
        raise ValueError(f"Invalid reduce_mode: {reduce_mode}")

    return hidden_state.cpu()


def reduce_layer_acts(hidden_states, input_ids, attention_mask, reduce_mode):

    assert len(hidden_states.shape) == 4

    num_samples, num_layers, num_tokens, num_features = hidden_states.shape

    reduced_hidden_states = []
    for i in range(num_layers):
        reduced_state = reduce_token_reps(hidden_states[:, i, :, :],
                                          attention_mask,
                                          input_ids,
                                          reduce_mode).unsqueeze(1)
        reduced_hidden_states.append(reduced_state)

    # Required for torch.save otherwise it will be a view and not a copy
    reduced_hidden_states = torch.cat(reduced_hidden_states, dim=1).clone()

    return reduced_hidden_states


@torch.jit.script
# @torch.no_grad()
def RandomProjection(out: torch.Tensor) -> torch.Tensor:

    device = 'cuda'
    dtype = torch.float64

    num_samples, num_tokens, feat_dim = out.shape
    output_dim = 4096

    out = out.to(dtype).to(device).reshape(num_samples, num_tokens*feat_dim)
    # out_i = torch.nn.functional.normalize(out_i, dim=1)

    R = torch.randn(size=(out.shape[-1], output_dim),  device=device, dtype=dtype)
    R /= torch.sqrt(output_dim)

    out = out @ R

    # del R, out_i
    # torch.cuda.empty_cache()

    # while is_oom:

    # try:
    #     for i in range(N):
    #         rand_proj = torch.randn(size=(feat_dim, output_dim//N), device=out.device).type(out.dtype)
    #         rand_proj = rand_proj / torch.linalg.norm(rand_proj, axis=1, keepdim=True)

    #         projected_out += [out @ rand_proj]

    #         is_oom = False

    # except torch.cuda.OutOfMemoryError:
    #     N = 2*N
    #     print(f"Cuda OOM - Trying {N} batched random projection")

    return out.cpu()


PROMPT_DICT = {
    "which_language_prompt": which_language_prompt,
    "which_topic_prompt": which_topic_prompt,
    "summary_prompt": summary_prompt,
    "english_summary_prompt": english_summary_prompt,
    "empty_system_prompt": empty_system_prompt,
}
