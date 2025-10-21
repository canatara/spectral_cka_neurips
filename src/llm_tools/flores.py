import os
import torch
from . import extract_tools
from . import llm_models

DATA_ROOT = os.environ.get('DATA_ROOT', './')


LANG_DICT = {"English": "sentence_eng_Latn",
             "French": "sentence_fra_Latn",
             "Korean": "sentence_kor_Hang",
             "Turkish": "sentence_tur_Latn", }


def get_flores_200(languages):

    # FLORES-200 is a high-quality evaluation dataset with 200+ languages
    # Meta FLORES-200 - https://github.com/facebookresearch/fairseq/tree/nllb
    from datasets import load_dataset

    flores = load_dataset("facebook/flores", "all")

    corpus = {key: [] for key in languages}
    for lang in languages:
        for sample in flores["dev"]:
            corpus[lang].append(sample[LANG_DICT[lang]])
        print(f"{lang}: {sample[LANG_DICT[lang]]}")

    for lang, val in corpus.items():
        print(f"{lang}: {len(val)} {len(val[0])}")

    return corpus


def process_flores_plus():

    from datasets import load_dataset
    import pandas as pd
    import pycountry

    flores = load_dataset("openlanguagedata/flores_plus")

    df = flores['dev'].to_pandas()

    def get_language(iso_639_3, iso_15924):

        lang = pycountry.languages.get(alpha_3=iso_639_3).name
        lang = lang + f"_{iso_15924}"
        return lang

    df['language'] = df.apply(lambda x: get_language(x['iso_639_3'], x['iso_15924']), axis=1)
    df['topic'] = df.apply(lambda x: x['topic'].lower(), axis=1)

    df.set_index(['id', 'domain', 'topic'], inplace=True)
    df.sort_index(inplace=True)

    # There are 219 languages in the dataset
    assert df.groupby(['id']).size().mean() == 219

    data = []
    for idx, item in df.groupby(['id']):

        id, domain, topic = item.index[0]

        if "business" in topic:
            topic = "business"

        if "world_war_ii" in topic:
            topic = "world_war_ii"

        if "internet" in topic:
            topic = "internet"

        if "natural wonders" in topic:
            topic = "natural wonders"

        if "travel" in topic:
            topic = "travel"

        if "health" in topic:
            topic = "health"

        if "cognitive psycology" in topic:
            topic = "cognitive psycology"

        if "biology" in topic:
            topic = "biology"

        if "science" in topic:
            topic = 'science'

        if "sport" in topic:
            topic = "sport"

        el = {'id': id,
              'domain': domain,
              'topic': topic}

        el |= {key: val for key, val in zip(item['language'], item['text'])}

        data.append(el)

    df = pd.DataFrame(data).set_index(['topic'])

    selected_topics = ['biology', 'business', 'cognitive psycology', 'health',
                       'science', 'sport', 'travel', 'natural wonders', "world_war_ii"]

    from pandas import IndexSlice as idx
    df = df.loc[idx[selected_topics, :]].sort_index()

    # Define languages we want to keep
    languages_to_keep = ['English', 'French', 'Spanish', 'Portuguese', 'German', 'Italian', 'Dutch',
                         'Turkish', 'Kazakh', 'Kyrgyz', 'Greek',
                         'Korean', 'Japanese', 'Thai', 'Vietnamese', 'Mandarin Chinese_Hant',
                         'Polish', 'Romanian', 'Slovak',
                         'Swedish', 'Norwegian', 'Danish', 'Icelandic',
                         'Ukrainian', 'Russian']

    # Filter columns that contain any of the languages in our list
    filtered_columns = [col for col in df.columns if any(lang in col for lang in languages_to_keep)]
    df = df[filtered_columns]

    return df


def compute_instruct_acts(model_name, quant_type, corpus, lang, prompt_fn, suffix="", override=False):

    suffix += f'_{prompt_fn.__name__}'
    filename = f'raw_activations/instruct_reps_{lang}_{model_name}_{quant_type}{suffix}.pt'
    filename = os.path.join(DATA_ROOT, filename)

    if os.path.exists(filename) and not override:
        pass

    else:
        model, tokenizer = llm_models.get_model(model_name, quant_type)
        if 'instruct' not in model_name:
            _, instruct_tokenizer = llm_models.get_model(f'{model_name}-instruct', quant_type)
        else:
            instruct_tokenizer = tokenizer

        messages = prompt_fn(corpus[lang][:])
        input_text = instruct_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        input_tokens = tokenizer(
            input_text,  # This adds <|begin_of_text|> at the beginning of the input
            padding=True,  # Adds padding tokens to make all sequences in a batch the same length
            padding_side='left',
            max_length=512,  # Limits input sequence length - 512, 1024, 2048, or 4096 by default
            truncation=True,  # Truncates sequences longer than max_length instead of raising an error
            return_tensors="pt",
        )

        _, hidden_states = llm_models.generate_responses(model, tokenizer,
                                                         input_text,
                                                         batch_size=250,
                                                         temperature=0.2,
                                                         max_new_tokens=1,
                                                         debug=False)

        # num_layers = len(model.model.layers)  # Get number of layers
        # layer_names = ["embedding"] + [f"layer_{i}" for i in range(1, num_layers+1)]

        # Hidden states have shape (batch_size, num_layers, num_input_tokens, hidden_dim)
        # input_tokens have shape (batch_size, num_input_tokens)
        input_ids = input_tokens.input_ids
        attention_mask = input_tokens.attention_mask

        data = [hidden_states, input_ids, attention_mask]

        print(input_tokens.input_ids.shape, input_tokens.attention_mask.shape, hidden_states.shape)
        torch.save(data, filename)

    return filename


def get_reduced_layer_acts(model_name, prompt_fn, reduce_mode, languages):

    print(model_name, prompt_fn.__name__, reduce_mode)

    all_activations = {}
    for lang in languages:
        suffix = ""
        quant_type = "4bit"
        corpus = None
        filename = compute_instruct_acts(model_name, quant_type, corpus, lang, prompt_fn=prompt_fn, suffix=suffix)

        reduced_suffix = f'{prompt_fn.__name__}_{reduce_mode}' + suffix
        reduced_filename = (
            f'reduced_activations/{str(reduce_mode)}/'
            f'instruct_reps_{model_name}_{lang}_{quant_type}_{reduced_suffix}.pt'
        )
        reduced_filename = os.path.join(DATA_ROOT, reduced_filename)

        if os.path.exists(reduced_filename):
            # print(f"Loading {os.path.basename(reduced_filename)}")
            reduced_act = torch.load(reduced_filename)
        else:
            os.makedirs(os.path.dirname(reduced_filename), exist_ok=True)
            print(f"Computing {os.path.basename(reduced_filename)}")

            data = torch.load(filename, mmap=True, map_location='cpu')
            hidden_states, input_ids, attention_mask = data
            reduced_act = extract_tools.reduce_layer_acts(hidden_states, input_ids, attention_mask, reduce_mode)
            torch.save(reduced_act, reduced_filename)

        all_activations[lang] = reduced_act.to(torch.float64)

    num_layers = reduced_act.shape[1]
    layer_names = ["embedding"] + [f"layer_{i}" for i in range(1, num_layers)]
    for lang in languages:
        all_activations[lang] = {layer: all_activations[lang][:, i] for i, layer in enumerate(layer_names)}

    return all_activations
