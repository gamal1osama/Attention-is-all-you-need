import torch
import torch.nn as nn
from torch.utils.data import random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists/9(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds = load_dataset('opus-books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
   
    src_tokenizer = get_or_build_tokenizer(config, ds, config['lang_src'])
    tgt_tokenizer = get_or_build_tokenizer(config, ds, config['lang_tgt'])
    
    train_ds_size = int(len(ds) * 0.9)
    val_ds_size = len(ds) - train_ds_size

    train_ds_row, val_ds_row = random_split(ds, [train_ds_size, val_ds_size])

    




