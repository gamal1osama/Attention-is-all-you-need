import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from src.config import get_weights_file_path, get_config
from src.dataset import BilingualDataset, causal_mask
from src.model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm
import warnings


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
   
    src_tokenizer = get_or_build_tokenizer(config, ds, config['lang_src'])
    tgt_tokenizer = get_or_build_tokenizer(config, ds, config['lang_tgt'])
    
    train_ds_size = int(len(ds) * 0.9)
    val_ds_size = len(ds) - train_ds_size

    train_ds_row, val_ds_row = random_split(ds, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_row, src_tokenizer, tgt_tokenizer, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_row, src_tokenizer, tgt_tokenizer, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src, max_len_tgt = 0, 0

    for item in ds:
        src_ids = src_tokenizer.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tgt_tokenizer.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def greedy_decode(model, src, src_mask, tgt_tokenizer, max_len, device):
    sos_idx = tgt_tokenizer.token_to_id('[SOS]')
    eos_idx = tgt_tokenizer.token_to_id('[EOS]')

    # Precompute encoder output once: (1, Seq_len, D_model)
    encoder_output = model.encode(src, src_mask)

    # Start with SOS token: (1, 1)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build causal mask for current decoder input: (1, 1, current_len, current_len)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)

        # Decode: (1, current_len, D_model)
        out = model.decode(decoder_input, encoder_output, src_mask, decoder_mask)

        # Project only the last token: (1, 1, Vocab_tgt_len)
        prob = model.project(out[:, -1])

        # Greedy: pick the token with the highest probability
        _, next_word = torch.max(prob, dim=1)  # (1,)

        # Append to decoder input: (1, current_len + 1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(src).fill_(next_word.item()).to(device)],
            dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)  # (generated_len,)


def validate(model, val_dataloader, tgt_tokenizer, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in val_dataloader:
            count += 1

            enc_input = batch['encoder_input'].to(device)  # (1, Seq_len)
            enc_mask = batch['encoder_mask'].to(device)    # (1, 1, 1, Seq_len)

            assert enc_input.size(0) == 1, "Validation batch size must be 1"

            model_out = greedy_decode(model, enc_input, enc_mask, tgt_tokenizer, max_len, device)  # (generated_len,)

            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]
            model_out_text = tgt_tokenizer.decode(model_out.detach().cpu().numpy())

            source_texts.append(src_text)
            expected.append(tgt_text)
            predicted.append(model_out_text)

            print_msg('-' * console_width)
            print_msg(f'{"SOURCE: ":>12s}{src_text}')
            print_msg(f'{"TARGET: ":>12s}{tgt_text}')
            print_msg(f'{"PREDICTED: ":>12s}{model_out_text}')

            if count == num_examples:
                break

    if writer:
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_ds(config)
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')

        state = torch.load(model_filename)
        
        initial_epoch = state['epoch'] + 1

        optimizer.load_state_dict(state['optimizer_state_dict'])

        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing Epoch {epoch:02d}')

        for batch in batch_iterator:
            enc_input = batch['encoder_input'].to(device)  # (B, Seq_len)
            dec_input = batch['decoder_input'].to(device)  # (B, Seq_len)
            enc_mask = batch['encoder_mask'].to(device)    # (B, 1, 1, Seq_len)
            dec_mask = batch['decoder_mask'].to(device)    # (B, 1, Seq_len, Seq_len)
            labels = batch['labels'].to(device)            # (B, Seq_len)

            enc_output = model.encode(enc_input, enc_mask)         # (B, Seq_len, D_model)
            dec_output = model.decode(dec_input, enc_output, enc_mask, dec_mask)  # (B, Seq_len, D_model)
            projected_output = model.project(dec_output)           # (B, Seq_len, Vocab_tgt_len)

            # (B, Seq_len, Vocab_tgt_len) -> (B * Seq_len, Vocab_tgt_len)
            loss = loss_fn(projected_output.view(-1, tgt_tokenizer.get_vocab_size()), labels.view(-1))

            batch_iterator.set_postfix({f'loss': f'{loss.item(): 6.3f}'})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Run validation at the end of each epoch
        validate(model, val_dataloader, tgt_tokenizer, config['seq_len'], device,
                 lambda msg: batch_iterator.write(msg), global_step, writer)

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
