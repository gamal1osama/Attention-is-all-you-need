import torch
import torch.nn as nn
from torch.utils.data import Dataset
from traitlets import Any


class BilingualDataset(Dataset):

    def __init__(self, ds, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.ds = ds
  
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
  
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
  
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tgt_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tgt_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tgt_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: Any) -> Any:
        src_tgt_pair = self.ds[idx]

        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.src_tokenizer.encode(src_text).ids
        dec_input_tokens = self.tgt_tokenizer.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Sequence length {self.seq_len} is too small for the given text: '{src_text}' or '{tgt_text}'")
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        labels = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        assert encoder_input.shape[0] == self.seq_len, f"Encoder input length {encoder_input.shape[0]} does not match expected sequence length {self.seq_len}"
        assert decoder_input.shape[0] == self.seq_len, f"Decoder input length {decoder_input.shape[0]} does not match expected sequence length {self.seq_len}"
        assert labels.shape[0] == self.seq_len, f"Labels length {labels.shape[0]} does not match expected sequence length {self.seq_len}"
        
        return {
            "encoder_input": encoder_input, # shape: (seq_len,)
            "decoder_input": decoder_input, # shape: (seq_len,)
            
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # shape: (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.shape[0]), # shape: (1, seq_len) & (1, 1, seq_len)
        
            "labels": labels, # shape: (seq_len,)

            "src_text": src_text,
            "tgt_text": tgt_text
        }
    

def causal_mask(seq_len):
    mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).type(torch.int64) # shape: (1, seq_len, seq_len)
    return mask==0 # shape: (1, seq_len, seq_len)