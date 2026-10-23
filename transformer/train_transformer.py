import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sacrebleu.metrics import BLEU, CHRF

@dataclass
class TransformerConfig:
    seed: int = 19
    name: str = "transformer"

    max_train_content_len: int = 128

    d_model: int = 384
    nhead: int = 6
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1536
    dropout: float = 0.1
    max_position: int = 512

    warmup_steps: int = 4000
    noam_factor: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    clip_norm: float = 1.0
    label_smoothing: float = 0.0

    batch_size: int = 160
    dev_batch_size: int = 160
    dev_decode_batch_size: int = 32
    dev_max_new_tokens: int = 256
    
    num_workers: int = 2
    use_amp: bool = True
    max_epochs: int = 20
    early_stopping_patience: int = 5
    interval: int = 500

    activation: str = "relu"
    norm_first: bool = False
    final_norm: bool = False
    tie_target_output: bool = False

    selection_metric: str = "val_loss"
    compute_dev_translation_metrics: bool = False
    save_epoch_checkpoints: bool = False
    zero_output_bias: bool = False

def get_paths(project_root, config):
    project_root = Path(project_root).resolve()

    return {
        "train_en": project_root / "data/clean/train.en",
        "train_hi": project_root / "data/clean/train.hi",
        "dev_en": project_root / "data/clean/dev.en",
        "dev_hi": project_root / "data/clean/dev.hi",
        "tokenizer_en": project_root / "output/models/en_unigram_500000_16000_0.9999.model",
        "tokenizer_hi": project_root / "output/models/hi_unigram_500000_16000_0.9999.model",
        "checkpoint_dir": project_root / f"output/checkpoints/{config.name}_seed19"
    }

class TranslationDataset(Dataset):
    def __init__(self, en_texts, hi_texts, tokenizer_en, tokenizer_hi, max_content_len=None):
        assert len(en_texts) == len(hi_texts), "English Hindi texts line counts differ"

        self.eos_src = tokenizer_en.eos_id()
        self.bos_tgt = tokenizer_hi.bos_id()
        self.eos_tgt = tokenizer_hi.eos_id()

        self.en_texts = []
        self.hi_texts = []
        self.tokenized_en = []
        self.tokenized_hi = []

        total = len(en_texts)

        print(
            f"Tokenizing {total} pairs "
            f"(max_content_len={max_content_len})"
        )

        for en_text, hi_text in zip(en_texts, hi_texts):
            en_ids = tokenizer_en.encode_as_ids(en_text)
            hi_ids = tokenizer_hi.encode_as_ids(hi_text)

            if max_content_len is not None:
                if len(en_ids) > max_content_len or len(hi_ids) > max_content_len:
                    continue

            self.tokenized_en.append(en_ids)
            self.tokenized_hi.append(hi_ids)
            self.en_texts.append(en_text)
            self.hi_texts.append(hi_text)

        print(f"Done. Retained {len(self.tokenized_en)}/{total} pairs.")

    def __len__(self):
        return len(self.tokenized_en)

    def __getitem__(self, index):
        en_ids = self.tokenized_en[index]
        hi_ids = self.tokenized_hi[index]

        src = en_ids + [self.eos_src]
        tgt_in = [self.bos_tgt] + hi_ids
        tgt_out = hi_ids + [self.eos_tgt]

        src = torch.tensor(src, dtype=torch.long)
        tgt_in = torch.tensor(tgt_in, dtype=torch.long)
        tgt_out = torch.tensor(tgt_out, dtype=torch.long)

        assert len(tgt_in) == len(tgt_out)

        return (src, tgt_in, tgt_out, len(src), len(tgt_in), index)

class TranslationCollate:
    def __init__(self, pad_src, pad_tgt):
        self.pad_src = pad_src
        self.pad_tgt = pad_tgt

    def __call__(self, batch):
        src_items = []
        tgt_in_items = []
        tgt_out_items = []
        src_lens = []
        tgt_lens = []
        indices = []

        for (src, tgt_in, tgt_out, src_len, tgt_len, index) in batch:
            src_items.append(src)
            tgt_in_items.append(tgt_in)
            tgt_out_items.append(tgt_out)
            src_lens.append(src_len)
            tgt_lens.append(tgt_len)
            indices.append(index)

        src = pad_sequence(
            src_items,
            batch_first=True,
            padding_value=self.pad_src,
        )

        tgt_in = pad_sequence(
            tgt_in_items,
            batch_first=True,
            padding_value=self.pad_tgt,
        )

        tgt_out = pad_sequence(
            tgt_out_items,
            batch_first=True,
            padding_value=self.pad_tgt,
        )

        src_lens = torch.tensor(src_lens, dtype=torch.long)
        tgt_lens = torch.tensor(tgt_lens, dtype=torch.long)
        indices = torch.tensor(indices, dtype=torch.long)

        return (src, tgt_in, tgt_out, src_lens, tgt_lens, indices)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=True)

    def forward(self, x):
        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            raise RuntimeError(
                f"Sequence length {seq_len} exceeds positional "
                f"encoding capacity {self.pe.size(1)}."
            )

        positional = self.pe[:, :seq_len].to(dtype=x.dtype)

        return self.dropout(x + positional)

class TransformerNMT(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_tgt, pad_src, pad_tgt, config):
        super().__init__()

        self.d_model = config.d_model
        self.pad_src = pad_src
        self.pad_tgt = pad_tgt
        self.zero_output_bias = config.zero_output_bias

        self.src_embedding = nn.Embedding(
            vocab_size_src,
            config.d_model,
            padding_idx=pad_src,
        )

        self.tgt_embedding = nn.Embedding(
            vocab_size_tgt,
            config.d_model,
            padding_idx=pad_tgt,
        )

        self.src_position = SinusoidalPositionalEncoding(
            config.d_model,
            max_len=config.max_position,
            dropout=config.dropout,
        )

        self.tgt_position = SinusoidalPositionalEncoding(
            config.d_model,
            max_len=config.max_position,
            dropout=config.dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=config.norm_first,
        )

        encoder_norm = nn.LayerNorm(config.d_model) if config.final_norm else None

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=encoder_norm,
            enable_nested_tensor=False,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=config.norm_first,
        )

        decoder_norm = nn.LayerNorm(config.d_model) if config.final_norm else None

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers,
            norm=decoder_norm,
        )

        self.output_projection = nn.Linear(
            config.d_model,
            vocab_size_tgt,
        )

        if config.tie_target_output:
            self.output_projection.weight = self.tgt_embedding.weight

        self.embedding_scale = math.sqrt(
            config.d_model
        )

        self.reset_parameters()

    def reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

        with torch.no_grad():
            self.src_embedding.weight[self.pad_src].zero_()
            self.tgt_embedding.weight[self.pad_tgt].zero_()
            if self.zero_output_bias:
                with torch.no_grad():
                    self.output_projection.bias.zero_()

    @staticmethod
    def causal_mask(length, device):
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def encode(self, src, src_padding_mask):
        src_emb = (self.src_embedding(src) * self.embedding_scale)
        src_emb = self.src_position(src_emb)

        memory = self.encoder(src_emb, src_key_padding_mask=(src_padding_mask))
        return memory

    def decode(self, tgt_in, memory, tgt_padding_mask, src_padding_mask):
        tgt_emb = (self.tgt_embedding(tgt_in) * self.embedding_scale)
        tgt_emb = self.tgt_position(tgt_emb)

        tgt_mask = self.causal_mask(tgt_in.size(1), tgt_in.device)

        decoded = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return decoded

    def forward(self, src, tgt_in, src_padding_mask, tgt_padding_mask):
        memory = self.encode(src, src_padding_mask)
        decoded = self.decode(tgt_in, memory, tgt_padding_mask, src_padding_mask)
        logits = self.output_projection(decoded)
        return logits

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def prepare_data(project_root, config=None):
    if config is None:
        config = TransformerConfig()

    paths = get_paths(project_root, config)

    train_en = read_lines(paths["train_en"])
    train_hi = read_lines(paths["train_hi"])
    dev_en = read_lines(paths["dev_en"])
    dev_hi = read_lines(paths["dev_hi"])

    print(f"Raw train pairs:{len(train_en)}")
    print(f"Raw dev pairs:{len(dev_en)}")

    tokenizer_en = (spm.SentencePieceProcessor(model_file=str(paths["tokenizer_en"])))
    tokenizer_hi = (spm.SentencePieceProcessor(model_file=str(paths["tokenizer_hi"])))

    pad_id_en = tokenizer_en.pad_id()
    pad_id_hi = tokenizer_hi.pad_id()

    vocab_size_en = tokenizer_en.vocab_size()
    vocab_size_hi = tokenizer_hi.vocab_size()

    train_dataset = TranslationDataset(
        train_en,
        train_hi,
        tokenizer_en,
        tokenizer_hi,
        max_content_len=(config.max_train_content_len),
    )

    dev_dataset = TranslationDataset(
        dev_en,
        dev_hi,
        tokenizer_en,
        tokenizer_hi,
        max_content_len=None,
    )

    del train_en, train_hi, dev_en, dev_hi
    gc.collect()

    max_dev_src = max(len(x) + 1 for x in dev_dataset.tokenized_en)
    max_dev_tgt = max(len(x) + 1 for x in dev_dataset.tokenized_hi)

    if max(max_dev_src, max_dev_tgt) > config.max_position:
        raise RuntimeError("max_position is too small for the dev set. ")

    collate_fn = TranslationCollate(pad_src=pad_id_en, pad_tgt=pad_id_hi)

    data = {
        "train_dataset": train_dataset,
        "dev_dataset": dev_dataset,
        "collate_fn": collate_fn,
        "pad_id_en": pad_id_en,
        "pad_id_hi": pad_id_hi,
        "bos_id_hi": tokenizer_hi.bos_id(),
        "eos_id_hi": tokenizer_hi.eos_id(),
        "eos_id_en": tokenizer_en.eos_id(),
        "vocab_size_en": vocab_size_en,
        "vocab_size_hi": vocab_size_hi,
        "tokenizer_en": tokenizer_en,
        "tokenizer_hi": tokenizer_hi,
        "paths": paths,
    }

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Dev dataset length  : {len(dev_dataset):,}")

    return data

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)

def make_loader(dataset, collate_fn, batch_size, num_workers, shuffle, generator=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=generator,
    )

def build_model(data, config, device):
    model = TransformerNMT(
        vocab_size_src=data["vocab_size_en"],
        vocab_size_tgt=data["vocab_size_hi"],
        pad_src=data["pad_id_en"],
        pad_tgt=data["pad_id_hi"],
        config=config,
    )

    return model.to(device)

def build_optimizer_scheduler(model, config):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )

    def noam_lambda(step_index):
        step = step_index + 1

        return (config.noam_factor * (config.d_model ** -0.5) * min(step ** -0.5, step * (config.warmup_steps ** -1.5)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)

    return optimizer, scheduler

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, scaler, device, pad_src, pad_tgt, config, epoch):
    model.train()

    amp_enabled = config.use_amp and device.type == "cuda"

    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.long)
    correct_tokens = torch.zeros((), device=device, dtype=torch.long)

    total_batches = len(loader)
    epoch_start = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        src, tgt_in, tgt_out, _, _, _ = batch

        src = src.to(device, non_blocking=True)
        tgt_in = tgt_in.to(device, non_blocking=True)
        tgt_out = tgt_out.to(device, non_blocking=True)

        src_padding_mask = (src == pad_src)
        tgt_padding_mask = (tgt_in == pad_tgt)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(src, tgt_in, src_padding_mask, tgt_padding_mask)
            flat_logits = logits.reshape(-1, logits.size(-1))

            targets = tgt_out.reshape(-1)
            loss = loss_fn(flat_logits,targets)

        if amp_enabled:
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            clip_grad_norm_(model.parameters(), max_norm=config.clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            clip_grad_norm_(model.parameters(),max_norm=config.clip_norm)

            optimizer.step()

        scheduler.step()

        valid_mask = (targets != pad_tgt)
        num_valid = valid_mask.sum()

        total_tokens += num_valid

        total_loss += loss.detach().double() * num_valid

        predictions = flat_logits.argmax(dim=-1)
        correct_tokens += ((predictions == targets) & valid_mask).sum()

        if (batch_idx % config.interval == 0 or batch_idx == total_batches):
            elapsed = (time.perf_counter() - epoch_start)

            running_loss = (total_loss / total_tokens).item()
            running_acc = (correct_tokens.double() / total_tokens * 100.0).item()

            sec_per_batch = elapsed / batch_idx

            remaining = (total_batches - batch_idx) * sec_per_batch

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"[Epoch {epoch}] "
                f"batch {batch_idx:,}/"
                f"{total_batches:,} | "
                f"loss {running_loss:.4f} | "
                f"token_acc "
                f"{running_acc:.2f}% | "
                f"lr {current_lr:.8f} | "
                f"elapsed "
                f"{elapsed/60:.1f} min | "
                f"ETA "
                f"{remaining/60:.1f} min"
            )

    epoch_loss = (total_loss / total_tokens).item()
    epoch_acc = (correct_tokens.double() / total_tokens* 100.0).item()

    return epoch_loss, epoch_acc

def evaluate(model, loader, loss_fn, device, pad_src, pad_tgt, config):
    model.eval()

    amp_enabled = config.use_amp and device.type == "cuda"

    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.long)
    correct_tokens = torch.zeros((), device=device, dtype=torch.long)

    with torch.inference_mode():
        for batch in loader:
            src, tgt_in, tgt_out, _, _, _ = batch

            src = src.to(device, non_blocking=True)
            tgt_in = tgt_in.to(device, non_blocking=True)
            tgt_out = tgt_out.to(device, non_blocking=True)
            
            src_padding_mask = (src == pad_src)
            tgt_padding_mask = (tgt_in == pad_tgt)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(src, tgt_in, src_padding_mask, tgt_padding_mask)
                flat_logits = (logits.reshape(-1, logits.size(-1)))

                targets = (tgt_out.reshape(-1))

                loss = loss_fn(flat_logits, targets)

            valid_mask = targets != pad_tgt
            valid_tokens = valid_mask.sum()

            total_loss += loss.detach().double() * valid_tokens
            total_tokens += valid_tokens

            predictions = flat_logits.argmax(dim=-1)

            correct_tokens += ((predictions == targets) & valid_mask).sum()

    epoch_loss = (total_loss / total_tokens).item()
    epoch_acc = (correct_tokens.double() / total_tokens * 100.0).item()

    return epoch_loss, epoch_acc

def save_checkpoint_atomic(path, *, epoch, model, optimizer, scheduler, scaler, best_val_loss, history, epochs_without_improvement, train_generator, config):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "history": history,
        "epochs_without_improvement": epochs_without_improvement,
        "config": asdict(config),
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_generator_state": train_generator.get_state() if train_generator is not None else None,
    }

    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)

def load_checkpoint(path, *, model, optimizer, scheduler, scaler, device, train_generator):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if (scaler is not None and checkpoint.get("scaler_state_dict") is not None):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if checkpoint.get("python_random_state") is not None:
        random.setstate(checkpoint["python_random_state"])

    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"])

    if (torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

    if (train_generator is not None and checkpoint.get("train_generator_state") is not None):
        train_generator.set_state(checkpoint["train_generator_state"])

    return {
        "start_epoch": checkpoint["epoch"] + 1,
        "best_val_loss": checkpoint["best_val_loss"],
        "history": checkpoint.get("history", []),
        "epochs_without_improvement": checkpoint.get("epochs_without_improvement", checkpoint.get("epochs_without_chrf_improvement", 0)),
    }

@torch.inference_mode()
def greedy_batch(model, src, pad_src, pad_tgt, bos_tgt, eos_tgt, max_new_tokens, use_amp):
    device = src.device
    amp_enabled = use_amp and device.type == "cuda"
    src_padding_mask = src.eq(pad_src)

    with torch.amp.autocast("cuda", enabled=amp_enabled):
        memory = model.encode(src, src_padding_mask)

    output = torch.full((src.size(0), 1), bos_tgt, dtype=torch.long, device=device)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            decoded = model.decode(output, memory, output.eq(pad_tgt), src_padding_mask)
            next_token = model.output_projection(decoded[:, -1]).argmax(-1)

        next_token = torch.where(done, torch.full_like(next_token, pad_tgt), next_token)
        output = torch.cat([output, next_token[:, None]], dim=1)
        done |= next_token.eq(eos_tgt)

        if bool(done.all()):
            break

    return output

@torch.inference_mode()
def dev_translation_metrics(model, data, config, device):
    dataset = data["dev_dataset"]
    loader = make_loader(dataset, data["collate_fn"], config.dev_decode_batch_size, config.num_workers, shuffle=False)

    hypotheses = [None] * len(dataset)
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for src, _, _, _, _, indices in loader:
        src = src.to(device, non_blocking=True)

        generated = greedy_batch(model, src, data["pad_id_en"], data["pad_id_hi"], data["bos_id_hi"], data["eos_id_hi"], config.dev_max_new_tokens, config.use_amp)

        for row, index in zip(generated.cpu().tolist(), indices.tolist()):
            ids = []
            for token in row:
                token = int(token)
                if token == data["bos_id_hi"]:
                    continue
                if token == data["eos_id_hi"]:
                    break
                if token != data["pad_id_hi"]:
                    ids.append(token)
            hypotheses[index] = data["tokenizer_hi"].decode_ids(ids)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    references = dataset.hi_texts

    return {
        "dev_bleu": BLEU().corpus_score(hypotheses, [references]).score,
        "dev_chrf": CHRF(word_order=2)
        .corpus_score(hypotheses, [references])
        .score,
        "dev_decode_seconds": elapsed,
    }


def fit(*, data, config, device, batch_size, num_workers, num_epochs, resume=False):
    paths = data["paths"]

    checkpoint_dir = (paths["checkpoint_dir"])

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epoch_models_dir = checkpoint_dir / "epoch_models"

    last_path = checkpoint_dir / "last_checkpoint.pt"
    best_path = checkpoint_dir / "best_checkpoint.pt"
    history_path = checkpoint_dir / "history.json"
    config_path = checkpoint_dir / "run_config.json"

    if (not resume and last_path.exists()):
        raise RuntimeError(
            f"Existing checkpoint found: "
            f"{last_path}\n"
            "Use --resume to continue it, "
        )

    run_config = TransformerConfig(**asdict(config))
    run_config.batch_size = batch_size
    run_config.num_workers = num_workers
    run_config.max_epochs = num_epochs
    train_generator = torch.Generator()
    train_generator.manual_seed(run_config.seed)

    if run_config.save_epoch_checkpoints:
        epoch_models_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(
        data["train_dataset"],
        data["collate_fn"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        generator=train_generator,
    )

    dev_loader = make_loader(
        data["dev_dataset"],
        data["collate_fn"],
        batch_size=run_config.dev_batch_size,
        num_workers=num_workers,
        shuffle=False,
        generator=None,
    )

    print(f"Train batches/epoch :{len(train_loader)}")

    print(f"Dev batches : {len(dev_loader)}"
    )

    model = build_model(data, run_config, device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters: {parameter_count}")

    optimizer, scheduler = (build_optimizer_scheduler(model, run_config))

    train_loss_fn = nn.CrossEntropyLoss(
        ignore_index=data["pad_id_hi"],
        label_smoothing=run_config.label_smoothing,
    )

    eval_loss_fn = nn.CrossEntropyLoss(
        ignore_index=data["pad_id_hi"],
    )

    amp_enabled = run_config.use_amp and device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    with open(config_path, "w", encoding="utf-8") as f:
        run_config_dict = (asdict(run_config))

        run_config_dict["trainable_parameters"] = parameter_count

        json.dump(run_config_dict, f, indent=2)

    start_epoch = 1
    best_val_loss = float("inf")
    history = []
    epochs_without_improvement = 0

    if resume:
        print(f"Loading checkpoint: {last_path}")

        state = load_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            train_generator=(train_generator),
        )

        start_epoch = state["start_epoch"]
        best_val_loss = state["best_val_loss"]
        history = state["history"]
        epochs_without_improvement = state["epochs_without_improvement"]
        print(f"Resuming at epoch: {start_epoch}.")

    if run_config.selection_metric == "val_loss":
        best_score = min((record["val_loss"] for record in history), default=float("inf"))

    elif run_config.selection_metric == "dev_chrf":
        best_score = max((record["dev_chrf"] for record in history if record.get("dev_chrf") is not None), default=float("-inf"))

    else:
        raise ValueError(
            f"Unsupported selection metric: "
            f"{run_config.selection_metric}"
        )

    for epoch in range(start_epoch, num_epochs + 1):
        print(
            f"Transformer Epoch "
            f"{epoch}/{num_epochs} started"
        )

        epoch_start = time.perf_counter()

        train_loss, train_acc = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=train_loss_fn,
                scaler=scaler,
                device=device,
                pad_src=data["pad_id_en"],
                pad_tgt=data["pad_id_hi"],
                config=run_config,
                epoch=epoch,
            )
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=dev_loader,
            loss_fn=eval_loss_fn,
            device=device,
            pad_src=data["pad_id_en"],
            pad_tgt=data["pad_id_hi"],
            config=run_config,
        )

        dev_bleu = None
        dev_chrf = None
        dev_decode_seconds = None

        if run_config.compute_dev_translation_metrics:
            sequence_metrics = dev_translation_metrics(
                model,
                data,
                run_config,
                device,
            )

            dev_bleu = sequence_metrics["dev_bleu"]
            dev_chrf = sequence_metrics["dev_chrf"]
            dev_decode_seconds = sequence_metrics[
                "dev_decode_seconds"
            ]

        best_val_loss = min(best_val_loss, val_loss)

        if run_config.selection_metric == "val_loss":
            current_score = val_loss
            improved = current_score < best_score

        elif run_config.selection_metric == "dev_chrf":
            if dev_chrf is None:
                raise RuntimeError(
                    "dev_chrf selection requires "
                    "compute_dev_translation_metrics=true"
                )

            current_score = dev_chrf
            improved = current_score > best_score

        if improved:
            best_score = current_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_seconds = time.perf_counter() - epoch_start

        current_lr = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_cross_entropy": val_loss,
            "val_accuracy": val_acc,
            "dev_bleu": dev_bleu,
            "dev_chrf": dev_chrf,
            "dev_decode_seconds": dev_decode_seconds,
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
        }

        history.append(record)

        if run_config.save_epoch_checkpoints:
            epoch_path = (epoch_models_dir/ f"epoch_{epoch:02d}.pt")

            epoch_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": record,
                "config": asdict(run_config),
            }

            temp_path = epoch_path.with_suffix(epoch_path.suffix + ".tmp")

            torch.save(epoch_payload, temp_path)
            os.replace(temp_path, epoch_path)

        print("")
        print(f"Epoch {epoch}/{num_epochs} complete", end = " ")
        print(f"Train Loss : {train_loss:.4f}", end = " ")
        print(f"Train Acc  : {train_acc:.2f}%", end = " ")
        print(f"Val Loss   : {val_loss:.4f}", end = " ")
        print(f"Val Acc    : {val_acc:.2f}%", end = " ")
        print(f"LR         : {current_lr:.8f}", end = " ")
        print(f"Epoch Time : {epoch_seconds/60:.2f} min ({epoch_seconds/3600:.2f} h)", end = " ")
        print(f"No-improve epochs: {epochs_without_improvement}/{run_config.early_stopping_patience}")

        save_checkpoint_atomic(
            last_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_loss=best_val_loss,
            history=history,
            epochs_without_improvement=epochs_without_improvement,
            train_generator=train_generator,
            config=run_config,
        )

        if improved:
            save_checkpoint_atomic(
                best_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_loss=best_val_loss,
                history=history,
                epochs_without_improvement=epochs_without_improvement,
                train_generator=train_generator,
                config=run_config,
            )

            print(f"Best checkpoint saved: {best_path}")

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if (epochs_without_improvement >= run_config.early_stopping_patience):
            print("Early stopping triggered.")
            break

    return history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        config_values = json.load(config_file)

    config = TransformerConfig(**config_values)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. ")

    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")

    script_dir = (Path(__file__).resolve().parent)
    project_root = (script_dir.parent)

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA device     : {torch.cuda.get_device_name(0)}")
    print(f"Project root    : {project_root}")

    data = prepare_data(project_root, config)

    preview_model = build_model(data, config, device,)

    param_count = sum(p.numel() for p in preview_model.parameters() if p.requires_grad)

    print(f"Transformer trainable parameters: {param_count:,}")

    del preview_model
    gc.collect()
    torch.cuda.empty_cache()
    
    fit(
        data=data,
        config=config,
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        num_epochs=config.max_epochs,
        resume=args.resume,
    )

if __name__ == "__main__":
    main()