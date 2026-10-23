import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader

SEED = 19
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
ENCODER_DIM = 512
DECODER_DIM = 512
ATTENTION_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
MAX_EPOCHS = 20
CLIP_NORM = 1.0
MAX_TRAIN_CONTENT_LEN = 128
TRAIN_BATCH_SIZE = 64
DEV_BATCH_SIZE = 64
NUM_WORKERS = 2
USE_AMP = True
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
MIN_LR = 1e-5
EARLY_STOPPING_PATIENCE = 5
INTERVAL = 500


class DataTransformation(Dataset):
    def __init__(self, en_texts, hi_texts, tokenizer_en, tokenizer_hi, max_len=None):
        assert len(en_texts) == len(hi_texts), "English Hindi texts line counts differ"

        self.eos_src = tokenizer_en.eos_id()
        self.bos_tgt = tokenizer_hi.bos_id()
        self.eos_tgt = tokenizer_hi.eos_id()
        self.tokenized_en = []
        self.tokenized_hi = []
        
        for en_text, hi_text in zip(en_texts, hi_texts):
            en_ids = tokenizer_en.encode_as_ids(en_text)
            hi_ids = tokenizer_hi.encode_as_ids(hi_text)
            if max_len is not None and (len(en_ids) > max_len or len(hi_ids) > max_len):
                continue
            self.tokenized_en.append(en_ids)
            self.tokenized_hi.append(hi_ids)


    def __len__(self):
        return len(self.tokenized_en)

    def __getitem__(self, i):
        en_ids = self.tokenized_en[i]
        hi_ids = self.tokenized_hi[i]
        src = torch.tensor(en_ids + [self.eos_src], dtype=torch.long)
        tgt_in = torch.tensor([self.bos_tgt] + hi_ids, dtype=torch.long)
        tgt_out = torch.tensor(hi_ids + [self.eos_tgt], dtype=torch.long)
        assert len(tgt_in) == len(tgt_out)
        return src, tgt_in, tgt_out, len(src), len(tgt_in)

class CollateFn:
    def __init__(self, pad_src, pad_tgt):
        self.pad_src = pad_src
        self.pad_tgt = pad_tgt

    def __call__(self, data):
        source, tgt_in, tgt_out, src_lens, tgt_lens = [], [], [], [], []
        for src, in_ids, out_ids, src_len, tgt_len in data:
            source.append(src)
            tgt_in.append(in_ids)
            tgt_out.append(out_ids)
            src_lens.append(src_len)
            tgt_lens.append(tgt_len)
        source = pad_sequence(source, batch_first=True, padding_value=self.pad_src)
        tgt_in = pad_sequence(tgt_in, batch_first=True, padding_value=self.pad_tgt)
        tgt_out = pad_sequence(tgt_out, batch_first=True, padding_value=self.pad_tgt)
        src_lens = torch.tensor(src_lens, dtype=torch.long)
        tgt_lens = torch.tensor(tgt_lens, dtype=torch.long)
        return source, tgt_in, tgt_out, src_lens, tgt_lens

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, pad_id):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.bilstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=src.size(1))
        hidden = hidden.reshape(self.num_layers, 2, src.size(0), self.hidden_dim)
        cell = cell.reshape(self.num_layers, 2, src.size(0), self.hidden_dim)
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=-1)
        cell = torch.cat((cell[:, 0], cell[:, 1]), dim=-1)
        return output, hidden, cell

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_dim, attention_dim)
        self.decoder_projection = nn.Linear(decoder_dim, attention_dim)
        self.score_projection = nn.Linear(attention_dim, 1)

    def project_encoder(self, encoder_output):
        return self.encoder_projection(encoder_output)

    def forward(self, encoder_output, projected_encoder, decoder_hidden, src_mask):
        projected_decoder = self.decoder_projection(decoder_hidden).unsqueeze(1)
        energy = torch.tanh(projected_encoder + projected_decoder)
        scores = self.score_projection(energy).squeeze(-1)
        scores = scores.masked_fill(src_mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_output).squeeze(1)
        return context, attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, num_layers, embedding_dim, encoder_dim, decoder_dim, dropout, pad_id, attention):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + encoder_dim,
            hidden_size=decoder_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear_projection = nn.Linear(encoder_dim + decoder_dim, vocab_size)

    def forward(self, inp_t, hidden, cell, outputs_encoder, projected_encoder, src_mask):
        embedded = self.dropout(self.embedding(inp_t)).unsqueeze(1)
        context, attention_weights = self.attention(
            outputs_encoder, projected_encoder, hidden[-1], src_mask
        )
        decoder_lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=-1)
        output, (hidden_next, cell_next) = self.lstm(decoder_lstm_input, (hidden, cell))
        output = output.squeeze(1)
        logits = self.linear_projection(torch.cat((output, context), dim=-1))
        return logits, hidden_next, cell_next, attention_weights

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_tgt, embedding_dim, encoder_dim, decoder_dim, hidden_dim, num_layers, dropout, pad_src, pad_tgt, attention):
        super().__init__()
        assert encoder_dim == 2 * hidden_dim
        assert decoder_dim == 2 * hidden_dim
        self.encoder = Encoder(vocab_size_src, embedding_dim, hidden_dim, num_layers, dropout, pad_src)
        self.decoder = Decoder(vocab_size_tgt, num_layers, embedding_dim, encoder_dim, decoder_dim, dropout, pad_tgt, attention)

    def forward(self, src, src_lens, tgt_in, src_mask):
        output_enc, hidden, cell = self.encoder(src, src_lens)
        projected_encoder = self.decoder.attention.project_encoder(output_enc)
        predictions = []
        for t in range(tgt_in.size(1)):
            logits, hidden, cell, _ = self.decoder(
                tgt_in[:, t], hidden, cell, output_enc, projected_encoder, src_mask
            )
            predictions.append(logits)
        return torch.stack(predictions, dim=1)

def build_model(vocab_size_en, vocab_size_hi, pad_id_en, pad_id_hi, device):
    attention = BahdanauAttention(ENCODER_DIM, DECODER_DIM, ATTENTION_DIM)
    model = Seq2Seq(
        vocab_size_en,
        vocab_size_hi,
        EMBEDDING_DIM,
        ENCODER_DIM,
        DECODER_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT,
        pad_id_en,
        pad_id_hi,
        attention,
    )
    return model.to(device)

def make_loader(dataset, collate_fn, batch_size, num_workers, shuffle, generator=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=generator,
    )

def train_one_epoch(model, loader, optimizer, loss_fn, device, pad_src, pad_tgt, clip_norm, scaler, use_amp, epoch, interval=500):
    model.train()
    amp_enabled = use_amp and device.type == "cuda"
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.long)
    correct_tokens = torch.zeros((), device=device, dtype=torch.long)
    epoch_start = time.perf_counter()
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        src, tgt_in, tgt_out, src_lens, _ = batch
        src = src.to(device, non_blocking=True)
        tgt_in = tgt_in.to(device, non_blocking=True)
        tgt_out = tgt_out.to(device, non_blocking=True)
        src_mask = src == pad_src
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(src, src_lens, tgt_in, src_mask)
            flat_logits = logits.reshape(-1, logits.size(-1))
            targets = tgt_out.reshape(-1)
            loss = loss_fn(flat_logits, targets)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at epoch {epoch}, batch {batch_idx}: "
                f"{loss.detach().float().cpu().item()}"
            )

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

        valid_mask = targets != pad_tgt
        num_valid = valid_mask.sum()
        total_tokens += num_valid
        total_loss += loss.detach().double() * num_valid
        predictions = flat_logits.argmax(dim=-1)
        correct_tokens += ((predictions == targets) & valid_mask).sum()

        if batch_idx % interval == 0 or batch_idx == total_batches:
            elapsed = time.perf_counter() - epoch_start
            running_loss = (total_loss / total_tokens).item()
            running_acc = (correct_tokens.double() / total_tokens * 100.0).item()
            sec_per_batch = elapsed / batch_idx
            remaining = (total_batches - batch_idx) * sec_per_batch
            print(f"[Epoch {epoch}] batch {batch_idx:,}/{total_batches:,} | ", end=" ")
            print(f"loss {running_loss:.4f} | token_acc {running_acc:.2f}% | ", end = " ")
            print(f"elapsed {elapsed/60:.1f} min | ETA {remaining/60:.1f} min")

    return (
        (total_loss / total_tokens).item(),
        (correct_tokens.double() / total_tokens * 100.0).item(),
    )

def evaluate(model, loader, loss_fn, device, pad_src, pad_tgt, use_amp):
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.long)
    correct_tokens = torch.zeros((), device=device, dtype=torch.long)

    with torch.inference_mode():
        for batch in loader:
            src, tgt_in, tgt_out, src_lens, _ = batch
            src = src.to(device, non_blocking=True)
            tgt_in = tgt_in.to(device, non_blocking=True)
            tgt_out = tgt_out.to(device, non_blocking=True)
            src_mask = src == pad_src

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(src, src_lens, tgt_in, src_mask)
                flat_logits = logits.reshape(-1, logits.size(-1))
                targets = tgt_out.reshape(-1)
                loss = loss_fn(flat_logits, targets)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss.")

            valid_mask = targets != pad_tgt
            valid_tokens = valid_mask.sum()
            total_loss += loss.detach().double() * valid_tokens
            total_tokens += valid_tokens
            predictions = flat_logits.argmax(dim=-1)
            correct_tokens += ((predictions == targets) & valid_mask).sum()

    return (
        (total_loss / total_tokens).item(),
        (correct_tokens.double() / total_tokens * 100.0).item(),
    )

def fit(*, model, train_loader, dev_loader, optimizer, scheduler, loss_fn, scaler, device, pad_src, pad_tgt, clip_norm, num_epochs, use_amp, checkpoint_dir, train_generator, config, start_epoch=1, best_val_loss=float("inf"), history=None, epochs_without_improvement=0):
    if history is None:
        history = []
    best_path = checkpoint_dir / "best_checkpoint.pt"
    last_path = checkpoint_dir / "last_checkpoint.pt"
    history_path = checkpoint_dir / "history.json"

    for epoch in range(start_epoch, num_epochs + 1):
        print("")
        print(f"Epoch {epoch}/{num_epochs} started")
        print()
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            pad_src, pad_tgt, clip_norm, scaler, use_amp,
            epoch, INTERVAL,
        )
        val_loss, val_acc = evaluate(
            model, dev_loader, loss_fn, device,
            pad_src, pad_tgt, use_amp,
        )
        scheduler.step(val_loss)
        epoch_seconds = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
        })

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print("")
        print(f"Epoch {epoch}/{num_epochs} complete")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.2f}%")
        print(f"LR         : {current_lr:.8f}")
        print(f"Epoch Time : {epoch_seconds/60:.2f} min ({epoch_seconds/3600:.2f} h)")
        print(f"No-improve epochs: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")

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
            config=config,
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
                config=config,
            )
            print(f"Best checkpoint saved: {best_path}")

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    return history


def save_checkpoint_atomic(path, *, epoch, model, optimizer, scheduler, scaler, best_val_loss, history, epochs_without_improvement, train_generator, config):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "config": config,
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

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if checkpoint.get("python_random_state") is not None:
        random.setstate(checkpoint["python_random_state"])
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    if train_generator is not None and checkpoint.get("train_generator_state") is not None:
        train_generator.set_state(checkpoint["train_generator_state"])
    return {
        "start_epoch": checkpoint["epoch"] + 1,
        "best_val_loss": checkpoint["best_val_loss"],
        "history": checkpoint.get("history", []),
        "epochs_without_improvement": checkpoint.get("epochs_without_improvement", 0),
    }

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last_checkpoint.pt.")
    args = parser.parse_args()
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available."
        )

    device = torch.device("cuda:0")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA device     : {torch.cuda.get_device_name(0)}")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    en_train_path = project_root / "data/clean/train.en"
    hi_train_path = project_root / "data/clean/train.hi"
    en_dev_path = project_root / "data/clean/dev.en"
    hi_dev_path = project_root / "data/clean/dev.hi"
    en_tokenizer_path = project_root / "output/models/en_unigram_500000_16000_0.9999.model"
    hi_tokenizer_path = project_root / "output/models/hi_unigram_500000_16000_0.9999.model"

    checkpoint_dir = project_root / "output/checkpoints/bilstm_bahdanau_seed19"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    last_path = checkpoint_dir / "last_checkpoint.pt"
    best_path = checkpoint_dir / "best_checkpoint.pt"
    history_path = checkpoint_dir / "history.json"

    if not args.resume and last_path.exists():
        raise RuntimeError(
            f"Existing checkpoint found at {last_path}. Use --resume to continue it."
        )

    print("Loading text files...")
    train_en = read_lines(en_train_path)
    train_hi = read_lines(hi_train_path)
    dev_en = read_lines(en_dev_path)
    dev_hi = read_lines(hi_dev_path)
    print(f"Raw train pairs : {len(train_en):,}")
    print(f"Raw dev pairs   : {len(dev_en):,}")

    tokenizer_en = spm.SentencePieceProcessor(model_file=str(en_tokenizer_path))
    tokenizer_hi = spm.SentencePieceProcessor(model_file=str(hi_tokenizer_path))

    bos_id_hi = tokenizer_hi.bos_id()
    eos_id_en = tokenizer_en.eos_id()
    eos_id_hi = tokenizer_hi.eos_id()
    pad_id_en = tokenizer_en.pad_id()
    pad_id_hi = tokenizer_hi.pad_id()
    vocab_size_en = tokenizer_en.vocab_size()
    vocab_size_hi = tokenizer_hi.vocab_size()

    print(f"EN vocab size   : {vocab_size_en:,}")
    print(f"HI vocab size   : {vocab_size_hi:,}")
    print(
        f"IDs: EN_EOS={eos_id_en}, HI_BOS={bos_id_hi}, HI_EOS={eos_id_hi}, "
        f"EN_PAD={pad_id_en}, HI_PAD={pad_id_hi}"
    )

    train_dataset = DataTransformation(
        train_en, train_hi, tokenizer_en, tokenizer_hi, max_len=MAX_TRAIN_CONTENT_LEN,
    )
    dev_dataset = DataTransformation(
        dev_en, dev_hi, tokenizer_en, tokenizer_hi, max_len=None,
    )

    del train_en, train_hi, dev_en, dev_hi
    gc.collect()

    print(f"Train dataset length: {len(train_dataset):,}")
    print(f"Dev dataset length  : {len(dev_dataset):,}")

    collate_fn = CollateFn(pad_src=pad_id_en, pad_tgt=pad_id_hi)
    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    train_loader = make_loader(
        train_dataset, collate_fn, TRAIN_BATCH_SIZE, NUM_WORKERS,
        shuffle=True, generator=train_generator,
    )
    dev_loader = make_loader(
        dev_dataset, collate_fn, DEV_BATCH_SIZE, NUM_WORKERS,
        shuffle=False, generator=None,
    )

    print(f"Train batches/epoch: {len(train_loader):,}")
    print(f"Dev batches         : {len(dev_loader):,}")

    model = build_model(vocab_size_en, vocab_size_hi, pad_id_en, pad_id_hi, device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id_hi)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    start_epoch = 1
    best_val_loss = float("inf")
    history = []
    epochs_without_improvement = 0

    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume requested but checkpoint does not exist: {last_path}")
        print(f"Loading checkpoint: {last_path}")
        state = load_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            train_generator=train_generator,
        )
        start_epoch = state["start_epoch"]
        best_val_loss = state["best_val_loss"]
        history = state["history"]
        epochs_without_improvement = state["epochs_without_improvement"]
        print(f"Resuming at epoch {start_epoch}.")

    print("")
    print("Final production configuration")
    print(f"  Batch size     : {TRAIN_BATCH_SIZE}")
    print(f"  Workers        : {NUM_WORKERS}")
    print(f"  AMP            : {USE_AMP}")
    print(f"  Learning Rate  : {LEARNING_RATE}")
    print(f"  Gradient clip  : {CLIP_NORM}")
    print(f"  Max epochs     : {MAX_EPOCHS}")
    print(f"  Checkpoints    : {checkpoint_dir}")
    print("")

    config = {
        "seed": SEED,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "encoder_dim": ENCODER_DIM,
        "decoder_dim": DECODER_DIM,
        "attention_dim": ATTENTION_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "clip_norm": CLIP_NORM,
        "max_train_content_len": MAX_TRAIN_CONTENT_LEN,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "dev_batch_size": DEV_BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "use_amp": USE_AMP,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "min_lr": MIN_LR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "vocab_size_en": vocab_size_en,
        "vocab_size_hi": vocab_size_hi,
        "train_pairs": len(train_dataset),
        "dev_pairs": len(dev_dataset),
        "en_tokenizer": str(en_tokenizer_path),
        "hi_tokenizer": str(hi_tokenizer_path),
    }

    fit(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        scaler=scaler,
        device=device,
        pad_src=pad_id_en,
        pad_tgt=pad_id_hi,
        clip_norm=CLIP_NORM,
        num_epochs=MAX_EPOCHS,
        use_amp=USE_AMP,
        checkpoint_dir=checkpoint_dir,
        train_generator=train_generator,
        config=config,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        history=history,
        epochs_without_improvement=epochs_without_improvement,
    )

    print("Training process finished.")


if __name__ == "__main__":
    main()
