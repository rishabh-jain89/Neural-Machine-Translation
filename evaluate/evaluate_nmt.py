import csv
import gc
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
from sacrebleu.metrics import BLEU, CHRF, TER
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SEED = 19
BATCH_SIZE = 32
MAX_NEW_TOKENS = 256
NUM_WORKERS = 2


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class TestDataset(Dataset):
    def __init__(self, en_texts, hi_texts, tokenizer_en, tokenizer_hi):
        assert len(en_texts) == len(hi_texts)

        self.en_texts = en_texts
        self.hi_texts = hi_texts

        self.en_ids = [tokenizer_en.encode_as_ids(text) for text in en_texts]
        self.hi_ids = [tokenizer_hi.encode_as_ids(text) for text in hi_texts]

        self.eos_src = tokenizer_en.eos_id()
        self.bos_tgt = tokenizer_hi.bos_id()
        self.eos_tgt = tokenizer_hi.eos_id()

    def __len__(self):
        return len(self.en_texts)

    def __getitem__(self, index):
        src = torch.tensor(self.en_ids[index] + [self.eos_src], dtype=torch.long)
        tgt_in = torch.tensor([self.bos_tgt] + self.hi_ids[index], dtype=torch.long)
        tgt_out = torch.tensor(self.hi_ids[index] + [self.eos_tgt], dtype=torch.long)

        return index, src, tgt_in, tgt_out, len(src)


class TestCollate:
    def __init__(self, pad_src, pad_tgt):
        self.pad_src = pad_src
        self.pad_tgt = pad_tgt

    def __call__(self, batch):
        indices = torch.tensor([item[0] for item in batch], dtype=torch.long)
        src = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=self.pad_src)
        tgt_in = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=self.pad_tgt)
        tgt_out = pad_sequence([item[3] for item in batch], batch_first=True, padding_value=self.pad_tgt)
        src_lens = torch.tensor([item[4] for item in batch], dtype=torch.long)

        return indices, src, tgt_in, tgt_out, src_lens


def make_loader(dataset, collate_fn):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )


def load_transformer(root, data_info, device):
    train_path = root / "transformer" / "train_transformer.py"
    checkpoint_path = root / "output" / "checkpoints" / "transformer_seed19" / "best_checkpoint.pt"

    module = load_module("baseline_transformer_train", train_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    saved_config = dict(checkpoint.get("config", {}))

    valid_fields = module.TransformerConfig.__dataclass_fields__

    config = module.TransformerConfig(
        **{
            key: value
            for key, value in saved_config.items()
            if key in valid_fields
        }
    )

    model_data = {
        "vocab_size_en": data_info["vocab_size_en"],
        "vocab_size_hi": data_info["vocab_size_hi"],
        "pad_id_en": data_info["pad_id_en"],
        "pad_id_hi": data_info["pad_id_hi"],
    }

    model = module.build_model(model_data, config, device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model, checkpoint_path


def load_bilstm(root, data_info, device):
    train_path = root / "BiLSTM" / "train_bilstm.py"
    checkpoint_path = root / "output" / "checkpoints" / "bilstm_bahdanau_seed19" / "best_checkpoint.pt"

    module = load_module("bilstm_train", train_path,)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    saved_config = checkpoint.get("config", {})

    mapping = {
        "embedding_dim": "EMBEDDING_DIM",
        "hidden_dim": "HIDDEN_DIM",
        "encoder_dim": "ENCODER_DIM",
        "decoder_dim": "DECODER_DIM",
        "attention_dim": "ATTENTION_DIM",
        "num_layers": "NUM_LAYERS",
        "dropout": "DROPOUT",
    }

    for config_name, module_name in mapping.items():
        if config_name in saved_config:
            setattr(module, module_name, saved_config[config_name])

    model = module.build_model(
        data_info["vocab_size_en"],
        data_info["vocab_size_hi"],
        data_info["pad_id_en"],
        data_info["pad_id_hi"],
        device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model, checkpoint_path


@torch.inference_mode()
def teacher_forced(model_name, model, loader, device, pad_src, pad_tgt, use_amp):
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_tgt)

    total_loss = torch.zeros((), dtype=torch.float64, device=device)

    total_tokens = torch.zeros((), dtype=torch.long, device=device)

    correct_tokens = torch.zeros((), dtype=torch.long, device=device)

    for _, src, tgt_in, tgt_out, src_lens in loader:
        src = src.to(device, non_blocking=True)
        tgt_in = tgt_in.to(device, non_blocking=True)
        tgt_out = tgt_out.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if model_name == "bilstm":
                logits = model(src, src_lens, tgt_in, src.eq(pad_src))
            else:
                logits = model(src, tgt_in, src.eq(pad_src), tgt_in.eq(pad_tgt))

            flat_logits = logits.reshape(-1, logits.size(-1))

            targets = tgt_out.reshape(-1)

            loss = loss_fn(flat_logits, targets)

        valid = targets.ne(pad_tgt)
        valid_tokens = valid.sum()

        total_loss += (loss.detach().double() * valid_tokens)

        total_tokens += valid_tokens

        predictions = flat_logits.argmax(dim=-1)

        correct_tokens += ((predictions == targets) & valid).sum()

    mean_loss = (total_loss / total_tokens).item()

    token_accuracy = (correct_tokens.double() / total_tokens * 100.0).item()

    return {
        "cross_entropy": mean_loss,
        "perplexity": math.exp(mean_loss),
        "token_accuracy": token_accuracy,
    }


@torch.inference_mode()
def transformer_greedy(model, src, pad_src, pad_tgt, bos_tgt, eos_tgt, use_amp):
    src_mask = src.eq(pad_src)

    with torch.amp.autocast("cuda", enabled=use_amp):
        memory = model.encode(src, src_mask)

    output = torch.full((src.size(0), 1), bos_tgt, dtype=torch.long, device=src.device)

    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

    for _ in range(MAX_NEW_TOKENS):
        with torch.amp.autocast("cuda", enabled=use_amp):
            decoded = model.decode(output, memory, output.eq(pad_tgt), src_mask)

            next_token = model.output_projection(decoded[:, -1]).argmax(dim=-1)

        next_token = torch.where(
            finished,
            torch.full_like(next_token, pad_tgt),
            next_token,
        )

        output = torch.cat([output, next_token[:, None]], dim=1)

        finished |= next_token.eq(eos_tgt)

        if bool(finished.all()):
            break

    return output


@torch.inference_mode()
def bilstm_greedy(model, src, src_lens, pad_src, pad_tgt, bos_tgt, eos_tgt, use_amp):
    src_mask = src.eq(pad_src)

    with torch.amp.autocast("cuda", enabled=use_amp):
        encoder_output, hidden, cell = model.encoder(src, src_lens)

        projected_encoder = model.decoder.attention.project_encoder(encoder_output)

    current_token = torch.full((src.size(0),), bos_tgt, dtype=torch.long, device=src.device)

    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

    generated = []

    for _ in range(MAX_NEW_TOKENS):
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, hidden, cell, _ = model.decoder(
                current_token,
                hidden,
                cell,
                encoder_output,
                projected_encoder,
                src_mask,
            )

        next_token = logits.argmax(dim=-1)

        next_token = torch.where(
            finished,
            torch.full_like(next_token, pad_tgt),
            next_token,
        )

        generated.append(next_token)

        finished |= next_token.eq(eos_tgt)
        current_token = next_token

        if bool(finished.all()):
            break

    if generated:
        output = torch.stack(generated, dim=1)
    else:
        output = torch.empty(
            src.size(0),
            0,
            dtype=torch.long,
            device=src.device,
        )

    bos_column = torch.full(
        (src.size(0), 1),
        bos_tgt,
        dtype=torch.long,
        device=src.device,
    )

    return torch.cat(
        [bos_column, output],
        dim=1,
    )


def clean_ids(row, bos_tgt, eos_tgt, pad_tgt):
    ids = []

    for token in row:
        token = int(token)

        if token == bos_tgt:
            continue

        if token == eos_tgt:
            break

        if token != pad_tgt:
            ids.append(token)

    return ids


@torch.inference_mode()
def generate_predictions(model_name, model, loader, tokenizer_hi, device, data_info, use_amp):
    hypotheses = [None] * len(loader.dataset)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for batch_index, batch in enumerate(loader, start=1):
        indices, src, _, _, src_lens = batch

        src = src.to(device, non_blocking=True)

        if model_name == "bilstm":
            generated = bilstm_greedy(
                model,
                src,
                src_lens,
                data_info["pad_id_en"],
                data_info["pad_id_hi"],
                data_info["bos_id_hi"],
                data_info["eos_id_hi"],
                use_amp,
            )
        else:
            generated = transformer_greedy(
                model,
                src,
                data_info["pad_id_en"],
                data_info["pad_id_hi"],
                data_info["bos_id_hi"],
                data_info["eos_id_hi"],
                use_amp,
            )

        for row, index in zip(generated.cpu().tolist(), indices.tolist()):
            ids = clean_ids(
                row,
                data_info["bos_id_hi"],
                data_info["eos_id_hi"],
                data_info["pad_id_hi"],
            )

            hypotheses[index] = tokenizer_hi.decode_ids(ids)

        if (batch_index % 20 == 0 or batch_index == len(loader)):
            print(
                f"{model_name}: "
                f"{min(batch_index * BATCH_SIZE, len(loader.dataset)):,}/"
                f"{len(loader.dataset):,}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    return (
        hypotheses,
        time.perf_counter() - start,
    )


def compute_metrics(hypotheses, references):
    bleu = BLEU()
    chrf = CHRF(word_order=2)
    ter = TER()

    return {
        "BLEU": bleu.corpus_score(
            hypotheses,
            [references],
        ).score,
        "chrF++": chrf.corpus_score(
            hypotheses,
            [references],
        ).score,
        "TER": ter.corpus_score(
            hypotheses,
            [references],
        ).score,
        "BLEU_signature": str(
            bleu.get_signature()
        ),
        "chrF_signature": str(
            chrf.get_signature()
        ),
        "TER_signature": str(
            ter.get_signature()
        ),
    }


def save_predictions(path, dataset, bilstm_predictions, transformer_predictions):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "id",
            "source_en",
            "reference_hi",
            "src_sp_tokens",
            "bilstm_greedy",
            "transformer_greedy",
        ])

        for index in range(len(dataset)):
            writer.writerow([
                index,
                dataset.en_texts[index],
                dataset.hi_texts[index],
                len(dataset.en_ids[index]),
                bilstm_predictions[index],
                transformer_predictions[index],
            ])


def main():
    set_seed(SEED)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_amp = device.type == "cuda"

    print("COMMON NMT TEST EVALUATION")
    print("=" * 72)
    print(f"Project root : {project_root}")
    print(f"Device       : {device}")

    if device.type == "cuda":
        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    tokenizer_en = spm.SentencePieceProcessor(model_file=str(project_root / "output" / "models" / "en_unigram_500000_16000_0.9999.model"))

    tokenizer_hi = spm.SentencePieceProcessor(model_file=str(project_root / "output" / "models" / "hi_unigram_500000_16000_0.9999.model"))

    test_en = read_lines(project_root / "data" / "clean" / "test.en")

    test_hi = read_lines(project_root / "data" / "clean" / "test.hi")

    print(f"Test pairs   : {len(test_en):,}")

    data_info = {
        "vocab_size_en": tokenizer_en.vocab_size(),
        "vocab_size_hi": tokenizer_hi.vocab_size(),
        "pad_id_en": tokenizer_en.pad_id(),
        "pad_id_hi": tokenizer_hi.pad_id(),
        "bos_id_hi": tokenizer_hi.bos_id(),
        "eos_id_hi": tokenizer_hi.eos_id(),
    }

    dataset = TestDataset(test_en, test_hi, tokenizer_en, tokenizer_hi)

    collate_fn = TestCollate(data_info["pad_id_en"], data_info["pad_id_hi"])

    loader = make_loader(dataset, collate_fn)

    output_dir = (project_root / "output" / "evaluation")

    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    predictions = {}

    model_loaders = {"bilstm": load_bilstm, "transformer": load_transformer}

    for model_name, load_model in model_loaders.items():
        model, checkpoint_path = load_model(project_root, data_info, device)

        parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print()
        print(f"EVALUATING {model_name.upper()}")
        print()
        print(f"Checkpoint : {checkpoint_path}")
        print(f"Parameters : {parameters}")

        teacher_metrics = teacher_forced(model_name, model, loader, device, data_info["pad_id_en"], data_info["pad_id_hi"], use_amp)

        hypotheses, decode_seconds = (
            generate_predictions(
                model_name,
                model,
                loader,
                tokenizer_hi,
                device,
                data_info,
                use_amp,
            )
        )

        mt_metrics = compute_metrics(
            hypotheses,
            test_hi,
        )

        summaries[model_name] = {
            "checkpoint": str(checkpoint_path),
            "parameters": parameters,
            "teacher_forced": teacher_metrics,
            "greedy": {**mt_metrics, "decode_seconds": decode_seconds},
        }

        predictions[model_name] = hypotheses

        print(
            f"CE={teacher_metrics['cross_entropy']:.4f} | "
            f"PPL={teacher_metrics['perplexity']:.4f} | "
            f"TokAcc={teacher_metrics['token_accuracy']:.2f}%"
        )

        print(
            f"BLEU={mt_metrics['BLEU']:.4f} | "
            f"chrF++={mt_metrics['chrF++']:.4f} | "
            f"TER={mt_metrics['TER']:.4f}"
        )

        del model
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    prediction_path = (output_dir / "test_predictions.csv")

    save_predictions(prediction_path, dataset, predictions["bilstm"], predictions["transformer"])

    summary_path = (output_dir / "evaluation_summary.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_pairs": len(dataset),
                "decoding": "greedy",
                "batch_size": BATCH_SIZE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "models": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("EVALUATION COMPLETE")
    print(f"Predictions : {prediction_path}")
    print(f"Summary     : {summary_path}")


if __name__ == "__main__":
    main()
