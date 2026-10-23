from pathlib import Path
import time
import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

files = {
    "en": PROJECT_ROOT / "data/clean/train.en",
    "hi": PROJECT_ROOT / "data/clean/train.hi",
}

model_type = "unigram"
char_cov = 0.9999
vocab_size = 16000
input_size = 500000

output_dir = PROJECT_ROOT / "output/models"
output_dir.mkdir(parents=True, exist_ok=True)


for lang, file in files.items():
    prefix = f"{lang}_{model_type}_{input_size}_{vocab_size}_{char_cov}"
    model_prefix = str(output_dir / prefix)
    start_time = time.perf_counter()

    spm.SentencePieceTrainer.train(
        input=str(file),
        model_prefix=model_prefix,
        model_type=model_type,
        character_coverage=char_cov,
        vocab_size=vocab_size,
        input_sentence_size=input_size,
        normalization_rule_name="identity",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        num_threads = 16,
    )

    elapsed = time.perf_counter() - start_time
    print(f"DONE {prefix} elapsed={elapsed:.1f}s")