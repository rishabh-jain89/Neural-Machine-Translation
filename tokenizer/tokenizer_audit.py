import sentencepiece as spm
import numpy as np
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

clean_dev_en = PROJECT_ROOT / "data/clean/dev.en"

with open(clean_dev_en, "r", encoding = "utf-8") as dev_file:
    en_dev = dev_file.readlines()

models = ["en_unigram_500000_16000_0.9999", "en_bpe_500000_16000_0.9999"]
models = [str(PROJECT_ROOT / "output/models" / (item + ".model")) for item in models]

for model in models:
    tokenizer = spm.SentencePieceProcessor(model_file = model)
    sen_unk = 0
    word_count = 0
    token_count = 0
    unk_count = 0
    char_count = 0
    mismatch = 0
    sample_unk = []
    length = []
    unique_tokens = set()
    token_freq = Counter()
    for text in en_dev:
        text = text.rstrip("\n")
        ids = tokenizer.encode(text)

        token_count += len(ids)
        unk_count += ids.count(0)
        word_count += len(text.split())
        char_count += len(text)
        token_freq.update(ids)

        length.append(len(ids))

        if 0 in ids:
            sample_unk.append((text, ids))
            sen_unk += 1

        conv = tokenizer.decode(ids)
        if text != conv:
            mismatch += 1

        for id in ids:
            if id not in unique_tokens:
                unique_tokens.add(id)

    unk_rate = (unk_count/token_count) * 100
    length = np.array(length)
    rare_tokens = []
    for token, count in token_freq.most_common()[-10:]:
        rare_tokens.append((tokenizer.decode(token), count))
    print("Dev Set English with tokenizer: ", model)
    print("Unknown Rate:", unk_rate)
    print("Number of sentences with at least 1 unknown:", sen_unk)
    print("Number of sentences differing in Round Trip Conversion:", mismatch)
    print("Mean Tokens per sentence:", np.mean(length))
    print("Median Tokenized Length:", np.median(length))
    print("90th percentile Length:", np.percentile(length, 90))
    print("95th percentile Length:", np.percentile(length, 95))
    print("99th percentile Length:", np.percentile(length, 99))
    print("100th percentile Length:", np.percentile(length, 100))
    print("Percentage Exceeding 64 Length:", np.mean(length > 64) * 100)
    print("Percentage Exceeding 128 Length:", np.mean(length > 128) * 100)
    print("Percentage Exceeding 160 Length:", np.mean(length > 160) * 100)
    print("Percentage Exceeding 192 Length:", np.mean(length > 192) * 100)
    print("Percentage Exceeding 256 Length:", np.mean(length > 256) * 100)
    print("Fragmentation:", token_count/word_count)
    print("Characters per token:", char_count/token_count)
    print("Unique Token Count:", len(unique_tokens))
    print("Vocabulary Utilization:", (len(unique_tokens)/tokenizer.get_piece_size())*100)
    print("Rare tokens:", rare_tokens)
    print("Sentences containing unknown character:", sample_unk)
    print()
    print("=====================")
    print()