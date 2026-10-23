import sentencepiece as spm
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

en_path = PROJECT_ROOT / "data/clean/train.en"
hi_path = PROJECT_ROOT / "data/clean/train.hi"

en_tokenizer_path = PROJECT_ROOT / "output/models/en_unigram_500000_16000_0.9999.model"
hi_tokenizer_path = PROJECT_ROOT / "output/models/hi_unigram_500000_16000_0.9999.model"

with open(en_path, "r", encoding = "utf-8") as train_file:
    train_en = train_file.readlines()

with open(hi_path, "r", encoding = "utf-8") as train_file:
    train_hi = train_file.readlines()

train_en = [item.rstrip("\n") for item in train_en]
train_hi = [item.rstrip("\n") for item in train_hi]

tokenizer_en = spm.SentencePieceProcessor(model_file=str(en_tokenizer_path))
tokenizer_hi = spm.SentencePieceProcessor(model_file=str(hi_tokenizer_path))

random.seed(5)

sample = random.sample(list(zip(train_en, train_hi)), k = 5)

for i, (en_text, hi_text) in enumerate(sample):
    print("Sample :", i+1)
    print("English:", en_text)
    print("Hindi:", hi_text)

    pieces_en = tokenizer_en.encode_as_pieces(en_text)
    ids_en = tokenizer_en.encode_as_ids(en_text)

    pieces_hi = tokenizer_hi.encode_as_pieces(hi_text)
    ids_hi = tokenizer_hi.encode_as_ids(hi_text)

    print("English Pieces:",pieces_en)
    print("English ids:", ids_en)
    print()
    print("Hindi Pieces:", pieces_hi)
    print("Hindi ids:", ids_hi)
    print()

    src_id = ids_en + [tokenizer_en.eos_id()]
    decoder_in_id = [tokenizer_hi.bos_id()] + ids_hi
    decoder_out_id = ids_hi + [tokenizer_hi.eos_id()]

    print("Source ID:", src_id)
    print("Decoder Input ID:", decoder_in_id)
    print("Decoder Output ID:", decoder_out_id)
    print()

