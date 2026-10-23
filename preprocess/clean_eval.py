import re
import html
import random
import unicodedata
from pathlib import Path

random.seed(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

raw_dev_en = PROJECT_ROOT / "data/raw/dev_test/dev.en"
raw_dev_hi = PROJECT_ROOT / "data/raw/dev_test/dev.hi"
raw_test_en = PROJECT_ROOT / "data/raw/dev_test/test.en"
raw_test_hi = PROJECT_ROOT / "data/raw/dev_test/test.hi"

clean_dev_en = PROJECT_ROOT / "data/clean/dev.en"
clean_dev_hi = PROJECT_ROOT / "data/clean/dev.hi"
clean_test_en = PROJECT_ROOT / "data/clean/test.en"
clean_test_hi = PROJECT_ROOT / "data/clean/test.hi"

with open(raw_dev_en, 'r', encoding='utf-8') as english_file_test:
    eng_dev = english_file_test.readlines()

with open(raw_dev_hi, 'r', encoding='utf-8') as hindi_file_dev:
    hin_dev = hindi_file_dev.readlines()

with open(raw_test_en, 'r', encoding='utf-8') as english_file_test:
    eng_test = english_file_test.readlines()

with open(raw_test_hi, 'r', encoding='utf-8') as hindi_file_test:
    hin_test = hindi_file_test.readlines()

def normalize(pair):
    eng_text = pair[0].strip()
    hin_text = pair[1].strip()

    eng_text = " ".join(eng_text.split())
    hin_text = " ".join(hin_text.split())

    eng_text = unicodedata.normalize("NFC", eng_text)
    hin_text = unicodedata.normalize("NFC", hin_text)

    return (eng_text, hin_text)

def clean_markup(pair):
    eng_text = html.unescape(pair[0])
    hin_text = html.unescape(pair[1])

    simple_tags = ["b", "i", "em", "strong", "small", "s", "u", "p", "br"]
    pattern = rf"<\s*/?\s*(?:{'|'.join(simple_tags)})\s*/?\s*>"

    eng_text = re.sub(pattern, "", eng_text, flags = re.IGNORECASE)
    hin_text = re.sub(pattern, "", hin_text, flags = re.IGNORECASE)

    return (eng_text, hin_text)

seen = set()
accepted_list = []

empty = 0
empty_after_cleaning = 0

merged_pair_dev = list(zip(eng_dev, hin_dev))

for i in range(0, len(merged_pair_dev)):
    normalized_pair = normalize(merged_pair_dev[i])

    (eng_line, hin_line) = normalized_pair

    if len(eng_line) == 0 or len(hin_line) == 0:
        empty += 1
        continue

    clean_pair = clean_markup(normalized_pair)

    clean_normalized = normalize(clean_pair)

    (eng_line, hin_line) = clean_normalized

    if len(eng_line) == 0 or len(hin_line) == 0:
        empty_after_cleaning += 1
        continue

    seen.add(clean_normalized)
    accepted_list.append(clean_normalized)

print(f"Length of Original and Accepted Dev set:{len(merged_pair_dev)}   {len(accepted_list)}")
print("Number of pairs survived:", len(seen))
print("Number of pairs which were empty:", empty)
print("Number of pairs which were empty after cleaning html:", empty_after_cleaning)

sample = random.sample(accepted_list, k=10)

print("10 random samples from the survived set:")
for pair in sample:
    print("English:", pair[0])
    print("Hindi:", pair[1])
    print()

eng_dev_final = []
hin_dev_final = []

for pair in accepted_list:
    eng_dev_final.append(pair[0])
    hin_dev_final.append(pair[1])

seen = set()
accepted_list = []

empty = 0
empty_after_cleaning = 0

merged_pair_test = list(zip(eng_test, hin_test))

for i in range(0, len(merged_pair_test)):
    normalized_pair = normalize(merged_pair_test[i])

    (eng_line, hin_line) = normalized_pair

    if len(eng_line) == 0 or len(hin_line) == 0:
        empty += 1
        continue

    clean_pair = clean_markup(normalized_pair)

    clean_normalized = normalize(clean_pair)

    (eng_line, hin_line) = clean_normalized

    seen.add(clean_normalized)
    accepted_list.append(clean_normalized)

print(f"Length of Original and Accepted Test set:{len(merged_pair_test)}   {len(accepted_list)}")
print("Number of pairs survived:", len(accepted_list))
print("Number of pairs which were empty:", empty)
print("Number of pairs which were empty after cleaning html:", empty_after_cleaning)

sample = random.sample(accepted_list, k=10)

print("10 random samples from the survived set:")
for pair in sample:
    print("English:", pair[0])
    print("Hindi:", pair[1])
    print()

eng_test_final = []
hin_test_final = []

for pair in accepted_list:
    eng_test_final.append(pair[0])
    hin_test_final.append(pair[1])

with open(clean_dev_en, 'w', encoding = 'utf-8') as dev_eng:
    for eng_line in eng_dev_final:
        dev_eng.write(f"{eng_line}\n")

with open(clean_dev_hi, 'w', encoding = 'utf-8') as dev_hin:
    for hin_line in hin_dev_final:
        dev_hin.write(f"{hin_line}\n")

with open(clean_test_en, 'w', encoding = 'utf-8') as test_eng:
    for eng_line in eng_test_final:
        test_eng.write(f"{eng_line}\n")

with open(clean_test_hi, 'w', encoding = 'utf-8') as test_hin:
    for hin_line in hin_test_final:
        test_hin.write(f"{hin_line}\n")

