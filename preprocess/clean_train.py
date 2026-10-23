import re
import html
import random
import unicodedata
from pathlib import Path

random.seed(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

raw_en = PROJECT_ROOT / "data/raw/parallel-n/IITB.en-hi.en"
raw_hi = PROJECT_ROOT / "data/raw/parallel-n/IITB.en-hi.hi"
clean_en = PROJECT_ROOT / "data/clean/train.en"
clean_hi = PROJECT_ROOT / "data/clean/train.hi"

with open(raw_en, 'r', encoding='utf-8') as english_file:
    eng = english_file.readlines()

with open(raw_hi, 'r', encoding='utf-8') as hindi_file:
    hin = hindi_file.readlines()

def normalize(pair):
    eng_text = pair[0].strip()
    hin_text = pair[1].strip()

    eng_text = " ".join(eng_text.split())
    hin_text = " ".join(hin_text.split())

    eng_text = unicodedata.normalize("NFC", eng_text)
    hin_text = unicodedata.normalize("NFC", hin_text)

    return (eng_text, hin_text)

def identify_corrupted(pair):
    marker_list = ["à¤","à¥","Ã","Â","�"]

    eng_text = pair[0]
    hin_text = pair[1]

    for marker in marker_list:
        if marker in eng_text or marker in hin_text:
            return True

    return False

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
changed_html = []

empty = 0
corrupted = 0
empty_after_cleaning = 0
duplicate = 0
length = 0
ratio_counter = 0

assert len(eng) == len(hin), "English, Hindi Number of samples mismatch"
merged_pair = list(zip(eng, hin))

for i in range(0, len(merged_pair)):
    normalized_pair = normalize(merged_pair[i])

    (eng_line, hin_line) = normalized_pair

    if len(eng_line) == 0 or len(hin_line) == 0:
        empty += 1
        continue

    if identify_corrupted(normalized_pair):
        corrupted += 1
        continue

    clean_pair = clean_markup(normalized_pair)
    if clean_pair != normalized_pair:
        changed_html.append((normalized_pair, clean_pair))

    clean_normalized = normalize(clean_pair)

    (eng_line, hin_line) = clean_normalized

    if len(eng_line) == 0 or len(hin_line) == 0:
        empty_after_cleaning += 1
        continue

    if clean_normalized in seen:
        duplicate += 1
        continue

    len_eng = len(clean_normalized[0].split())
    len_hin = len(clean_normalized[1].split())

    if len_eng > 100 or len_hin > 100:
        length += 1
        continue

    ratio = max(len_eng, len_hin)/min(len_eng, len_hin)

    if ratio > 5:
        ratio_counter += 1
        continue

    seen.add(clean_normalized)
    accepted_list.append(clean_normalized)

print("Number of pairs survived:", len(seen))
print("Number of pairs which were empty:", empty)
print("Number of pairs which were corrupted:", corrupted)
print("Number of pairs which were empty after cleaning html:", empty_after_cleaning)
print("Number of pairs which were duplicates:", duplicate)
print("Number of pairs which had either length greater than 100:", length)
print("Number of pairs which had ratio greater than 5:", ratio_counter)

sample = random.sample(accepted_list, k=10)

print("10 random samples from the survived set:")
for pair in sample:
    print("English:", pair[0])
    print("Hindi:", pair[1])
    print()

sample_html = random.sample(changed_html, k=min(10, len(changed_html)))

print("30 html corrected samples:")
for sample in sample_html:
    print("Original English:", sample[0][0])
    print("Original Hindi:", sample[0][1])
    print()
    print("Changed English:", sample[1][0])
    print("Changed Hindi:", sample[1][1])
    print()
    print("=======")
    print()

eng_train = []
hin_train = []

for pair in accepted_list:
    eng_train.append(pair[0])
    hin_train.append(pair[1])

with open(clean_en, 'w', encoding = 'utf-8') as train_eng:
    for eng_line in eng_train:
        train_eng.write(f"{eng_line}\n")

with open(clean_hi, 'w', encoding = 'utf-8') as train_hin:
    for hin_line in hin_train:
        train_hin.write(f"{hin_line}\n")