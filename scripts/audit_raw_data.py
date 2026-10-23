import random 
import numpy as np
from pathlib import Path
from collections import Counter

random.seed(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

raw_en = PROJECT_ROOT / "data/raw/parallel-n/IITB.en-hi.en"
raw_hi = PROJECT_ROOT / "data/raw/parallel-n/IITB.en-hi.hi"
raw_dev_en = PROJECT_ROOT / "data/raw/dev_test/dev.en"
raw_dev_hi = PROJECT_ROOT / "data/raw/dev_test/dev.hi"
raw_test_en = PROJECT_ROOT / "data/raw/dev_test/test.en"
raw_test_hi = PROJECT_ROOT / "data/raw/dev_test/test.hi"

with open(raw_en, 'r', encoding='utf-8') as english_file:
    english_lines = english_file.readlines()

with open(raw_hi, 'r', encoding='utf-8') as hindi_file:
    hindi_lines = hindi_file.readlines()

with open(raw_dev_en, 'r', encoding='utf-8') as english_file_dev:
    english_lines_dev = english_file_dev.readlines()

with open(raw_dev_hi, 'r', encoding='utf-8') as hindi_file_dev:
    hindi_lines_dev = hindi_file_dev.readlines()

with open(raw_test_en, 'r', encoding='utf-8') as english_file_test:
    english_lines_test = english_file_test.readlines()

with open(raw_test_hi, 'r', encoding='utf-8') as hindi_file_test:
    hindi_lines_test = hindi_file_test.readlines()

print("Number of lines in English:", len(english_lines))
print("Number of lines in Hindi:", len(hindi_lines))

print("Number of lines in English dev:", len(english_lines_dev))
print("Number of lines in Hindi dev:", len(hindi_lines_dev))

print("Number of lines in English Test:", len(english_lines_test))
print("Number of lines in Hindi Test:", len(hindi_lines_test))

print()

empty_eng = 0
empty_hin = 0
empty_either = 0
empty_both = 0

for line in english_lines:
    if len(line.strip()) == 0:
        empty_eng += 1

for line in hindi_lines:
    if len(line.strip()) == 0:
        empty_hin += 1

merged_train = list(zip(english_lines, hindi_lines))

for line in merged_train:
    if len(line[0].strip()) == 0 or len(line[1].strip()) == 0:
        empty_either += 1

    if len(line[0].strip()) == 0 and len(line[1].strip()) == 0:
        empty_both += 1

print("Empty sentences in English:", empty_eng)
print("Empty sentences in Hindi:", empty_hin)
print("Empty sentences in either language:", empty_either)
print("Empty sentences in both language:", empty_both)

eng_lens = [len(line.split()) for line in english_lines]
hin_lens = [len(line.split()) for line in hindi_lines]

print("Length of eng_lens list:", len(eng_lens))
print("Length of hin_lens list:", len(hin_lens))

eng_mean = np.mean(eng_lens)
hin_mean = np.mean(hin_lens)

eng_median = np.median(eng_lens)
hin_median = np.median(hin_lens)

eng_90th_per = np.percentile(eng_lens, 90)
eng_95th_per = np.percentile(eng_lens, 95)
eng_99th_per = np.percentile(eng_lens, 99)
hin_90th_per = np.percentile(hin_lens, 90)
hin_95th_per = np.percentile(hin_lens, 95)
hin_99th_per = np.percentile(hin_lens, 99)

eng_max = np.max(eng_lens)
hin_max = np.max(hin_lens)

print("English mean:", eng_mean)
print("Hindi mean:", hin_mean)

print("English median:", eng_median)
print("Hindi median:", hin_median)

print("English 90th percentile:", eng_90th_per)
print("English 95th percentile:", eng_95th_per)
print("English 99th percentile:", eng_99th_per)

print("Hindi 90th percentile:", hin_90th_per)
print("Hindi 95th percentile:", hin_95th_per)
print("Hindi 99th percentile:", hin_99th_per)

print("English Maximum length:", eng_max)
print("Hindi Maximum length:", hin_max)

eng_long50 = len(list(filter(lambda line: len(line.split()) >= 50, english_lines)))
eng_long100 = len(list(filter(lambda line: len(line.split()) >= 100, english_lines)))
eng_long200 = len(list(filter(lambda line: len(line.split()) >= 200, english_lines)))

hin_long50 = len(list(filter(lambda line: len(line.split()) >= 50,hindi_lines)))
hin_long100 = len(list(filter(lambda line: len(line.split()) >= 100,hindi_lines)))
hin_long200 = len(list(filter(lambda line: len(line.split()) >= 200,hindi_lines)))

print("English sentences with length greater than or equal to 50:", eng_long50)
print("English sentences with length greater than or equal to 100:", eng_long100)
print("English sentences with length greater than or equal to 200:", eng_long200)
print("Hindi sentences with length greater than or equal to 50:", hin_long50)
print("Hindi sentences with length greater than or equal to 100:", hin_long100)
print("Hindi sentences with length greater than or equal to 200:", hin_long200)

ratios = [(max(eng_len, hin_len)/min(eng_len, hin_len), i) for i, (eng_len, hin_len) in enumerate(zip(eng_lens, hin_lens)) if eng_len != 0 and hin_len != 0]

ratio_g2 = sum(1 for ratio in ratios if ratio[0] > 2)
ratio_g3 = sum(1 for ratio in ratios if ratio[0] > 3)
ratio_g5 = sum(1 for ratio in ratios if ratio[0] > 5)

print("Length Ratio greater than 2:", ratio_g2)
print("Length Ratio greater than 3:", ratio_g3)
print("Length Ratio greater than 5:", ratio_g5)

train_dup = 0
train_dev_dup = 0
train_test_dup = 0

merged_dev = list(zip(english_lines_dev, hindi_lines_dev))
merged_test = list(zip(english_lines_test, hindi_lines_test))

train_set = set(merged_train)
dev_set = set(merged_dev)
test_set = set(merged_test)

train_dup  = len(merged_train) - len(train_set)
train_dev_dup = sum(1 for pair in dev_set if pair in train_set)
train_test_dup = sum(1 for pair in test_set if pair in train_set)

print("Duplicates in Training set with itself:", train_dup)
print("Overlapping with Training and Dev set:", train_dev_dup)
print("Overlapping with Training and Test set:", train_test_dup)

train_counts = Counter(merged_train)
duplicate_pairs = [(pair, count) for pair, count in train_counts.items() if count > 1]

print("Unique pairs appearing more than once:", len(duplicate_pairs))
print("First 10 duplicate pairs:")
for pair, count in duplicate_pairs[:10]:
    print("Count:", count)
    print("English:", repr(pair[0]))
    print("Hindi:", repr(pair[1]))
    print()

ratios_sorted = sorted(ratios, reverse=True)
ratio_top10 = ratios_sorted[:10]

print("Top 10 Sentences with highest ratio:")
for (ratio, ind) in ratio_top10:
    print("Ratio:", ratio)
    print("English Length:", eng_lens[ind])
    print("Hindi Length:", hin_lens[ind])
    print("English:", merged_train[ind][0])
    print("Hindi:", merged_train[ind][1])
    print()

empty_eng_pair = []
empty_hin_pair = []

for i in range(0, len(english_lines)):
    if len(english_lines[i].strip()) == 0 and len(hindi_lines[i].strip()) != 0:
        empty_eng_pair.append((english_lines[i], hindi_lines[i]))

for i in range(0, len(hindi_lines)):
    if len(hindi_lines[i].strip()) == 0 and len(english_lines[i].strip()) != 0:
        empty_hin_pair.append((english_lines[i], hindi_lines[i]))

print("Empty English pairs Count:", len(empty_eng_pair))
for (eng, hin) in empty_eng_pair[:5]:
    print("English:", eng)
    print("Hindi:", hin)
    print()

print("Empty Hindi pairs Count:", len(empty_hin_pair))
for (eng, hin) in empty_hin_pair[:5]:
    print("English:", eng)
    print("Hindi:", hin)
    print()

top10_dup = sorted(duplicate_pairs, key = lambda item: item[1], reverse=True)[:10]

for pair, count in top10_dup:
    print("Count:", count)
    print("English:", pair[0])
    print("Hindi:", pair[1])
    print()

random10 = random.sample(merged_train, k=10)

for pair in random10:
    print("English:", pair[0])
    print("Hindi:", pair[1])
    print()

eng_train_set = set(english_lines)
eng_dev_set = set(english_lines_dev)
eng_test_set = set(english_lines_test)

eng_train_dev_overlap = sum(1 for x in eng_dev_set if x in eng_train_set)
eng_train_test_overlap = sum(1 for x in eng_test_set if x in eng_train_set)
eng_dev_test_overlap = sum(1 for x in eng_dev_set if x in eng_test_set)

print("English sentences overlapping in train and dev set:", eng_train_dev_overlap)
print("English sentences overlapping in train and test set:", eng_train_test_overlap)
print("English sentences overlapping in dev and test set:", eng_dev_test_overlap)

sample_ratio_band_2_3 = random.sample(list(filter(lambda item: item[0] >= 2 and item[0] < 3, ratios)), k=20)
sample_ratio_band_3_4 = random.sample(list(filter(lambda item: item[0] >= 3 and item[0] < 4, ratios)), k=20)
sample_ratio_band_4_5 = random.sample(list(filter(lambda item: item[0] >= 4 and item[0] < 5, ratios)), k=20)
sample_ratio_band_5_8 = random.sample(list(filter(lambda item: item[0] >= 5 and item[0] < 8, ratios)), k=20)
sample_ratio_band_g8 = random.sample(list(filter(lambda item: item[0] >= 8, ratios)), k=20)

print("Band 2-3:")
for (ratio, ind) in sample_ratio_band_2_3:
    print("English:", english_lines[ind])
    print("Hindi:", hindi_lines[ind])

print("Band 3-4:")
for (ratio, ind) in sample_ratio_band_3_4:
    print("English:", english_lines[ind])
    print("Hindi:", hindi_lines[ind])

print("Band 4-5:")
for (ratio, ind) in sample_ratio_band_4_5:
    print("English:", english_lines[ind])
    print("Hindi:", hindi_lines[ind])

print("Band 5-8:")
for (ratio, ind) in sample_ratio_band_5_8:
    print("English:", english_lines[ind])
    print("Hindi:", hindi_lines[ind])

print("Band >8:")
for (ratio, ind) in sample_ratio_band_g8:
    print("English:", english_lines[ind])
    print("Hindi:", hindi_lines[ind])