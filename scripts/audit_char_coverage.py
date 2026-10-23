from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

clean_en = PROJECT_ROOT / "data/clean/train.en"
clean_hi = PROJECT_ROOT / "data/clean/train.hi"

with open(clean_en, "r", encoding = "utf-8") as train_file:
    train_eng = train_file.readlines()

with open(clean_hi, "r", encoding = "utf-8") as train_file:
    train_hin = train_file.readlines()

eng_freq = Counter("".join(train_eng))
hin_freq = Counter("".join(train_hin))

print("Number of unique characters in English:", len(eng_freq))
print("Number of unique characters in Hindi:", len(hin_freq))
print("20 Least Common English characters:", eng_freq.most_common()[-20:])
print("20 Least Common Hindi characters:", hin_freq.most_common()[-20:])

eng_freq = sorted(eng_freq.items(), key = lambda item: item[1], reverse = True)
hin_freq = sorted(hin_freq.items(), key = lambda item: item[1], reverse = True)

total_eng = sum(count for key, count in eng_freq)
total_hin = sum(count for key, count in hin_freq)

current_eng = 0
eng_coverage = []
threshold = [99, 99.9, 99.95, 99.99, 100]
ind = 0

for i, (key, count) in enumerate(eng_freq):
    current_eng += count
    percent = (current_eng/total_eng) * 100

    if percent >= threshold[ind]:
        eng_coverage.append((threshold[ind], i+1))
        ind += 1

print("English coverage Statistics:")
for pair in eng_coverage:
    print("Threshold:", pair[0])
    print("Characters required:", pair[1])
    print()

current_hin = 0
hin_coverage = []
ind = 0

for i, (key, count) in enumerate(hin_freq):
    current_hin += count
    percent = (current_hin/total_hin) * 100

    if percent >= threshold[ind]:
        hin_coverage.append((threshold[ind], i+1))
        ind += 1

print("Hindi coverage Statistics:")
for pair in hin_coverage:
    print("Threshold:", pair[0])
    print("Characters required:", pair[1])
    print()