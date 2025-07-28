import csv
from collections import Counter
import os
import csv
import shutil

TSV_PATH = os.path.join(os.path.dirname(__file__), 'other.tsv')
CLIPS_DIR = os.path.join(os.path.dirname(__file__), 'data')
CLIPS_NEW_DIR = os.path.join(os.path.dirname(__file__), 'clips')

clip_files = set(f for f in os.listdir(CLIPS_DIR) if f.endswith('.mp3'))

speaker_counts = Counter()
file_to_speaker = {}
with open(TSV_PATH, 'r', encoding='utf-8') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    for row in reader:
        filename = row['path']
        if filename in clip_files:
            speaker = row['client_id']
            speaker_counts[speaker] += 1
            file_to_speaker[filename] = speaker

top_speakers = [speaker for speaker, _ in speaker_counts.most_common(12)]
print('Top speakers in clips:')
for idx, speaker in enumerate(top_speakers):
    print(f'{idx}: {speaker} ({speaker_counts[speaker]})')

os.makedirs(CLIPS_NEW_DIR, exist_ok=True)
for i in range(12):
    os.makedirs(os.path.join(CLIPS_NEW_DIR, str(i)), exist_ok=True)

speaker_to_idx = {speaker: idx for idx, speaker in enumerate(top_speakers)}
for filename, speaker in file_to_speaker.items():
    if speaker in speaker_to_idx:
        src = os.path.join(CLIPS_DIR, filename)
        dst = os.path.join(CLIPS_NEW_DIR, str(speaker_to_idx[speaker]), filename)
        shutil.copy2(src, dst)