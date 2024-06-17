file_path = '/workspace/Chimera/bert_data/wikipedia.segmented.nltk.txt.1'
with open(file_path, 'r') as f:
    lines = f.readlines()
print(f'Total lines in file: {len(lines)}')