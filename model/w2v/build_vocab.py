import os

from . import w2v_root

operators1 = {
    '(', ')', '[', ']', '.',
    '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':', ';',
    '{', '}'
}
operators2 = {
    '->', '++', '--',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators3 = {'<<=', '>>='}

def tokenize(line):
    '''
    Tokenize a line of code
    Returns:
        A list of tokens
    '''
    tmp, w = [], []
    i = 0
    while i < len(line):
        # Ignore spaces and combine previously collected chars to form words
        if line[i] == ' ':
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        # Check operators and append to final list
        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 3])
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 2])
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        # Character appended to word list
        else:
            w.append(line[i])
            i += 1
    # Filter out irrelevant strings
    res = list(filter(lambda c: c != '', tmp))
    return list(filter(lambda c: c != ' ', res))

def gen_vocab():
    '''
    1. Loads the raw vocab from the vocab file
    1. Modifies the vocab to include special tokens,
    like <pad>
    2. generate the word table from the given vocab

    Returns:
        vocab: the modified vocab
        word2idx: the word table
    '''
    special_tokens = ['<pad>', '<unk>']
    vocab = build_raw_vocab_from_file()
    vocab.add('\n')
    vocab.add(' ')
    word2idx = {word: idx+len(special_tokens) for idx, word in enumerate(vocab)}
    for i, st in enumerate(special_tokens):
        word2idx[st] = i
        vocab.add(st)
    return vocab, word2idx

def to_ids(code):
    _, word2idx = gen_vocab()
    
    tokens = tokenize(code)

    # TODO: MODEL SUCKS BECAUSE OF THIS
    idx = []
    for token in tokens:
        if token in word2idx:
            idx.append(word2idx[token])
        else:
            idx.append(word2idx['<unk>'])

    return idx

def build_vocab_from_code():
    out_path = 'out'
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Directory {out_path} not found,",
                                "preprocess the code files first")
    dirs = os.listdir(out_path)

    vocab = set()

    for dir in dirs:
        kind_dir = os.path.join(out_path, dir)
        cla = os.listdir(kind_dir)
        for c in cla:
            cla_dir = os.path.join(kind_dir, c)
            dirs = os.listdir(cla_dir)
            for d in dirs:
                single_entry_path = os.path.join(cla_dir, d)
                files = os.listdir(single_entry_path)
                for file in files:
                    if file != 'antlr.txt' and file != 'sliced.txt':
                        print(file)
                        continue;

                    file_path = os.path.join(single_entry_path, file)
                    with open(file_path, 'r') as f:
                        code = f.read().replace('\n', ' ')
                        tokens = set(tokenize(code))
                        for token in tokens:
                            vocab.add(token)
    return vocab

def build_raw_vocab_from_file():
    vocab_path = os.path.join(w2v_root, 'vocab.txt')
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found at {vocab_path},",
                                "run `python -m model.w2v.build_vocab`",
                                "to generate vocab file first")
    with open(vocab_path, 'r') as f:
        vocab = f.read().split('\n')
    vocab = set(vocab)
    return vocab

if __name__ == '__main__':
    vocab = build_vocab_from_code()
    vocab_path = os.path.join(w2v_root, 'vocab.txt')
    with open(vocab_path, 'w') as f:
        for word in vocab:
            f.write(f'{word}\n')
