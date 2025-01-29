import os

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

def build_vocab_from_code():
    dirs = os.listdir('./out')

    vocab = set()
    vocab.add('\n')
    vocab.add(' ')

    for dir in dirs:
        cla = os.listdir(f'./out/{dir}')
        for c in cla:
            dirs = os.listdir(f'./out/{dir}/{c}')
            for d in dirs:
                files = os.listdir(f'./out/{dir}/{c}/{d}')
                for file in files:
                    if file != 'antlr.txt' and file != 'sliced.txt':
                        print(file)
                        continue;

                    with open(f'./out/{dir}/{c}/{d}/{file}', 'r') as f:
                        code = f.read().replace('\n', ' ')
                        tokens = set(tokenize(code))
                        for token in tokens:
                            vocab.add(token)
    vocab.add('\n')
    vocab.add(' ')
    return vocab

def build_vocab_from_file():
    with open('model/w2v/vocab.txt', 'r') as f:
        vocab = f.read().split('\n')
    vocab = set(vocab)
    vocab.add('\n')
    vocab.add(' ')
    return vocab

if __name__ == '__main__':
    vocab = build_vocab_from_code()
    with open('model/w2v/vocab.txt', 'w') as f:
        for word in vocab:
            f.write(f'{word}\n')
