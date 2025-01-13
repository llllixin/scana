import torch
import torch.nn as nn
import torch.nn.functional as F

# word2vec model, skip-gram
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGram, self).__init__()

        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, target, pos_context, neg_context):
        v = self.in_embed(target) # batch_size, embed_size
        pos_u = self.out_embed(pos_context) # batch_size, 2*window_size, embed_size
        neg_u = self.out_embed(neg_context) # batch_size, neg_size, embed_size

        pos_score = torch.sum(torch.bmm(pos_u, v.unsqueeze(2)), dim=2) # batch_size, 2*window_size
        neg_score = torch.sum(torch.bmm(neg_u, v.unsqueeze(2)), dim=2) # batch_size, neg_size

        return pos_score, neg_score

    def loss(self, pos_score, neg_score, mask):
        pos_loss = -self.log_sigmoid(pos_score)
        pos_loss = pos_loss * mask
        pos_loss = torch.sum(pos_loss, dim=1) / torch.sum(mask, dim=1)
        neg_loss = -self.log_sigmoid(-neg_score)

        return pos_loss.mean() + neg_loss.mean()

    def train_sg(self, epoch, target, pos_context, neg_context, mask, lr=0.01):
        '''
        Args:
            target is a tensor of shape (b, d), 
            pos_context is a matrix of shape (b, n, d) and each row represents a positive context word,
            neg_context is a matrix of shape (b, m, d) and each row represents a negative context word.
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr)

        for e in range(epoch):
            optimizer.zero_grad()
            pos_score, neg_score = self.forward(target, pos_context, neg_context)
            loss = self.loss(pos_score, neg_score, mask)
            loss.backward()
            optimizer.step()

            if e % 100 == 0:
                print(f"Epoch {e}, loss: {loss.item()}")

# testing code, will remove in the future
if __name__ == '__main__':
    sentences = [
            "the quick brown fox jumps over the lazy dog",
            "i love machine learning",
            "this is not a good idea",
            "where is everybody",
            ]
    print("sentences:")
    print(sentences)
    vocab = set()
    for s in sentences:
        for word in s.split():
            vocab.add(word)
    print("constructed vocab:")
    print(vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    word2idx = {word: idx+1 for idx, word in enumerate(vocab)}
    word2idx["<pad>"] = 0
    vocab.add("<pad>")
    idx2word = {idx: word for word, idx in word2idx.items()}
    print("word2idx:")
    print(word2idx)
    print("idx2word:")
    print(idx2word)

    batch_size = 0
    for s in sentences:
        batch_size += len(s.split())
    window_size = 2
    neg_size = 4
    targets = []
    pos_context = []
    neg_context = []
    mask = []
    for s in sentences:
        words = s.split()
        for i, word in enumerate(words):
            target = word2idx[word]
            targets.append(target)
            pos = [word2idx[words[j]] for j in range(max(i-window_size, 0), min(i+window_size+1, len(words))) if j != i]
            pad_len = 2*window_size - len(pos)
            if pad_len > 0:
                pos += [0] * pad_len
                mask.append([1] * (2*window_size - pad_len) + [0] * pad_len)
            else:
                mask.append([1] * 2*window_size)

            pos_context.append(pos)
            neg = torch.randint(0, len(vocab), (neg_size,)).tolist()
            neg_context.append(neg)

    print("targets:", targets)
    print("num_classes:", len(vocab))
    targets = torch.tensor(targets, dtype=torch.long).to(device)
    pos_context = torch.tensor(pos_context, dtype=torch.long).to(device)
    neg_context = torch.tensor(neg_context, dtype=torch.long).to(device)
    mask = torch.tensor(mask, dtype=torch.float).to(device)

    print("targets:", targets.shape)
    print(targets)
    print("pos_context:", pos_context.shape)
    print(pos_context)
    print("neg_context:", neg_context.shape)
    print(neg_context)

    idx = word2idx["good"]
    idy = word2idx["idea"]
    idx = torch.tensor([idx]).to(device)
    idy = torch.tensor([idy]).to(device)

    sg = SkipGram(len(vocab), 10).to(device)
    bef_idx = sg.in_embed(idx)
    bef_idy = sg.out_embed(idy)
    bef_cos = F.cosine_similarity(bef_idx, bef_idy)
    print("before training:", bef_cos)

    sg.train_sg(1000, targets, pos_context, neg_context, mask)

    sg.eval()
    aft_idx = sg.in_embed(idx)
    aft_idy = sg.out_embed(idy)
    aft_cos = F.cosine_similarity(aft_idx, aft_idy)
    print("after training:", aft_cos)
