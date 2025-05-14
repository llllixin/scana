# Setup
```bash
nix develop
```

# Getting Dataset:
```bash
git clone git@github.com:smartbugs/smartbugs-curated.git
# or
git clone https://github.com/smartbugs/smartbugs-curated.git

mv smartbugs-curated/dataset .
rm -rf smartbugs-curated
```

# Setting up ANTLR(Requires Internet):
```bash
./java/build.sh
```

# Code Slicing:
```bash
python -m pycmd.process
```

# Out Structure:
```bash
out/{kind}/{vul/non_vul}/{filename}/{content}
```
Where content could be `ast.json`, `*.dot`, `sliced.txt`, `antlr.txt`
If `sliced.txt` is not present, no vulnerabilities were found in that code.

# Generating Vocabulary From ./out:
```bash
python -m model.w2v.build_vocab
```

# Training Word2Vec:
```bash
# List present w2v models:
python -m model.w2v.train list
# Train a new w2v model:
python -m model.w2v.train train
# Use --help to see flags & options
```

# Training BLSTM:
```bash
python -m model.train train
# Use --help to see flags & options
```

##### You can run clean.sh to get rid of:
(Read what it does before running it)
```
__pycache__
.DS_Store
```
