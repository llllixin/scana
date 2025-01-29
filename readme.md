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
python cmd/process.py
```

# Out Structure:
```bash
out/{kind}/{good/bad}/{filename}/{content}
```
Where content could be `ast.json`, `*.dot`, `sliced.txt`, `antlr.txt`
If `sliced.txt` is not present, no vulnerabilities were found in that code.

# Generating Vocabulary From ./out:
```bash
python model/w2v/build_vocab.py
```

# Training Word2Vec:
```bash
python model/w2v/model.py
# or
python model/w2v/model.py <path-to-checkpoint> <epoch>
# or
python model/w2v/model.py <epoch>
```

##### You can run clean.sh to get rid of:
(Read what it does before running it)
```
__pycache__
.DS_Store
```
