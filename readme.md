# Setup
```
nix develop
```

# Getting Dataset:
```
git clone git@github.com:smartbugs/smartbugs-curated.git
# or
git clone https://github.com/smartbugs/smartbugs-curated.git

mv smartbugs-curated/dataset .
rm -rf smartbugs-curated
```

# Code Slicing:
```
python cmd/process.py
```

# Out Structure:
```
out/{kind}/{filename}/{content}
```
Where content could be `ast.json`, `*.dot`, `sliced.txt`.
If `sliced.txt` is not present, no vulnerabilities were found in that code.

# Abstract Syntax Tree:
```
cd antlr
antlr4 -Dlanguage=Python3 Solidity.g4
python driver.py [path to solidity file]
# note stderr and stdout are printed to console.
# consider manually directing them.
```

##### You can run clean.sh to get rid of:
(Read what it does before running it)
```
__pycache__
.DS_Store
```
