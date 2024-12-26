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
