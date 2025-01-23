# find doc at https://solcx.readthedocs.io/en/latest/

# external
import solcx

# internal
from chain import get_chains
# reentrency
from reentrency import getCallValueRelatedByteLocs
# time stamp dependency
from ts_dependency import getTSDependencyByteLocs
from ast_util import slice_sol
from clean import clean

# std
import os
import json
import shutil

SOLCX_INSTALL_DIR = os.environ["SOLCX_BINARY_PATH"]
print(f"Installing solc to {SOLCX_INSTALL_DIR}")

def get_solc_version(file):
    '''
    Get the version of solc used in the .sol file.
    Return None if not found,
    otherwise return the version in the format of "x.y.z".
    e.g. "0.4.11"
    '''
    with open(file, "r") as f:
        for line in f:
            if line.startswith("pragma solidity"):
                ver = line.split(" ")[2].split(";")[0]
                if ver[0] == "^":
                    ver = ver[1:]
                return ver
    return None

# solcx only supports solc version >= 0.4.11
def assert_solc_version(ver):
    '''
    Assert the version of solc used in the .sol file.
    Since solcx only supports solc version >= 0.4.11,
    assertion will fail if the version is not supported.
    '''
    if ver is None:
        return False
    x, y, z = ver.split(".")
    if not (int(x) >= 0 and int(y) >= 4 and int(z) >= 11):
        return False
    return True

# gather files & versions
print("Gathering files & versions...")
# tuples of (file, its root, pragma version) for later processing
files = set()
# all found versions
ver_pragmas = set()
for root, _, fs in os.walk("dataset"):
    for f in fs:
        if f.endswith(".sol"):
            toadd = f"{root}/{f}"
            clean(toadd)
            ver = get_solc_version(toadd)
            if assert_solc_version(ver):
                files.add((toadd, root, ver))
                ver_pragmas.add(ver)

files = set()
# files.add(("dataset/reentrancy/reentrancy_dao2.sol", "dataset/reentrancy", "0.4.19"))
# files.add(("dataset/reentrancy/reentrancy_dao3.sol", "dataset/reentrancy", "0.4.19"))
files.add(("dao2.sol/test.sol", "dao2.sol", "0.4.19"))
print("Files found:")
print(files)
print("Versions found:")
print(ver_pragmas)

# installing all versions of solc
print("Installing versions of solc...")
if not os.path.exists(SOLCX_INSTALL_DIR):
    os.makedirs(SOLCX_INSTALL_DIR)
for ver in ver_pragmas:
    # installed
    if os.path.exists(f"{SOLCX_INSTALL_DIR}/solc-v{ver}"):
        continue

    solcx.install_solc(version=ver, show_progress=False)
print("Installation complete.")

# clearing .sol files

# generate antlr trees of .sol files
print("Generating ANTLR trees...")
for file, kind, ver in files:
    kind = kind.split("/")[-1]
    filename = file.split("/")[-1]
    target_dir = f"./out/{kind}/{filename}"
    if os.path.exists(f"{target_dir}/antlr.txt"):
        print(f"ANTLR of {file} already exists, skipping...")
        continue
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    cmd = f"java -cp \"java/antlr.jar:java/antlrsrc.jar\" Tokenize {file} > {target_dir}/antlr.txt"
    os.system(cmd)
print("ANTLR generation complete.")

# generate json of ast of .sol files
print("Generating ASTs...")
print("Target directory: out/")
for file, kind, ver in files:
    kind = kind.split("/")[-1]
    filename = file.split("/")[-1]
    target_dir = f"./out/{kind}/{filename}"
    if os.path.exists(f"{target_dir}/ast.json"):
        print(f"Ast of {file} already exists, skipping...")
        continue
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    cmd = "./.solcx/solc-v" + ver
    os.system(f"{cmd} --ast-json {file} > {target_dir}/tmp.json")
    # filter the "===" and "JSON AST" and empty lines
    with open(f"{target_dir}/ast.json", "w") as to:
        with open(f"{target_dir}/tmp.json", "r") as orig:
            for line in orig:
                if line.startswith("\n") or line.startswith("===") or line.startswith("JSON AST"):
                    continue
                to.write(line)
    os.remove(f"{target_dir}/tmp.json")
print("AST generation complete.")

# generate dot files of .sol files
print("Generating .dot files...")
for file, path, ver in files:
    kind = path.split("/")[-1]
    filename = file.split("/")[-1]
    target_dir = f"./out/{kind}/{filename}"
    # if os.path.exists(f"{target_dir}/{filename}.dot"):
    #     print(f"Dot of {file} already exists, skipping...")
    #     continue
    # if not os.path.exists(target_dir):
    #     os.makedirs(target_dir)
    solc = "./.solcx/solc-v" + ver
    cmd = f"slither --solc {solc} {file} --print call-graph"
    os.system(f"{cmd} > /dev/null 2>&1")

    # move all .dot files to out/dot/
    path_files = os.listdir(path)
    for f in path_files:
        if f.endswith(".dot"):
            if os.path.exists(f"{target_dir}/{f}"):
                print(f"Dot of {file} already exists, skipping...")
                os.remove(f"{path}/{f}")
                continue
            shutil.move(f"{path}/{f}", f"{target_dir}/{f}")

EDGE_INDICATOR = " -> "
LABEL_INDICATOR = "[label="
# try and slice the code
for file, path, ver in files:
    kind = path.split("/")[-1]
    filename = file.split("/")[-1]
    target_dir = f"./out/{kind}/{filename}"
    cur_dir_fs = os.listdir(target_dir)

    if not os.path.exists(f"{target_dir}/ast.json"):
        print(f"AST of {file} does not exist!!! skipping...")
        continue
    print("="*30)
    print("processing", file)
    ast_json = json.load(open(f"{target_dir}/ast.json"))
    dotfiles = [f for f in cur_dir_fs if f.endswith(".dot")]

    # generate call graphs from .dot files
    # --------------------------------------------------
    # p.s. I `rg`-ed on dataset, only two kinds of line indicates an edge:
    # 1. ".*" -> ".*"
    # 2. }".*" -> ".*"
    edges = set()
    singles = set()
    for df in dotfiles:
        with open(f"{target_dir}/{df}", "r") as f:
            for line in f:
                if EDGE_INDICATOR in line:
                    comp1 = line.split(EDGE_INDICATOR)[0]
                    comp2 = line.split(EDGE_INDICATOR)[1]
                    if comp1.startswith("}"):
                        comp1 = comp1[1:]
                    if comp2.endswith("\n"):
                        comp2 = comp2[:-1]
                    comp1 = comp1[1:-1]
                    comp2 = comp2[1:-1]
                    edges.add((comp1, comp2))
                if LABEL_INDICATOR in line:
                    funcname = line.split(LABEL_INDICATOR)[0]
                    singles.add(funcname.split(' ')[0][1:-1])

    print("edges", edges)
    # get all function call chains
    # e.g. f1 -> f2 -> f3, g1 -> g2, h1, ...
    # e.g. f and g(edges)
    chains = get_chains(edges)
    # e.g. h(single node)
    for single in singles:
        chains.append([single])
    # exit()

    byte_locs = getCallValueRelatedByteLocs(ast_json, chains, dotfiles, target_dir)
    # byte_locs = getTSDependencyByteLocs(ast_json, chains, dotfiles, target_dir)
    # exit(0)
    print(byte_locs)
    sliced_lines = slice_sol(f"{path}/{filename}", byte_locs)
    # print(sliced_lines)
    for l in sliced_lines:
        print(l)

    if len(sliced_lines) > 0:
        with open(f"{target_dir}/sliced.txt", "w") as f:
            for line in sliced_lines:
                f.write(line.strip() + "\n")
