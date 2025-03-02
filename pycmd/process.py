# find doc at https://solcx.readthedocs.io/en/latest/

# external
import subprocess
import solcx

# internal
from .chain import get_chains
# reentrency
from .reentrency import getCallValueRelatedByteLocs
# time stamp dependency
from .ts_dependency import getConditionRelatedSC
from .ast_util import slice_sol
from .clean import clean

# std
import os
import re
import json
import shutil
import threading

SOLCX_INSTALL_DIR = os.environ["SOLCX_BINARY_PATH"]
print(f"Installing solc to {SOLCX_INSTALL_DIR}")

def get_solc_version(file):
    '''
    Get the version of solc used in the .sol file.
    Return None if not found,
    otherwise return the version in the format of "x.y.z".
    e.g. "0.4.11"
    '''
    pattern = re.compile(r"\s*pragma solidity\s+(?:\^|>=|=|>)?\s*(\d+\s*\.\s*\d+\s*\.\s*\d+)\s*[^\n]*;")
    with open(file, "r") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                return match.group(1).replace(" ", "")
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
    try:
        x, y, z = ver.split(".")
        if (int(x) >= 0 and int(y) > 4):
            return True
        if (int(x) >= 0 and int(y) == 4 and int(z) >= 11):
            return True
    except:
        print(f"Invalid version: {ver}")
        return False
    return False

def process(filepath=None, slice_kind=None):
    '''
    :param filepath: the path of the single .sol file to process, 
    only used to slice single file
    :param slice_kind: the kind of slicing to perform, 
    'ree' for reentrancy slicing, 'ts' for timestamp dependency slicing, 
    only used to slice single file
    '''
    # gather files & versions
    print("Gathering files & versions...")
    # tuples of (file, its root, pragma version) for later processing
    files = set()
    # all found versions
    ver_pragmas = set()
    if filepath:
        if not filepath.endswith(".sol"):
            raise ValueError("Invalid file path, not a .sol file.")
        clean(filepath)
        ver = get_solc_version(filepath)
        if assert_solc_version(ver):
            files.add((filepath, os.path.dirname(filepath), ver))
            ver_pragmas.add(ver)
        else:
            print(f"Version not supported for file {filepath}: {ver}")
            return
    else:
        for root, _, fs in os.walk("dataset"):
            for f in fs:
                if f.endswith(".sol"):
                    toadd = os.path.join(root, f)
                    clean(toadd)
                    ver = get_solc_version(toadd)
                    if assert_solc_version(ver):
                        files.add((toadd, root, ver))
                        ver_pragmas.add(ver)
                    else:
                        print(f"Version not supported for file {toadd}: {ver}")

    # files = set()
    # files.add(("dataset/ts/vul/14953.sol", "dataset/ts/vul", "0.4.16"))
    # files.add(("dataset/reentrancy/reentrancy_dao2.sol", "dataset/reentrancy", "0.4.19"))
    # files.add(("dataset/reentrancy/reentrancy_dao3.sol", "dataset/reentrancy", "0.4.19"))
    # files.add(("dao2.sol/test.sol", "dao2.sol", "0.4.19"))
    # files.add(("dataset/ree/vul/33851.sol", "dataset/ree/vul", "0.4.11"))
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
    faulty_files = set()
    def gen_antlr(file, target_dir):
        cmd = f"java -cp \"java/antlr.jar:java/antlrsrc.jar\" Tokenize {file} > {target_dir}/antlr.txt"
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print(f"Error generating antlr tree for {file}: {res.stderr}")
            faulty_files.add(file)

    for file, kind, ver in files:
        if filepath:
            target_dir = os.path.join("eval")
        else:
            kind = kind.split("/")[-2] + "/" + kind.split("/")[-1]
            filename = file.split("/")[-1]
            target_dir = os.path.join("out", kind, filename)
        antlr_path = os.path.join(target_dir, "antlr.txt")
        if os.path.exists(antlr_path):
            print(f"ANTLR of {file} already exists, skipping...")
            continue
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        print(f"Generating ANTLR tree for {file}...")
        gen_antlr(file, target_dir)

    print("ANTLR generation complete.")
    with open("out3.txt", "a+") as f:
        for file in faulty_files:
            f.write(file + "\n")
        f.write("done")
    faulty_files = set()

    # generate json of ast of .sol files
    print("Generating ASTs...")
    print("Target directory: out/")
    def gen_ast(file, ver, target_dir):
        cmd = f"./.solcx/solc-v{ver} --ast-json {file} > {target_dir}/tmp.json"
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print(f"Error generating AST for {file}: {res.stderr}")
            faulty_files.add(file)
            return
        with open(f"{target_dir}/ast.json", "w") as to:
            with open(f"{target_dir}/tmp.json", "r") as orig:
                for line in orig:
                    if line.startswith("\n") or line.startswith("===") or line.startswith("JSON AST"):
                        continue
                    to.write(line)
        os.remove(f"{target_dir}/tmp.json")
    for file, kind, ver in files:
        if filepath:
            target_dir = os.path.join("eval")
        else:
            kind = kind.split("/")[-2] + "/" + kind.split("/")[-1]
            filename = file.split("/")[-1]
            target_dir = os.path.join("out", kind, filename)
            
        ast_path = os.path.join(target_dir, "ast.json")
        if os.path.exists(ast_path):
            print(f"Ast of {file} already exists, skipping...")
            continue
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        gen_ast(file, ver, target_dir)
        # t = threading.Thread(target=gen_ast, args=(file, ver, target_dir))
        # threads.append(t)
        # t.start()
    with open("out3.txt", "a+") as f:
        for file in faulty_files:
            f.write(file + "\n")
        f.write("done")
    faulty_files = set()
    print("AST generation complete.")

    # generate dot files of .sol files
    print("Generating .dot files...")
    def gen_dot(file, path, ver, target_dir):
        target_files = os.listdir(target_dir)
        for f in target_files:
            if f.endswith(".dot"):
                print(f"Dot of {file} already exists, skipping...")
                return
        solc = "./.solcx/solc-v" + ver
        cmd = f"slither --solc {solc} {file} --print call-graph"
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print(f"Error generating dot for {file}: {res.stderr}")
            faulty_files.add(file)
            return
        # move all .dot files to out/dot/
        path_files = os.listdir(path)
        for f in path_files:
            if f.endswith(".dot"):
                shutil.move(f"{path}/{f}", f"{target_dir}/{f}")

    for file, path, ver in files:
        if filepath:
            target_dir = os.path.join("eval")
        else:
            kind = path.split("/")[-2] + "/" + path.split("/")[-1]
            filename = file.split("/")[-1]
            target_dir = os.path.join("out", kind, filename)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if filepath:
            gen_dot(filepath, '.', ver, target_dir)
        else:
            gen_dot(file, path, ver, target_dir)
    print("Dot generation complete.")
    with open("out3.txt", "a+") as f:
        for file in faulty_files:
            f.write(file + "\n")
        f.write("done")
    faulty_files = set()


    print("Faulty files:")
    print(faulty_files)

    EDGE_INDICATOR = " -> "
    LABEL_INDICATOR = "[label="
    # try and slice the code
    for file, path, ver in files:
        filename = file.split("/")[-1]
        if filepath:
            target_dir = os.path.join("eval")
        else:
            kind = path.split("/")[-2] + "/" + path.split("/")[-1]
            slice_kind = kind.split("/")[0]
            target_dir = os.path.join("out", kind, filename)
        cur_dir_fs = os.listdir(target_dir)

        print(target_dir)
        ast_path = os.path.join(target_dir, "ast.json")
        if not os.path.exists(ast_path):
            print(f"AST of {file} does not exist!!! skipping...")
            continue
        print("="*30)
        print("processing", file)
        ast_json = json.load(open(ast_path, "r"))
        dotfiles = [f for f in cur_dir_fs if f.endswith(".dot")]
        print("dotfiles", dotfiles)
        # print("ast_json", ast_json)

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

        # get all function call chains
        # e.g. f1 -> f2 -> f3, g1 -> g2, h1, ...
        # e.g. f and g(edges)
        chains = get_chains(edges)
        # e.g. h(single node)
        for single in singles:
            chains.append([single])
        # exit()

        try:
            if slice_kind == 'ree':
                byte_locs = getCallValueRelatedByteLocs(ast_json, chains, dotfiles, target_dir)
            elif slice_kind == 'ts':
                byte_locs = getConditionRelatedSC(ast_json, chains, dotfiles, target_dir)
            else:
                raise ValueError(f"Invalid slice kind: {slice_kind}")

            # byte_locs = getTSDependencyByteLocs(ast_json, chains, dotfiles, target_dir)
            # exit(0)
            # print(byte_locs)
            solpath = os.path.join(path, filename)
            if filepath:
                solpath = filepath
            sliced_lines = slice_sol(solpath, byte_locs)
            # print(sliced_lines)
            # for l in sliced_lines:
            #     print(l)
            if len(sliced_lines) > 0:
                with open(f"{target_dir}/sliced.txt", "w") as f:
                    for line in sliced_lines:
                        f.write(line.strip() + "\n")
            else:
                print("No sliced code, good")
        except Exception as e:
            print(f"Error processing {file}: {e}")
            with open("out5.txt", "a+") as f:
                f.write(file + "\n")
                f.write(str(e) + "\n")

        # except Exception as e:
        #     print(f"Error processing {file}: {e}")
        #     with open("out5.txt", "a+") as f:
        #         f.write(file + "\n")
        #     continue

if __name__ == "__main__":
    process()
