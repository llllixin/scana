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

def gen_jobs():
    '''
    Walk through the dataset directory and gather all .sol files and their pragma versions.
    Required solc versions are installed.
    Return a set of tuples of (file, its root, pragma version) for later processing.
    '''
    # gather files & versions
    print("Gathering files & versions...")
    # tuples of (file, its root, pragma version) for later processing
    jobs = set()
    # all found versions
    ver_pragmas = set()
    for root, _, fs in os.walk("dataset"):
        for f in fs:
            if f.endswith(".sol"):
                toadd = os.path.join(root, f)
                clean(toadd)
                ver = get_solc_version(toadd)
                if assert_solc_version(ver):
                    jobs.add((toadd, root, ver))
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
    print(jobs)
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

    return jobs

# helper
def gen_antlr(file, target_dir):
    cmd = f"java -cp \"java/antlr.jar:java/antlrsrc.jar\" Tokenize {file} > {target_dir}/antlr.txt"
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f"Error generating antlr tree for {file}: {res.stderr}")
        return False, res.stderr
def gen_antlr_for_job(job):
    '''
    :param job: a tuple of (file, its root, pragma version)
    '''
    file, root, _ = job
    kind = root.split("/")[-2] + "/" + root.split("/")[-1]
    filename = file.split("/")[-1]
    target_dir = os.path.join("out", kind, filename)
    antlr_path = os.path.join(target_dir, "antlr.txt")
    if os.path.exists(antlr_path):
        print(f"ANTLR of {file} already exists, skipping...")
        return
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"Generating ANTLR tree for {file}...")
    gen_antlr(file, target_dir)
    return True, None

# helper
def gen_ast(file, ver, target_dir):
    cmd = f"./.solcx/solc-v{ver} --ast-json {file} > {target_dir}/tmp.json"
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f"Error generating AST for {file}: {res.stderr}")
        return False, res.stderr
    with open(f"{target_dir}/ast.json", "w") as to:
        with open(f"{target_dir}/tmp.json", "r") as orig:
            for line in orig:
                if line.startswith("\n") or line.startswith("===") or line.startswith("JSON AST"):
                    continue
                to.write(line)
    os.remove(f"{target_dir}/tmp.json")
def gen_ast_for_job(job):
    file, root, ver = job
    kind = root.split("/")[-2] + "/" + root.split("/")[-1]
    filename = file.split("/")[-1]
    target_dir = os.path.join("out", kind, filename)
        
    ast_path = os.path.join(target_dir, "ast.json")
    if os.path.exists(ast_path):
        print(f"Ast of {file} already exists, skipping...")
        return
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    gen_ast(file, ver, target_dir)
    return True, None

# helper
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
        return False, res.stderr
    # move all .dot files to out/dot/
    import time
    time.sleep(0.1)
    path_files = os.listdir(path)
    filename = file.split("/")[-1]
    for f in path_files:
        if f.endswith(".dot") and f.startswith(filename):
            shutil.move(f"{path}/{f}", f"{target_dir}/{f}")
        else:
            # print(f"Skipping {f} {filename}")
            pass
def gen_dot_for_job(job):

    file, root, ver = job
    kind = root.split("/")[-2] + "/" + root.split("/")[-1]
    filename = file.split("/")[-1]
    target_dir = os.path.join("out", kind, filename)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # if filepath:
    #     gen_dot(filepath, '.', ver, target_dir)
    gen_dot(file, root, ver, target_dir)
    return True, None

EDGE_INDICATOR = " -> "
LABEL_INDICATOR = "[label="
# helper
def slice_job(job, slice_kind=None):
    # try and slice the code
    file, path, _ = job
    filename = file.split("/")[-1]
    kind = path.split("/")[-2] + "/" + path.split("/")[-1]
    if slice_kind is None:
        target_dir = os.path.join("out", kind, filename)
    else: 
        target_dir = os.path.join("single_source")
    slice_kind = kind.split("/")[0] if slice_kind is None else slice_kind
    cur_dir_fs = os.listdir(target_dir)

    print(target_dir)
    ast_path = os.path.join(target_dir, "ast.json")
    if not os.path.exists(ast_path):
        print(f"AST of {file} does not exist!!! skipping...")
        return False
    print("="*30)
    print("processing", file)
    ast_json = json.load(open(ast_path, "r"))
    dotfiles = [f for f in cur_dir_fs if f.endswith(".dot")]
    print("dotfiles", dotfiles)

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
        return True, None
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return False, str(e)

def worker(job):
    name, _, _ = job
    print("Generating ANTLR trees...")
    res1 = gen_antlr_for_job(job)
    print("ANTLR generation complete.")

    print("Generating ASTs...")
    res2 = gen_ast_for_job(job)
    print("AST generation complete.")

    print("Generating .dot files...")
    res3 = gen_dot_for_job(job)
    print("Dot generation complete.")

    print("Slicing...")
    res4 = slice_job(job)
    print("Slicing complete.")

    return name, res1, res2, res3, res4

def process():
    '''
    :param filepath: the path of the single .sol file to process, 
    only used to slice single file
    :param slice_kind: the kind of slicing to perform, 
    'ree' for reentrancy slicing, 'ts' for timestamp dependency slicing, 
    only used to slice single file
    '''
    jobs = gen_jobs()
    antlr_errfiles = set()
    ast_errfiles = set()
    dot_errfiles = set()
    slice_errfiles = set()

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=12) as pool:
        fts = [pool.submit(worker, job) for job in jobs]
        for ft in fts:
            name, res1, res2, res3, res4 = ft.result()
            if not res1:
                antlr_errfiles.add(name)
            if not res2:
                ast_errfiles.add(name)
            if not res3:
                dot_errfiles.add(name)
            if not res4:
                slice_errfiles.add(name)

    if len(antlr_errfiles) > 0:
        with open("antlr_err_files.txt", "a+") as f:
            for file, err in antlr_errfiles:
                f.write(file + "\n")
                f.write(err + "\n")
            f.write("-- end --")

    if len(ast_errfiles) > 0:
        with open("ast_err_files.txt", "a+") as f:
            for file, err in ast_errfiles:
                f.write(file + "\n")
                f.write(err + "\n")
            f.write("-- end --")

    if len(dot_errfiles) > 0:
        with open("dot_err_files.txt", "a+") as f:
            for file, err in dot_errfiles:
                f.write(file + "\n")
                f.write(err + "\n")
            f.write("-- end --")

    if len(slice_errfiles) > 0:
        with open("slice_err_files.txt", "a+") as f:
            for file, err in slice_errfiles:
                f.write(file + "\n")
                f.write(err + "\n")
            f.write("-- end --")

def process_single(filepath, slice_kind):
    filepath = os.path.abspath(filepath)
    job = (filepath, os.path.dirname(filepath), get_solc_version(filepath))
    file, _, _ = job
    target_dir = os.path.join(os.getcwd(), "single_source")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # job = [("dao2.sol/test.sol", "dao2.sol", "0.4.19")]
    print("Generating ANTLR trees...")
    gen_antlr(file, target_dir)
    print("ANTLR generation complete.")
    
    print("Generating ASTs...")
    gen_ast(file, get_solc_version(file), target_dir)
    print("AST generation complete.")

    print("Generating .dot files...")
    gen_dot(file, 
            os.getcwd(),
            get_solc_version(file),
            target_dir)
    print("Dot generation complete.")

    slice_job(job, slice_kind)

if __name__ == "__main__":
    process()
