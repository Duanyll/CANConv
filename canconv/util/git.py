import subprocess
import os
from datetime import datetime
import json

def git(args: str) -> str:
    cmd = "git " + args
    print("$ " + cmd)
    res = subprocess.run(cmd, capture_output=True, text=True, shell=True).stdout.strip()
    print(res)
    return res

def check_output_directory(exp_name):
    if not os.path.exists('runs'):
        os.mkdir('runs')
    out_dir = os.path.join('runs', exp_name)
    if os.path.exists(out_dir):
        option = input("Output directory exists. Overwrite? (y/n)")
        if option == 'y':
            return out_dir
        else:
            exit(0)
    else:
        os.mkdir(out_dir)
        return out_dir
    
def get_git_commit() -> str:
    return git("rev-parse HEAD")