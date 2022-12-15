import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("message", help="the commit message")
args = parser.parse_args()

subprocess.run(["git", "add", "--all"])
subprocess.run(["git", "commit", "-m", args.message])
subprocess.run(["git", "push"])
