import os
import re

if __name__ == "__main__":
    with open("pyproject.toml") as f:
        toml = f.read()
    groups = re.findall(r"\[tool\.poetry\.group\.([^\.]+)\]", toml)
    group_options = " ".join(f"--with {group}" for group in groups)
    cmd = f"poetry install {group_options}"
    print(cmd)
    os.system(cmd)
