# Taken from
# https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/spellchecker/README.html
# and minimally adjusted

import base64
import os.path
from urllib.request import urlopen

from jupyter_core.paths import jupyter_data_dir
from notebook.services.config import ConfigManager

remote_base_url = (
    "https://chromium.googlesource.com/" + "chromium/deps/hunspell_dictionaries/+/refs/heads/main"
)
local_base_url = os.path.join(
    jupyter_data_dir(), "nbextensions", "spellchecker", "typo", "dictionaries"
)

lang_code = "en_US"

if not os.path.exists(local_base_url):
    print("creating directory {!r}".format(local_base_url))
    os.makedirs(os.path.realpath(local_base_url))

cm = ConfigManager()
for ext in ("dic", "aff"):
    dict_fname = lang_code + "." + ext
    remote_path = f"{remote_base_url}/{dict_fname}?format=TEXT"
    local_path = os.path.join(local_base_url, dict_fname)
    print("saving {!r}\n    to {!r}".format(remote_path, local_path))
    with open(local_path, "wb") as loc_file:
        base64.decode(urlopen(remote_path), loc_file)
    rel_path = "./typo/dictionaries/" + dict_fname
    cm.update("notebook", {"spellchecker": {ext + "_url": rel_path}})

cm.update("notebook", {"spellchecker": {"lang_code": lang_code}})
