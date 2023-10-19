from jsonargparse import set_docstring_parse_options

# centrally configure docstring parsing, which is necessary for use cases
# where the high-level API's dataclasses are used in CLIs (as in our examples)
set_docstring_parse_options(attribute_docstrings=True)
