import os
import re
from argparse import ArgumentParser
from pathlib import Path
from types import ModuleType
from typing import Optional

from pynavio import infer_imported_code_path

# names of imported submodules of the models module
MODELS = [
    name for name, obj in globals().items()
    if isinstance(obj, ModuleType) and obj.__name__.startswith('examples')
]


def _format_group(group: str) -> str:
    return group.replace('_', '') if group.startswith('_') else group


def main(path: str,
         name: str,
         data: Optional[str] = None,
         oodd: Optional[str] = None,
         explanations: Optional[str] = None) -> None:

    script_path = Path(os.path.abspath(__file__))
    code_path = infer_imported_code_path(path=script_path,
                                         root_path=script_path.parents[1])

    globals()[name].setup(with_data=bool(data),
                          with_oodd=bool(oodd),
                          explanations=explanations or 'disabled',
                          path=path,
                          code_path=code_path)


if __name__ == '__main__':
    parser = ArgumentParser('main')
    parser.add_argument('path', type=str, help='path of the model to build')
    args = parser.parse_args()

    name = args.path.split('/')[-1]
    pattern = f'^({"|".join(MODELS)})(_data)?(_oodd)?(_plotly|_default)?$'
    assert re.match(pattern, name), f'Model name {name} not recognized'

    groups = [
        _format_group(group) if group is not None else None
        for group in re.search(pattern, name).groups()
    ]
    main(args.path, *groups)
