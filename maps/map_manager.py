import random
from pathlib import Path
import utils

def get_maps(root: Path = Path("./")) -> list[dict[str, str]]:
    maps: list[dict[str, str]] = []
    for sub_file in root.iterdir():
        if sub_file.is_dir():
            category = sub_file.name
            for _m in sub_file.iterdir():
                maps.append({category:str(_m.absolute().as_posix())})
    return maps