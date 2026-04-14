from pathlib import Path

DATASETS = {
    "eikelboom": {
        "url": "https://data.4tu.nl/file/9b1a2fcb-930e-4cc5-b9f0-a381dd1c7206/f978a7a0-f2aa-4c2b-a663-1165b247b56a", 
        "type": "file",
        "filename": "eikelboom.zip",
    },
    "qian_penguins": {
        "url": "blabla",  
        "type": "file",
        "filename": "penguins.zip",
    },
    "waid": {
        "url": "blabla",
        "type": "git",
    },
    "aed": {
        "url": "blabla",
        "type": "file",
        "filename": "aed.zip",
    },
    "delplanque": {
        "url": None,  # manual
        "type": "manual",
    },
}


DATA_DIR = Path("data/raw")