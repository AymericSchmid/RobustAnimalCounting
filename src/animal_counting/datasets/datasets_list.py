from pathlib import Path

DATASETS = {
    "eikelboom": {
        "url": "https://data.4tu.nl/file/9b1a2fcb-930e-4cc5-b9f0-a381dd1c7206/f978a7a0-f2aa-4c2b-a663-1165b247b56a", 
        "type": "file",
        "filename": "eikelboom.zip",
    },
    "qian_penguins": {
        "url": "https://doi.org/10.5061/dryad.8931zcrv8",
        "type": "manual",
    },
    "waid": {
        "url": "https://github.com/xiaohuicui/WAID.git",
        "type": "git",
    },
    "aed": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/davidrpugh/aerial-elephant-dataset",
        "type": "file",
        "filename": "aed.zip",
    },
    "delplanque": {
        "url": "https://dataverse.uliege.be/api/access/datafile/11098",
        "type": "file",
        "filename": "delplanque.zip",
    },
}


DATA_DIR = Path("data/raw")