"""
Utils for reading database config and list of assays to be queried from ChEMBL
"""
import json
import os
from configparser import ConfigParser
from typing import Dict, List, Optional
import pandas as pd


def read_db_config(filename: Optional[str] = None, section: str = "mysql") -> Dict[str, str]:
    """
    Read database configuration file and return a dictionary object.
    Parameters:
        filename (str) name of the configuration file
        section (str) section of database configuration
    Returns:
      (dict) a dictionary of database parameters
    """
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), "config.ini")

    # create parser and read ini configuration file
    parser = ConfigParser()
    print(f"Reading config from {filename}")
    parser.read(filename)

    # get section, default to mysql
    db = {}
    if parser.has_section(section):
        items = parser.items(section)
        for item in items:
            db[item[0]] = item[1]
    else:
        raise Exception(f"{section} not found in the {filename} file")

    return db


def read_assay_list(assay_list_file: str) -> List[str]:

    """
    Reads json or csv file containing a list of all chembl_ids for assays
    for which full information is required.
    """
    assay_list = []
    if ".json" in assay_list_file:
        with open(assay_list_file, "r") as f:
            s = json.load(f)
        assay_list = s["assays"]
    elif assay_list_file.endswith(".csv"):
        df = pd.read_csv(assay_list_file)
        assay_list = df["chembl_id"].values

    return assay_list
