"""
Query to make list of assays meeting a set of criteria prior to full query

In this case, the query looks for assays that have more than 32 datapoints.
"""

import os
import sys
import csv
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import mysql.connector
from mysql.connector import Error

import pandas as pd

from pyreporoot import project_root

sys.path.insert(0, str(project_root(Path(__file__), root_files="requirements.txt")))

from fs_mol.preprocessing.utils.db_utils import read_db_config

logger = logging.getLogger(__name__)


def query_db(cursor, c_score):
    query = (
        "SELECT assays.chembl_id, assays.assay_type, counts.mol_num, assays.confidence_score"
        " FROM "
        "(SELECT acts.assay_id, count(acts.assay_id) AS mol_num "
        "  FROM activities acts GROUP BY acts.assay_id having mol_num > 32"
        ") AS counts "
        " JOIN assays ON counts.assay_id = assays.assay_id "
        f"WHERE assays.confidence_score = '{c_score}';"
    )
    cursor.execute(query)
    rows = cursor.fetchall()
    return rows


def get_confidence_scores(cursor, output_dir: str) -> Tuple[str, str]:
    """
    With an open database cursor, query for the meanings of the
    confidence scores and save out to a csv.
    """
    query = "select csl.confidence_score, csl.description " "from confidence_score_lookup as csl;"
    cursor.execute(query)
    score_rows = cursor.fetchall()
    print(score_rows)
    with open(os.path.join(output_dir, "confidence_scores.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=["confidence_score", "description"])
        writer.writeheader()
    with open(os.path.join(output_dir, "confidence_scores.csv"), "a") as f:
        writer = csv.writer(f)
        writer.writerows(score_rows)

    return score_rows


def run_initial_query(
    db_config: Dict[str, Any], base_output_dir: str, close_cursor: bool = True
) -> str:

    """
    Query to get confidence scores and all assay names for each score
    that meet basic criteria of having more than 32 datapoints.
    """
    conn = None
    cursor = None

    output_dir = os.path.join(base_output_dir, "assay_lists")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving assay lists to {output_dir}")

    assay_list = []
    assay_list_file = os.path.join(base_output_dir, "assays.jsonl")

    try:
        conn = mysql.connector.connect(**db_config)

        if conn.is_connected():
            logger.info("Connected to mysql database")

        # first, grab the confidence scores
        cursor = conn.cursor()

        confidence_scores = get_confidence_scores(cursor, output_dir)

        if cursor is not None:
            cursor.close()

        # for each confidence score, extract the set of assays with > 32 molregnos
        cursor = conn.cursor()
        for score in confidence_scores:
            logger.info("Querying database for confidence score {}".format(score[0]))
            rows = query_db(cursor, score[0])
            filename = os.path.join(output_dir, "assays_{}.csv".format(score[0]))
            logger.info(f"Queried, found {len(rows)}, saving data to {filename}.")
            with open(filename, "w") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "chembl_id",
                        "assay_type",
                        "molregno_num",
                        "confidence_score",
                    ],
                )
                writer.writeheader()
                writer = csv.writer(f)
                writer.writerows(rows)

            for row in rows:
                assay_list.append(row[0])

    except Error as e:
        print(e)

    finally:
        if conn is not None and conn.is_connected() and close_cursor:
            conn.close()
            if cursor is not None:
                cursor.close()

    # write out all the chembl_ids of the assays meeting the criteria
    logger.info(f"saving out list of assays to {assay_list_file}")
    assay_list_dict = {"assays": assay_list}
    with open(assay_list_file, "w") as jf:
        json.dump(assay_list_dict, jf)

    return assay_list_file


def recreate_assay_list_file(base_output_dir: str, assay_list_file: str) -> None:
    all_assays = []
    for cs in range(0, 10):
        assays = list(
            pd.read_csv(os.path.join(base_output_dir, f"assays_{cs}.csv")).chembl_id.values
        )
        all_assays.extend(assays)
    assays = {}
    assays["assays"] = all_assays

    with open(assay_list_file, "w") as jf:
        json.dump(assays, jf)


def run():

    db_config = read_db_config()
    assay_config = read_db_config(section="initialquery")
    base_output_dir = assay_config["output_dir"]

    assay_list_file = run_initial_query(db_config, base_output_dir)

    # check that the assay list file exists and is not empty
    with open(assay_list_file, "r") as jf:
        assays = json.load(jf)["assays"]

    if len(assays) == 0:
        logger.info("Assay list file is empty, repopulating from intermediate files.")
        recreate_assay_list_file(base_output_dir, assay_list_file)


if __name__ == "__main__":
    run()
