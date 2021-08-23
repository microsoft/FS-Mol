"""
Query to make list of assays meeting a set of criteria prior to full query

In this case, the query looks for assays that have more than 32 datapoints.
"""

import os
import csv
import json
import logging
from typing import Tuple, Dict, Any, List
import mysql.connector
from mysql.connector import Error

from preprocessing.utils.db_utils import read_db_config

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
) -> List[str]:

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
    assay_list_file = os.path.join(base_output_dir, "assay_list.json")

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
            with open(filename, "a") as f:
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

    return assay_list


def run():

    db_config = read_db_config()
    assay_config = read_db_config(section="assays")
    base_output_dir = assay_config["output_dir"]

    _ = run_initial_query(db_config, base_output_dir)


if __name__ == "__main__":
    run()
