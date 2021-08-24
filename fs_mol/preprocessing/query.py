"""
Querying ChEMBL database for all information.

Assays of interest are passed in a .jsonl file, the default is set in config.ini.

"""

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
import mysql.connector
from mysql.connector import Error

from pyreporoot import project_root

sys.path.insert(0, str(project_root(Path(__file__), root_files="requirements.txt")))

from fs_mol.preprocessing.utils.db_utils import read_db_config, read_assay_list
from fs_mol.preprocessing.utils.queries import (
    CHEMBL_ASSAY_PROTEIN,
    DISTINCT_TABLES,
    COUNT_QUERIES,
    EXTENDED_SINGLE_ASSAY_NOPROTEIN,
    FIELDNAMES,
    SUMMARY_FIELDNAMES,
    COUNTED_SUMMARY_FIELDNAMES,
    PROTEIN_FIELDS,
    CELL_FIELDS,
)

logging.basicConfig(filename="querying.log", format="%(asctime)s %(message)s", filemode="w")

logger = logging.getLogger(__name__)


def query_db(cursor, assay: str, query: str) -> Dict[str, Any]:
    assay = '"' + assay + '"'
    query = query.format(assay)
    query = query + ";"
    cursor.execute(query)
    rows = cursor.fetchall()
    return rows


def run_query_on_assay(
    cursor,
    output_dir: str,
    assay: str,
    counting_filename: str,
    summary_filenames: Dict[str, str],
) -> None:

    """
    Runs a query on a ChEMBL assay, using the chembl_id string.

    """
    logger.info("Querying database for assay {}".format(assay))
    fieldnames = FIELDNAMES.copy()

    query = CHEMBL_ASSAY_PROTEIN
    fieldnames.extend(PROTEIN_FIELDS)

    # output filename
    filename = os.path.join(output_dir, "{}.csv".format(assay))

    # full query attempt
    rows = query_db(cursor, assay, query)

    if len(rows) == 0:

        logger.warning(f"{assay} has no protein info, querying for all other fields.")
        with open(os.path.join(output_dir, "failed_protein_queries.txt"), "a") as f:
            f.writelines([f"{assay}"])

        # run alternative query to try to get additional information
        query = EXTENDED_SINGLE_ASSAY_NOPROTEIN
        rows = query_db(cursor, assay, query)
        if len(rows) == 0:
            logger.info(f"{assay} has no other info in the assays table.")
            with open(os.path.join(output_dir, "failed_all_queries.txt"), "a") as f:
                f.writelines([f"{assay}"])
        else:
            logger.info(f"Queried without protein information, saving {len(rows)} to {filename}.")
            fieldnames = FIELDNAMES.copy()
            fieldnames.extend(CELL_FIELDS)
            with open(filename, "w") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                )
                writer.writeheader()
            with open(filename, "a") as f:
                writer = csv.writer(f)
                writer.writerows(rows)

    else:
        logger.info(f"Queried, saving {len(rows)} to {filename}.")
        with open(filename, "w") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
            writer.writeheader()
        with open(filename, "a") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            assay_size = len(rows)

        # build up statistics over a number of measurements for the queries that return protein information
        # first, all units, comments, targets, target levels in each assay
        count_summary = {}
        for field in SUMMARY_FIELDNAMES:
            quote_assay = '"' + assay + '"'
            newtable = CHEMBL_ASSAY_PROTEIN.format(
                quote_assay
            )  # table for this assay with everything
            query = DISTINCT_TABLES[field].format(
                newtable
            )  # select distinct for each property table
            cursor.execute(query)
            rows = cursor.fetchall()
            with open(summary_filenames[field], "a") as f:
                writer = csv.writer(f)
                writer.writerows(rows)

            count_query = query[:-1]
            field_count_query = COUNT_QUERIES[f"num_{field}"].format(count_query)
            cursor.execute(field_count_query)
            rows = cursor.fetchall()
            count_summary[f"num_{field}"] = rows[0][0]

        count_summary["size"] = assay_size
        count_summary["chembl_id"] = assay
        with open(counting_filename, "a") as outfile:
            count_writer = csv.DictWriter(
                outfile,
                fieldnames=COUNTED_SUMMARY_FIELDNAMES,
            )
            count_writer.writerow(count_summary)


def run(args):

    db_config = read_db_config()
    assay_config = read_db_config(section="assays")

    if args.save_dir is None:
        output_dir = assay_config["output_dir"]
    else:
        output_dir = args.save_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.assay_list_file is None:
        assay_config = read_db_config(section="assays")
        assay_list = read_assay_list(assay_config["assay_list_file"])
    else:
        assay_list = read_assay_list(args.assay_list_file)

    conn = None
    cursor = None

    logger.info("Assays to query {}".format(len(assay_list)))

    # arrange output files for summaries over each field
    summary_filenames = {
        x: os.path.join(output_dir, f"summary_{x}.csv") for x in SUMMARY_FIELDNAMES
    }
    for key, f in summary_filenames.items():
        with open(f, "w") as outfile:
            writer = csv.DictWriter(
                outfile,
                fieldnames=["chembl_id", key],
            )
            writer.writeheader()

    # make file that summarises the number of unique entries for each
    # of these fields.
    counting_filename = os.path.join(output_dir, "summary_counts.csv")
    with open(counting_filename, "w") as outfile:
        count_writer = csv.DictWriter(
            outfile,
            fieldnames=COUNTED_SUMMARY_FIELDNAMES,
        )
        count_writer.writeheader()

    # access the database and perform the queries
    try:
        conn = mysql.connector.connect(**db_config)

        if conn.is_connected():
            logger.info("Connected to mysql database")

        cursor = conn.cursor()

        for assay in assay_list:
            try:
                run_query_on_assay(
                    cursor,
                    output_dir,
                    assay,
                    counting_filename,
                    summary_filenames,
                )
            except Exception as e:
                logger.exception(f"Unsuccessful query for for assay {assay}: {e}")
                continue

    except Error as e:
        logger.error(e)

    finally:
        if conn is not None and conn.is_connected():
            conn.close()
            if cursor is not None:
                cursor.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Query ChEMBL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--assay-list-file",
        type=str,
        default=None,
        help=(
            'CSV or json file containing list of assays to query the database for under "chembl_id".',
            " Optional: config.ini can be used to take default assays.jsonl",
        ),
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional output directory name as alternative to config.",
    )

    args = parser.parse_args()

    run(args)
