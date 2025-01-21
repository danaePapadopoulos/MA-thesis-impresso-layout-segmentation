import math
import json
import logging

import boto3

import pandas as pd
from dask import bag as db
import dask.dataframe as dd
from dask.distributed import Client
from dask.diagnostics import ProgressBar

from impresso_commons.utils.s3 import IMPRESSO_STORAGEOPT

# Local cluster using all CPU cores
client = Client(
    processes=False
)  # here set processes=False to run a dask worker not as a daemon


# read config variables
with open("/scratch/students/danae/src/scripts/data_preparation_config.json", "r") as file:
    config = json.load(file)

# set dictionary of journals
KNOWN_JOURNALS_DICT = {
    "SNL-RERO": [
        "BDC",
        "CDV",
        "DLE",
        "EDA",
        "EXP",
        "IMP",
        "JDF",
        "JDV",
        "LBP",
        "LCE",
        "LCG",
        "LCR",
        "LCS",
        "LES",
        "LNF",
        "LSE",
        "LSR",
        "LTF",
        "LVE",
        "EVT",
    ],
    "LeTemps": ["JDG", "GDL"],
    "NZZ": ["NZZ"],
    "SWA": ["arbeitgeber", "handelsztg"],
    "FedGaz": ["FedGazDe", "FedGazFr"],
    "BNL": [
        "actionfem",
        "armeteufel",
        "avenirgdl",
        "buergerbeamten",
        "courriergdl",
        "deletz1893",
        "demitock",
        "diekwochen",
        "dunioun",
        "gazgrdlux",
        "indeplux",
        "kommmit",
        "landwortbild",
        "lunion",
        "luxembourg1935",
        "luxland",
        "luxwort",
        "luxzeit1844",
        "luxzeit1858",
        "obermosel",
        "onsjongen",
        "schmiede",
        "tageblatt",
        "volkfreu1869",
        "waechtersauer",
        "waeschfra",
    ],
    "SNL-RERO2": [
        "BLB",
        "BNN",
        "DFS",
        "DVF",
        "EZR",
        "FZG",
        "HRV",
        "LAB",
        "LLE",
        "MGS",
        "NTS",
        "NZG",
        "SGZ",
        "SRT",
        "WHD",
        "ZBT",
    ],
    "SNL-RERO3": [
        "CON",
        "DTT",
        "FCT",
        "GAV",
        "GAZ",
        "LLS",
        "OIZ",
        "SAX",
        "SDT",
        "SMZ",
        "VDR",
        "VHT",
    ],
    "BNF": ["excelsior", "lafronde", "marieclaire", "oeuvre"],
    "BNF-EN": [
        "jdpl",
        "legaulois",
        "lematin",
        "lepji",
        "lepetitparisien",
        "oecaen",
        "oerennes",
    ],
    "BCUL": [
        "ACI",
        "Castigat",
        "CL",
        "Croquis",
        "FAMDE",
        "FAN",
        # "feuilleP",  # (no OCR)
        # "feuillePMA",  # (no OCR)
        "GAVi",
        "AV",
        "JY2",
        "JV",
        "JVE",
        "JH",
        "OBS",
        "Bombe",
        "Cancoire",
        "Fronde",
        "Griffe",
        "Guepe1851",
        "Guepe1887",
        "RLA",
        "Charivari",
        "CharivariCH",
        "Grelot",
        "Moniteur",
        # "Moustique",  # (no OCR)
        "ouistiti",
        # "PDN",  # (no OCR)
        "PDL",
        "PJ",
        "TouSuIl",
        "VVS1",
        "MESSAGER",
        "PS",
        "NV",
        "ME",
        "MB",
        "NS",
        # "RN",  # (no OCR)
        "FAM",
        "FAV1",
        "EM",
        "esta",
        "PAT",
        "VVS",
        "NV1",
        "NV2",
        # "RN1",  # (no OCR)
        # "RN2",  # (no OCR)
    ],
}


def get_journals_df(journal_dict, libraries):
    # filter the journal dictionary to only keep journals of libraries of interest
    journal_dict = {key: journal_dict[key] for key in libraries}

    # create a dataframe containing information about each journal of interest
    data = [
        (library, journal)
        for library, journals in journal_dict.items()
        for journal in journals
    ]
    journals_df = pd.DataFrame(data=data, columns=["library", "journal"])

    return journals_df


s3 = boto3.client(
    "s3",
    aws_secret_access_key=IMPRESSO_STORAGEOPT["secret"],
    aws_access_key_id=IMPRESSO_STORAGEOPT["key"],
    endpoint_url=IMPRESSO_STORAGEOPT["client_kwargs"]["endpoint_url"],
)


### ISSUES


def list_journal_issue_files(journal, bucket_name="12-canonical-final"):
    print(f"Listing Issue Files of '{journal}' journal")

    prefix = journal + "/issues/"
    delimiter = ""
    result = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter=delimiter)

    files = []
    if "Contents" in result:
        files = [file["Key"] for file in result["Contents"]]

    # check that all files are issues
    for f in files:
        try:
            f.endswith("-issues.jsonl.bz2")
        except Exception as err:
            print(
                f"Unexpected {err=}, file name not in correct format '-issues.jsonl.bz2'"
            )
            raise

    return files


def get_bag_of_issues(
    issues_file, bucket_name="12-canonical-final", storage_options=IMPRESSO_STORAGEOPT
):
    bag = db.read_text(
        "s3://" + bucket_name + "/" + issues_file,
        storage_options=storage_options,
    ).map(json.loads)

    return bag  # use compute() to get values


def get_cis_of_issue_df(issue):
    keys = ["id", "pp", "tp", "t", "l", "ro"]  # content item elements/keys to lookup
    nan = math.nan  # pre-define NaN for missing values

    cis = []
    for ci_dic in issue["i"]:
        ci_m = ci_dic.get(
            "m", {}
        )  # access 'm' only once, default to empty dict if missing
        ci = [ci_m.get(key, nan) for key in keys]
        cis.append(ci)

    cis = pd.DataFrame(data=cis, columns=["ci_" + key for key in keys])

    return cis


def get_issue_df(issue):

    keys = ["id", "cdt", "pp", "iiif_manifest_uri"]
    nan = math.nan  # pre-define NaN for missing values

    issue_items = [issue.get(key, nan) for key in keys]
    ci_df = get_cis_of_issue_df(issue)
    df = pd.DataFrame([issue_items] * len(ci_df), columns=keys)
    df = pd.concat([df, ci_df], axis=1).reset_index(drop=True)

    return df


def get_issues_df(issues):
    return pd.concat(map(get_issue_df, issues), ignore_index=True)


def process_journal_issues(
    journals,
    bucket_name="12-canonical-final",
    storage_options=IMPRESSO_STORAGEOPT,
    file_names=False,
):
    issue_bags = []

    # iterate over journals and issue files
    for journal in journals:
        issue_files = list_journal_issue_files(journal, bucket_name=bucket_name)

        # for each file, create a bag
        for fn in issue_files:
            if (file_names and (fn in file_names)) | (not file_names):
                print(fn)
                issue_bag = get_bag_of_issues(fn, bucket_name, storage_options)
                issue_bags.append(issue_bag)

    # concatenate all issue bags together into a single large Bag for parallel processing
    issue_bags = db.concat(issue_bags)

    # apply the get_issues_df function lazily across all issues in parallel
    issues_df_bag = issue_bags.map(get_issue_df)

    with ProgressBar():
        # convert the result to a single Pandas df using Dask df for parallelism
        result_df = issues_df_bag.compute()

    combined_df = pd.concat(
        result_df, ignore_index=True
    )  # can convert to Dask df instead of a Pandas df

    return combined_df


### PAGES


def list_journal_page_files(journal, bucket_name="12-canonical-final"):
    print(f"Listing Page Files of '{journal}' journal")

    prefix = journal + "/pages/"
    delimiter = ""

    # we use paginator to overcome the 1000 record limitation of list-objects-v2
    paginator = s3.get_paginator("list_objects_v2")
    results = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter=delimiter)

    files = []
    for result in results:
        if "Contents" in result:
            files.append([obj["Key"] for obj in result["Contents"]])
    # flatten list
    files = [f for file_sublist in files for f in file_sublist]

    # check that all files are pages
    for f in files:
        try:
            f.endswith("-pages.jsonl.bz2")
        except Exception as err:
            print(
                f"Unexpected {err=}, file name not in correct format '-pages.jsonl.bz2'"
            )
            raise

    return files
