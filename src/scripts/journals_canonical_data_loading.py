import sys
import os
import random

import dask
import numpy as np
import pandas as pd
# from dask.bag import random
from dask.dataframe import from_pandas
from dask.distributed import Client, progress, LocalCluster
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
# set Dask configuration to avoid automatic string conversion in DataFrames
# dask.config.set({"dataframe.convert-string": False})
dask.config.set({'distributed.scheduler.worker-ttl': None})
dask.config.set({"distributed.comm.retry.count": 10})
dask.config.set({"distributed.comm.timeouts.connect": 30})
dask.config.set({"distributed.worker.memory.terminate": False})

# add the path to the parent directory of `src` to the system path
# sys.path.insert(0, os.path.abspath('../..'))
# import custom utility functions from the `src` package
# from src.utils.data_preparation import *
from data_preparation import *


#########################################################

# helper functions to read/select the relevant info from the page files
def get_bbxs(page):

    rows = []

    for region in page["r"]:

        for paragraph in region["p"]:
            lines_bbxs = []
            lines_tokens = []

            for line in paragraph["l"]:
                lines_bbxs.append(line["c"])
                tokens = []

                for token in line["t"]:
                    tokens.append(token["tx"])

                lines_tokens.append(tokens)

            row = {
                "bbx_region": region["c"],
                "bbx_paragraph": paragraph["c"],
                "lines_bbxs": lines_bbxs,
                "lines_tokens": lines_tokens,
                "pOf": region.get(
                    "pOf", np.nan
                ),  # pOf doesn't always appear as a key in dict
            }
            rows.append(row)

    return pd.DataFrame(rows)


def get_page_info(page):

    keys = ["id", "cdt", "iiif_img_base_uri"]
    nan = math.nan  # pre-define NaN for missing values

    items = [page.get(key, nan) for key in keys]
    bbx_df = get_bbxs(page)
    df = pd.DataFrame([items] * len(bbx_df), columns=keys)
    df = pd.concat([df, bbx_df], axis=1).reset_index(drop=True)

    return df

#########################################################

if __name__ == '__main__':
    memory_per_worker_gb = 15
    cluster = LocalCluster(n_workers=5, threads_per_worker=1, memory_limit=f"{memory_per_worker_gb}GB")
    client = cluster.get_client()

    fn = "/scratch/students/danae/data/data_preparation/samples/issues_20241112-135609.parquet.gzip"
    issues_df = pd.read_parquet(fn)

    journals = list(set(issues_df["journal"]))
    loaded_journals = ["actionfem", "avenirgdl", "courriergdl", "deletz1893", "excelsior", "lepji", "lunion", "marieclaire", "oeuvre", "oerennes"]
    journals = [j for j in journals if j not in loaded_journals]

    bucket_name = "12-canonical-final"
    storage_options = IMPRESSO_STORAGEOPT

    # sample
    k = 12000

    for journal in journals:
        print(f"Journal {journal}.")
        # list all the issues for which the canonical data is available
        file_names = list_journal_page_files(journal, bucket_name=bucket_name)

        issues = issues_df[issues_df["journal"] == journal]["id"].drop_duplicates().to_list()
        
        sample_file_names = [
            fn for fn in file_names if "-".join(fn.split("/")[-1].split("-")[:-1]) in issues
        ]
        sample_size = len(sample_file_names)
        print(k, sample_size)
        if k < sample_size:
            sample_file_names = random.sample(sample_file_names, k)

        # load the canonical data from s3 bucket
        print("Loading the canonical data from bucket...")
        bags = []
        
        for fn in sample_file_names:

            bag = db.read_text(
                "s3://" + bucket_name + "/" + fn, storage_options=storage_options
            ).map(json.loads)

            bags.append(bag)

        bags = db.concat(bags)
        
        # apply the function lazily across all pages in parallel
        pages_info = bags.map(get_page_info)
        print("Computing...")
        with ProgressBar():
            # convert the result to a single Pandas df using Dask df for parallelism
            result = pages_info.compute()

        pages_df = pd.concat(
            result, ignore_index=True
        )  # can convert to Dask df instead of a Pandas df
        
        print("Writing file...")
        # pages_df.to_parquet(f"../../data/data_preparation/samples/journals/{journal}_sample_{k}_pages.parquet.gzip", compression="gzip")
        pages_df.to_parquet(f"/scratch/students/danae/data/data_preparation/samples/journals/{journal}_sample_{k}_pages.parquet.gzip", compression="gzip")
        
        