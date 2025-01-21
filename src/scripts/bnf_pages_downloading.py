import os
import pandas as pd
from tqdm import tqdm

test_sample = pd.read_csv("/scratch/students/danae/data/data_preparation/samples/pages/test.csv")
train_small_sample = pd.read_csv("/scratch/students/danae/data/data_preparation/samples/pages/train_small.csv")
train_large_sample = pd.read_csv("/scratch/students/danae/data/data_preparation/samples/pages/train_large.csv")

#pages_df = pd.concat([test_sample, train_small_sample, train_large_sample], ignore_index=True).drop_duplicates(subset="page_id")
pages_df = pd.concat([train_large_sample], ignore_index=True).drop_duplicates(subset="page_id")
pages_df["issue_id"] = pages_df["page_id"].apply(lambda x: x.split("-p")[0])

# get journals
journals = pages_df["journal"].sort_values().drop_duplicates().to_list()

# get image uri for every page of a particular journal collection
uris_df = pd.DataFrame(columns=["page_id", "iiif_img_base_uri"])
for journal in tqdm(journals, desc="Looping through Journals"):
    df = pd.read_parquet(f"/scratch/students/danae/data/data_preparation/samples/pages_bbox/{journal}_pages_v2.parquet.gzip", columns=["page_id", "iiif_img_base_uri"]).drop_duplicates()
    uris_df = pd.concat([uris_df, df], ignore_index=True)
    
uris_df = uris_df[uris_df["page_id"].isin(pages_df["page_id"])]

# add '/full/full/0/default.jpg' to url to get full page
uris_df["iiif_img_base_uri"] = uris_df["iiif_img_base_uri"].apply(lambda x: x + "/full/full/0/default.jpg")

import time
import requests
import shutil
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def download_iiif_image(iiif_url, dest_path, session):
    try:
        with session.get(iiif_url, stream=True) as r:
            if r.status_code == 200:
                r.raw.decode_content = True
                with open(dest_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                # print(f"Downloaded: {iiif_url}")
            else:
                print(f"Failed: {iiif_url} with status {r.status_code}")
    except Exception as e:
        print(f"Error: {iiif_url} -> {str(e)}")
        time.sleep(60*10)
        

def download_images_sequential(image_urls, page_ids, out_dir_images, delay=0.5):
    session = create_session()
    for iiif_url, page_id in tqdm(zip(image_urls, page_ids), total=len(image_urls), desc="Downloading Images"):
        dest_path = os.path.join(out_dir_images, page_id + ".jpg")
        if not os.path.exists(dest_path):
            print(f"Downloading {iiif_url} -> {dest_path}")
            start = time.time()
            download_iiif_image(iiif_url, dest_path, session)
            end = time.time()
            if (end - start) < 10:
                time.sleep(10)
            else:
                time.sleep(delay)
        else:
            print(f"Skipping {dest_path}, already exists.")
            
# output dirs
out_dir = "/scratch/students/danae/data"
out_dir_images = os.path.join(out_dir, "images")
os.makedirs(out_dir_images, exist_ok=True)

# BNF only
df = uris_df[uris_df["iiif_img_base_uri"].str.startswith("https://gallica.bnf")]
# download images
download_images_sequential(image_urls=df["iiif_img_base_uri"], page_ids=df["page_id"], out_dir_images=out_dir_images, delay=1)