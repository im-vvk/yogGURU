# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import requests
from tqdm import tqdm
import os


# %%
srcpath = "..\data\Yoga-82\Yoga-82\yoga_dataset_links\Bow_Pose_or_Dhanurasana_.txt"
df = pd.read_csv(srcpath, sep='\t', header=None)
try:
    os.mkdir(df.iloc[0][0].split('/')[0])
except:
    pass
dirname = df.iloc[0][0].split('/')[0]
corrupted_files = []
df.head()


# %%
for i in tqdm(range(len(df))):
    try:
        url = df.iloc[i][1]
        filename = df.iloc[i][0].split('/')[1]
        r = requests.get(url)

        header = r.headers
        content_type = header.get('content-type')
        if 'text' in content_type.lower():
            raise RuntimeError
        if 'html' in content_type.lower():
            raise RuntimeError

        if (r.status_code)//100 == 2:
            with open(f"./{dirname}/{filename}", "wb") as f:
                f.write(r.content)
        else:
            raise RuntimeError
    except:
        corrupted_files.append(i)

pd.DataFrame(corrupted_files).to_csv(
    f"{dirname}-corr_files.csv", index=False, header=False)


# %%
# for i in tqdm(corrupted_files):
#     try:
#         url = df.iloc[i][1]
#         filename = df.iloc[i][0].split('/')[1]
#         r = requests.get(url)

#         header = r.headers
#         content_type = header.get('content-type')

#         with open(f"./{dirname}/{filename}", "wb") as f:
#             f.write(r.content)
#     except:
#         print(df.iloc[i])


# %%
