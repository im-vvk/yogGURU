import pandas as pd
import requests
from tqdm import tqdm
import os

# asanas = ["Bow_Pose_or_Dhanurasana_.txt",
#           "Bridge_Pose_or_Setu_Bandha_Sarvangasana_.txt",
#           "Cobra_Pose_or_Bhujangasana_.txt",
#           "Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_.txt",
#           "Tree_Pose_or_Vrksasana_.txt"]

cwd = os.getcwd()
dataset_path = cwd + "/data/yoga_dataset"

asanas_files = os.listdir(dataset_path+"/links/")

for fasana in asanas_files:
    srcpath = dataset_path + "/links/" + fasana

    df = pd.read_csv(srcpath, sep='\t', header=None)
    casana = df.iloc[0][0].split('/')[0]
    destpath = cwd + "/data/yoga_dataset/images/train/" + casana
    corrupted_files_path = cwd + "/data/yoga_dataset/corrupted_files"

    try:
        os.mkdir(destpath)
    except:
        pass
    corrupted_files = []

    for i in tqdm(range(len(df))):
        filename = df.iloc[i][0].split('/')[1]
        try:
            url = df.iloc[i][1]
            r = requests.get(url)

            header = r.headers
            content_type = header.get('content-type')
            if 'text' in content_type.lower():
                raise RuntimeError
            if 'html' in content_type.lower():
                raise RuntimeError

            if r.status_code // 100 == 2:
                with open(f"{destpath}/{filename}", "wb") as f:
                    f.write(r.content)
            else:
                raise RuntimeError
        except:
            corrupted_files.append(filename)

    pd.DataFrame(corrupted_files).to_csv(
        f"{corrupted_files_path}/{casana}_corr.csv", index=False, header=False)
