# yogGURU

Test Accuracy: **90%**

**Yoga-82 Dataset**

> Verma, M., Kumawat, S., Nakashima, Y., & Raman, S. (2020). Yoga-82: a new dataset for fine-grained classification of human poses. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 1038-1039).

The dataset consists of variations in body positions, and the actual pose names.

## Dataset Preparation

1. `pip install -r requirements.txt` to install dependencies

2. `python download_dataset.py` to download the dataset.

3. `python clean_dataset.py` to remove corrupted images.
4. `test_train_split.py` to split dataset in 80 : 20 ratio for training and testing. You can change the ratio by
   updating the `RATIO` variable.

## Train Model

Note: You can skip step1 (training part) if you want to save time. A already trained model is already there in `models` directory.

1. `python train_model.py` to train the Model. It saves the model in models directory.

2. `python -W ignore predict_img.py [Image Path]` to check an image.

**_Note:_** A more compact and illustrative **Jupyter Notebook** named `main.ipynb` is in `notebooks` dir for further reference.

---
