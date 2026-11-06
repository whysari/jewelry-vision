import argparse
import numpy as np
import pickle
import platform
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.preprocessing.image
from tensorflow import keras
from modelFunctions import CNNModel, RNNModel
import dataFunctions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def searchCaption(caption, data):
    predicted_image = False
    for img, cap in data.items():
        if caption == cap:
            predicted_image = True
            break

    return predicted_image, img


def obtain_CCR(ccr, total):
    if total == 0:
        return 0.0
    return (ccr / total) * 100


def get_num_accesories_4(captions_array):
    num_necklaces = 0
    num_earrings  = 0
    num_rings     = 0
    num_bracelets = 0

    for caption in captions_array:
        # grab the last word and normalize to lowercase
        category = caption.strip().split()[-1].lower()

        if category == 'зүүлт':       # necklace
            num_necklaces += 1
        elif category == 'ээмэг':     # earrings
            num_earrings += 1
        elif category == 'бөгж':      # ring
            num_rings += 1
        elif category == 'бугуйвч':   # bracelet
            num_bracelets += 1

    return num_necklaces, num_earrings, num_rings, num_bracelets



def get_num_accesories_6(captions_array):
    num_earrings_silver   = 0
    num_earrings_gold     = 0
    num_necklaces_silver  = 0
    num_necklaces_gold    = 0
    num_rings_gold        = 0
    num_bracelets_gold    = 0

    for caption in captions_array:
        cat = caption.strip().split()[-1].lower()
        text = caption.lower()

        # Earrings
        if cat == 'ээмэг':
            if 'мөнгөн' in text:
                num_earrings_silver += 1
            elif 'алтан' in text:
                num_earrings_gold += 1

        # Necklaces
        elif cat == 'зүүлт':
            if 'мөнгөн' in text:
                num_necklaces_silver += 1
            elif 'алтан' in text:
                num_necklaces_gold += 1

        # Rings (only gold)
        elif cat == 'бөгж' and 'алтан' in text:
            num_rings_gold += 1

        # Bracelets (only gold)
        elif cat == 'бугуйвч' and 'алтан' in text:
            num_bracelets_gold += 1

    return (
        num_earrings_silver,
        num_earrings_gold,
        num_necklaces_silver,
        num_necklaces_gold,
        num_rings_gold,
        num_bracelets_gold,
    )


def get_CCR_4(caption, ccr_necklaces, ccr_earrings, ccr_rings, ccr_bracelets):
    text = caption.lower()

    if 'зүүлт' in text:        # зүүлт = necklace
        ccr_necklaces += 1
    elif 'ээмэг' in text:      # ээмэг = earrings
        ccr_earrings += 1
    elif 'бөгж' in text:       # бөгж = ring
        ccr_rings += 1
    elif 'бугуйвч' in text:    # бугуйвч = bracelet
        ccr_bracelets += 1

    return ccr_necklaces, ccr_earrings, ccr_rings, ccr_bracelets



def get_CCR_6(
    caption,
    ccr_earrings_silver,
    ccr_earrings_gold,
    ccr_necklaces_silver,
    ccr_necklaces_gold,
    ccr_rings_gold,
    ccr_bracelets_gold,
):
    text = caption.lower()

    # Earrings: silver vs gold
    if 'ээмэг' in text:
        if 'мөнгөн' in text:
            ccr_earrings_silver += 1
        elif 'алтан' in text:
            ccr_earrings_gold += 1

    # Necklaces: silver vs gold
    elif 'зүүлт' in text:
        if 'мөнгөн' in text:
            ccr_necklaces_silver += 1
        elif 'алтан' in text:
            ccr_necklaces_gold += 1

    # Rings: only gold
    elif 'бөгж' in text and 'алтан' in text:
        ccr_rings_gold += 1

    # Bracelets: only gold
    elif 'бугуйвч' in text and 'алтан' in text:
        ccr_bracelets_gold += 1

    return (
        ccr_earrings_silver,
        ccr_earrings_gold,
        ccr_necklaces_silver,
        ccr_necklaces_gold,
        ccr_rings_gold,
        ccr_bracelets_gold,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to test Image Captioning models')

    parser.add_argument('--model', help="Model to test",
                        dest="model_name", type=str, required=True)
    parser.add_argument('--test_path', help="Path to test data",
                        dest="test_path", type=str, required=True)
    parser.add_argument("--rw_images", help="Re-write encoded images if already exists.",
                        dest="rewrite_images", type=bool, default=False)

    args = parser.parse_args()

    # Get path files
    print("Getting data...")
    images_path = os.path.join(args.test_path, "test")
    captions_path = os.path.join(args.test_path, "captions_mn.txt")
    test_list = os.path.join(args.test_path, 'test.txt')

    # Get data from files
    data, max_length = dataFunctions.getData(captions_path)
    lex = dataFunctions.getLexicon(data)

    # Get images and captions arrays
    images_array, captions_array = dataFunctions.getDataArrays(data, test_list)

    # Tokenize words using idxtoword and wordtoidx
    print("Tokenizing captions...")
    idxtoword, wordtoidx = dataFunctions.getTokenizers(lex)
    vocab_size = len(idxtoword) + 1

    # Get model info
    if platform.system() == "Linux":
        cnn_type = args.model_name.split('/')[-1].split('_')[1]
        rnn_type = args.model_name.split('/')[-1].split('_')[2]
        use_embedding = args.model_name.split('/')[-1].split('_')[3]
        epochs = args.model_name.split('/')[-1].split('_')[4]
        neurons = args.model_name.split('/')[-1].split('_')[5]
        batch_size = args.model_name.split('/')[-1].split('_')[6].split('.')[0]
        name = args.test_path.split('/')[-1]
    elif platform.system() == "Windows":
        cnn_type = args.model_name.split('\\')[-1].split('_')[1]
        rnn_type = args.model_name.split('\\')[-1].split('_')[2]
        use_embedding = args.model_name.split('\\')[-1].split('_')[3]
        epochs = args.model_name.split('\\')[-1].split('_')[4]
        neurons = args.model_name.split('\\')[-1].split('_')[5]
        batch_size = args.model_name.split(
            '\\')[-1].split('_')[6].split('.')[0]
        name = args.test_path.split('\\')[-1]
    else:
        cnn_type = args.model_name.split('/')[-1].split('_')[1]
        rnn_type = args.model_name.split('/')[-1].split('_')[2]
        use_embedding = args.model_name.split('/')[-1].split('_')[3]
        epochs = args.model_name.split('/')[-1].split('_')[4]
        neurons = args.model_name.split('/')[-1].split('_')[5]
        batch_size = args.model_name.split('/')[-1].split('_')[6].split('.')[0]
        name = args.test_path.split('/')[-1]

    # Choose CNN model and get useful params
    print("Creating CNN model...")
    cnn_model = CNNModel(cnn_type)

    # Encode images and save them
    images_save = os.path.join(
        "data", f'test_images_{name}_{cnn_model.get_output_dim()}.pk1')
    if not os.path.exists(images_save) or args.rewrite_images == True:
        print(f"Encoding images to {images_save}...")
        shape = (len(images_array), cnn_model.get_output_dim())
        encoded_images = np.zeros(shape=shape, dtype=np.float16)

        for i, img in enumerate(images_array):
            image_path = os.path.join(images_path, img)
            img = tensorflow.keras.preprocessing.image.load_img(
                image_path, target_size=(cnn_model.get_height(), cnn_model.get_width()))
            encoded_images[i] = cnn_model.encode_image(img)

        with open(images_save, 'wb') as f:
            pickle.dump(encoded_images, f)
            print("Saved encoded test images to disk")
    else:
        print(f"Loading images from {images_save}...")
        with open(images_save, 'rb') as f:
            encoded_images = pickle.load(f)

    # Get number of accesories are of each type
    if name == "Medium":
        num_earrings_silver, num_earrings_gold, num_necklaces_silver, num_necklaces_gold, num_rings_gold, num_bracelets_gold = get_num_accesories_6(
            captions_array)
    else:
        num_necklaces, num_earrings, num_rings, num_bracelets = get_num_accesories_4(
            captions_array)

    print("Loading model...: ")
    rnn_model = RNNModel(max_length=max_length)
    rnn_model.set_model(keras.models.load_model(args.model_name))
    print("Model loaded")

    with open(os.path.join("models", name, f'idxtoword_{cnn_type}_{rnn_type}_{use_embedding}_{epochs}_{neurons}_{batch_size}.pk1'), 'rb') as f:
        idxtoword = pickle.load(f)
        print("Loaded idxtoword from disk")

    with open(os.path.join("models", name, f'wordtoidx_{cnn_type}_{rnn_type}_{use_embedding}_{epochs}_{neurons}_{batch_size}.pk1'), 'rb') as f:
        wordtoidx = pickle.load(f)
        print("Loaded wordtoidx from disk")
        print("================================")

    ccr = 0

    ccr_necklaces = 0
    ccr_earrings = 0
    ccr_rings = 0
    ccr_bracelets = 0

    ccr_earrings_silver = 0
    ccr_earrings_gold = 0
    ccr_necklaces_silver = 0
    ccr_necklaces_gold = 0
    ccr_rings_gold = 0
    ccr_bracelets_gold = 0

    expected_array = []
    obtained_array = []
    matrix_labels = []

    for i in range(len(images_array)):
        image = images_array[i]
        image_encoded = encoded_images[i]

        print("Expected image: ", image)
        print("Expected caption:", data[image])
        caption = rnn_model.generate_caption(
            image_encoded, wordtoidx, idxtoword)

        print("Obtained caption:", caption)

        expected_array.append(data[image])
        obtained_array.append(caption)

        if data[image] not in matrix_labels:
            matrix_labels.append(data[image])

        predicted_image, img = searchCaption(caption, data)

        if predicted_image:
            if caption == data[image]:
                ccr += 1

                if name == "Medium":
                    ccr_earrings_silver, ccr_earrings_gold, ccr_necklaces_silver, ccr_necklaces_gold, ccr_rings_gold, ccr_bracelets_gold = get_CCR_6(caption, ccr_earrings_silver, ccr_earrings_gold, ccr_necklaces_silver,
                                                                                                                                                    ccr_necklaces_gold, ccr_rings_gold, ccr_bracelets_gold)

                else:
                    ccr_necklaces, ccr_earrings, ccr_rings, ccr_bracelets = get_CCR_4(caption, ccr_necklaces, ccr_earrings,
                                                                                        ccr_rings, ccr_bracelets)

        else:
            print("The obtained caption does not exist in the dataset")

        print("================================")

    print(ccr)
    ccr = obtain_CCR(ccr, len(images_array))
    print("CCR =", round(ccr, 2), "%")

    if name == "Medium":

        ccr_earrings_silver = obtain_CCR(
            ccr_earrings_silver, num_earrings_silver)
        ccr_earrings_gold = obtain_CCR(ccr_earrings_gold, num_earrings_gold)
        ccr_necklaces_silver = obtain_CCR(
            ccr_necklaces_silver, num_necklaces_silver)
        ccr_necklaces_gold = obtain_CCR(ccr_necklaces_gold, num_necklaces_gold)
        ccr_rings_gold = obtain_CCR(ccr_rings_gold, num_rings_gold)
        ccr_bracelets_gold = obtain_CCR(ccr_bracelets_gold, num_bracelets_gold)

        print("CCR Earrings Silver =", round(ccr_earrings_silver, 2), "%")
        print("CCR Earrings Gold =", round(ccr_earrings_gold, 2), "%")
        print("CCR Necklaces Silver =", round(ccr_necklaces_silver, 2), "%")
        print("CCR Necklaces Gold =", round(ccr_necklaces_gold, 2), "%")
        print("CCR Rings Gold =", round(ccr_rings_gold, 2), "%")
        print("CCR Bracelets Gold =", round(ccr_bracelets_gold, 2), "%")

    else:

        ccr_necklaces = obtain_CCR(ccr_necklaces, num_necklaces)
        ccr_earrings = obtain_CCR(ccr_earrings, num_earrings)
        ccr_rings = obtain_CCR(ccr_rings, num_rings)
        ccr_bracelets = obtain_CCR(ccr_bracelets, num_bracelets)

        # print("CCR Necklaces =", round(ccr_necklaces, 2), "%")
        # print("CCR Earrings =", round(ccr_earrings, 2), "%")
        # print("CCR Rings =", round(ccr_rings, 2), "%")
        # print("CCR Bracelets =", round(ccr_bracelets, 2), "%")

    print("---------------------")
    if name == "Medium" or name == "General_Accessories":
        print("CONFUSION MATRIX")
        matrix = confusion_matrix(
            expected_array, obtained_array, labels=matrix_labels)

        print(matrix)
        print(matrix_labels)

        df_cm = pd.DataFrame(matrix, matrix_labels, matrix_labels)
        sn.set_theme(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})

        plt.show()
