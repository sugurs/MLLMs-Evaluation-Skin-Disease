from PIL import Image
import google.generativeai as genai
import time
import os
import re
import sys


def get_all_imgs_private_dataset(directory, flag):
    if flag == "clinical":
        distinguish_word = "L"  # "L" identifies clinical images
    elif flag == "dermoscopic":
        distinguish_word = "J"  # "J" identifies clinical images
    
    file_paths = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            if file_path.endswith((".zip", ".xlsx", ".pdf")):
                continue
            elif file_path.endswith((".JPG", ".jpg", ".png", ".jpeg", ".PNG")):
                if (f"{distinguish_word}\\" in file_path or f" {distinguish_word}" in file_path):
                    file_paths.append(file_path)
                
    return file_paths


def get_all_imgs_derm7pt_dataset(directory, flag):
    file_paths = []

    with open('derm7pt_dataset.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            content = line.strip()
            # Split content into components
            parts = content.split(', ')
            clinic_path, derm_path, dis_orig, dis = parts[0], parts[1], parts[2], parts[3]
            
            selected_path = clinic_path if flag == 'clinical' else derm_path
            file_paths.append(os.path.join(directory, selected_path))
    
    return file_paths


def get_all_imgs_clinical_dataset(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith((".JPG", ".jpg", ".png", ".jpeg", ".PNG")):
                file_paths.append(file_path)
    return file_paths


def get_all_imgs_isic_dataset(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith((".JPG", ".jpg", ".png", ".jpeg", ".PNG")):
                file_paths.append(file_path)
    return file_paths


if __name__ == "__main__":
    TEST_TIMES = 10

    TEST_DATASET = "Private_dataset"    # "Private_dataset", "Derm7pt_dataset", "Clinical_dataset", or "ISIC_dataset"
    TEST_MODALITY = "clinical"          # "clinical" or "dermoscopic"
    TEST_LANGUAGE = "English"           # "English" or "Chinese"

    if TEST_LANGUAGE == "English":
        TEXT_QUERY_SYSTEM = "You are a medical artificial intelligence assistant. You can diagnose patients based on the provided information. Your goal is to correctly diagnose the patient. Based on the provided information you will provide the main diagnosis. Don't write any further information. Give only a single diagnosis."
        TEXT_QUERY_USER = f"This is the patient's {TEST_MODALITY} image. Please provide the most likely final diagnosis based on this image."
    elif TEST_LANGUAGE == "Chinese":
        TEXT_QUERY_SYSTEM = "您是一名医疗人工智能助手。您可以根据提供的信息为患者诊断。您的目标是准确诊断患者。根据提供的信息，您需给出主要诊断。无需附加其他信息，仅提供一个诊断结果。"
        test_modality_chinese = "临床" if TEST_MODALITY == 'clinical' else "皮肤镜"
        TEXT_QUERY_USER = f"这是患者的{test_modality_chinese}影像。请根据该影像提供最可能的最终诊断。"
    else:
        exit(-1)

    if TEST_DATASET == "Private_dataset":
        ALL_IMGS_DIR = "D:\\BaiduNetdiskDownload\\TestImages"
        all_imgs = get_all_imgs_private_dataset(ALL_IMGS_DIR, TEST_MODALITY)
    elif TEST_DATASET == "Derm7pt_dataset":
        ALL_IMGS_DIR = "D:\\BaiduNetdiskDownload\\Derm7pt\\release_v0\\images"
        all_imgs = get_all_imgs_derm7pt_dataset(ALL_IMGS_DIR, TEST_MODALITY)
    elif TEST_DATASET == "Clinical_dataset" and TEST_MODALITY == "clinical":
        ALL_IMGS_DIR = "D:\\BaiduNetdiskDownload\\Clinical_Selected_100"
        all_imgs = get_all_imgs_clinical_dataset(ALL_IMGS_DIR)
    elif TEST_DATASET == "ISIC_dataset" and TEST_MODALITY == "dermoscopic":
        ALL_IMGS_DIR = "D:\\BaiduNetdiskDownload\\ISIC_Selected_100"
        all_imgs = get_all_imgs_isic_dataset(ALL_IMGS_DIR)
    else:
        exit(-1)

    print(f"Test on the {TEST_DATASET}, finding a total of {len(all_imgs)} images")

    TXT_FILE = f'results/gemini_{TEST_DATASET}_{TEST_MODALITY}.txt'

    API_KEY = "AIzaSyAQNrLLNp3FoSs8j_asLZTq2-Ma1RZFk94"  # Replace with your API key of Gemini

    img_cnt = 0

    genai.configure(api_key=API_KEY, transport='rest')
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    for each_img in all_imgs:
        img_cnt += 1

        img_base = Image.open(each_img)
        if img_base.mode == 'RGBA':
            img_base = img_base.convert('RGB')

        print(f"Processing [{img_cnt}] {each_img}")
        res = []
        for i in range(TEST_TIMES):
            while True:
                try:
                    response = model.generate_content([(TEXT_QUERY_SYSTEM), img_base, (TEXT_QUERY_USER),])
                    break
                except Exception as e:
                    print(f"    {e}，retry------------------------------------")
                    time.sleep(1)

            try:
                res_data = re.sub("[\r\n]", "", str(response.text))
            except Exception as e:
                res_data = re.sub("[\r\n]", "", str(sys.exc_info()))

            print(res_data)

            res.append(f"[{i+1}]{res_data}")
            time.sleep(3)

        result = ' '.join(res)
        new_content = f"{img_cnt}. {each_img} {result}\n"

        print(new_content)

        with open(TXT_FILE, 'a', encoding='utf-8') as file:
            file.writelines(new_content)
