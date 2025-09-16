import os
import re
import sys
import time
from PIL import Image
import chardet
import dashscope


def run_sequence_cd_en(model, clinical_img, dermoscopy_img):
    
    clinical_img = Image.open(clinical_img)
    dermoscopy_img = Image.open(dermoscopy_img)

    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_en}]}
        ]
                
        image_message_1 = [
            {"role": "user",
            "parts": [clinical_img, ("This is the patient's clinical image. Please provide a single, most likely diagnosis based on this image. Don't write any further information.")]}
        ]
        messages.append(image_message_1)
        
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model, messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_1 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_1 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_1]})
        res1 = f"Current diagnosis: {diagnosis_1}"
        print(res1)

        image_message_2 = {
            "role": "user",
            "parts": [dermoscopy_img, ("This is the patient's dermoscopic image. Please provide a single, most likely diagnosis by combining the provided clinical image and dermoscopic image. Don't write any further information.")]
        }
        messages.append(image_message_2)
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model, messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_2 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_2 = re.sub("[\r\n]", "", str(sys.exc_info()))

        messages.append({'role':'model',
                        'parts':[diagnosis_2]})
        res2 = f"Final diagnosis: {diagnosis_2}"
        print(res2)

        res.append("[%d]%s" % (i, res1+res2))
        print("[%d]%s" % (i, res1+res2))
        
    result = ' '.join([str(item)for item in res])
    return result

def run_sequence_cm_en(model, clinical_img, medical_hist):
    
    clinical_img = Image.open(clinical_img)
    
    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_en}]}
        ]
                
        image_message_1 = [
            {"role": "user",
            "parts": [clinical_img, ("This is the patient's clinical image. Please provide a single, most likely diagnosis based on this image. Don't write any further information.")]}
        ]
        messages.append(image_message_1)
        
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model, messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_1 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_1 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_1]})
        res1 = f"Current diagnosis: {diagnosis_1}"
        print(res1)

        messages.append({
            "role": "user",
            "parts": [f"This is the patient's medical history: {medical_hist}. Please provide a single, most likely diagnosis by combining the provided clinical image and medical history. Don't write any further information."]
        })
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model, messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_2 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_2 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_2]})
        res2 = f"Final diagnosis: {diagnosis_2}"
        print(res2)

        res.append("[%d]%s" % (i, res1+res2))
        print("[%d]%s" % (i, res1+res2))
        
    result = ' '.join([str(item)for item in res])
    return result

def run_sequence_cdm_en(model, clinical_img, dermoscopy_img, medical_hist):
    
    clinical_img = Image.open(clinical_img)
    dermoscopy_img = Image.open(dermoscopy_img)

    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_en}]}
        ]
                
        image_message_1 = [
            {"role": "user",
            "parts": [clinical_img, ("This is the patient's clinical image. Please provide a single, most likely diagnosis based on this image. Don't write any further information.")]}
        ]
        messages.append(image_message_1)

        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_1 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_1 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_1]})
        res1 = f"Current diagnosis: {diagnosis_1}"
        print(res1)

        image_message_2 = {
            "role": "user",
            "parts": [dermoscopy_img, ("This is the patient's dermoscopic image. Please provide a single, most likely diagnosis by combining the provided clinical image and dermoscopic image. Don't write any further information.")]
        }
        messages.append(image_message_2)
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_2 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_2 = re.sub("[\r\n]", "", str(sys.exc_info()))

        messages.append({'role':'model',
                        'parts':[diagnosis_2]})
        res2 = f"Current diagnosis: {diagnosis_2}"
        print(res2)

        messages.append({
            "role": "user",
            "parts": [f"This is the patient's medical history: {medical_hist}. Please provide a single, most likely diagnosis by combining the provided clinical image, dermoscopic image, and medical history. Don't write any further information."]
        })
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_3 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_3 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_3]})
        res3 = f"Final diagnosis: {diagnosis_3}"
        print(res3)

        res.append("[%d]%s" % (i, res1+res2+res3))
        print("[%d]%s" % (i, res1+res2+res3))
        
    result = ' '.join([str(item)for item in res])
    return result

def run_sequence_cmd_en(model, clinical_img, dermoscopy_img, medical_hist):
    
    clinical_img = Image.open(clinical_img)
    dermoscopy_img = Image.open(dermoscopy_img)
    
    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_en}]}
        ]
                
        image_message_1 = [
            {"role": "user",
            "parts": [clinical_img, ("This is the patient's clinical image. Please provide a single, most likely diagnosis based on this image. Don't write any further information.")]}
        ]
        messages.append(image_message_1)
        
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_1 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_1 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_1]})
        res1 = f"Current diagnosis: {diagnosis_1}"
        print(res1)

        messages.append({
            "role": "user",
            "parts": [f"This is the patient's medical history: {medical_hist}. Please provide a single, most likely diagnosis by combining the provided clinical image and medical history. Don't write any further information."]
        })
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_2 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_2 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_2]})
        res2 = f"Current diagnosis: {diagnosis_2}"
        print(res2)

        image_message_2 = {
            "role": "user",
            "parts": [dermoscopy_img, ("This is the patient's dermoscopic image. Please provide a single, most likely diagnosis by combining the provided clinical image, medical history, and dermoscopic image. Don't write any further information.")]
        }
        messages.append(image_message_2)
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_3 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_3 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_3]})
        res3 = f"Final diagnosis: {diagnosis_3}"
        print(res3)

        res.append("[%d]%s" % (i, res1+res2+res3))
        print("[%d]%s" % (i, res1+res2+res3))
        
    result = ' '.join([str(item)for item in res])
    return result

def run_sequence_dcm_en(model, clinical_img, dermoscopy_img, medical_hist):
    
    clinical_img = Image.open(clinical_img)
    dermoscopy_img = Image.open(dermoscopy_img)

    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_en}]}
        ]
                
        image_message_1 = [
            {"role": "user",
            "parts": [dermoscopy_img, ("This is the patient's dermoscopic image. Please provide a single, most likely diagnosis based on this image. Don't write any further information.")]}
        ]
        messages.append(image_message_1)

        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_1 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_1 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_1]})
        res1 = f"Current diagnosis: {diagnosis_1}"
        print(res1)

        image_message_2 = {
            "role": "user",
            "parts": [clinical_img, ("This is the patient's clinical image. Please provide a single, most likely diagnosis by combining the provided dermoscopic image and clinical image. Don't write any further information.")]
        }
        messages.append(image_message_2)
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_2 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_2 = re.sub("[\r\n]", "", str(sys.exc_info()))

        messages.append({'role':'model',
                        'parts':[diagnosis_2]})
        res2 = f"Current diagnosis: {diagnosis_2}"
        print(res2)

        messages.append({
            "role": "user",
            "parts": [f"This is the patient's medical history: {medical_hist}. Please provide a single, most likely diagnosis by combining the provided dermoscopic image, clinical image, and medical history. Don't write any further information."]
        })
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_3 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_3 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_3]})
        res3 = f"Final diagnosis: {diagnosis_3}"
        print(res3)

        res.append("[%d]%s" % (i, res1+res2+res3))
        print("[%d]%s" % (i, res1+res2+res3))
        
    result = ' '.join([str(item)for item in res])
    return result

def run_sequence_dmc_en(model, clinical_img, dermoscopy_img, medical_hist):
    
    clinical_img = Image.open(clinical_img)
    dermoscopy_img = Image.open(dermoscopy_img)
    
    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_en}]}
        ]
                
        image_message_1 = [
            {"role": "user",
            "parts": [dermoscopy_img, ("This is the patient's dermoscopic image. Please provide a single, most likely diagnosis based on this image. Don't write any further information.")]}
        ]
        messages.append(image_message_1)
        
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_1 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_1 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_1]})
        res1 = f"Current diagnosis: {diagnosis_1}"
        print(res1)

        messages.append({
            "role": "user",
            "parts": [f"This is the patient's medical history: {medical_hist}. Please provide a single, most likely diagnosis by combining the provided dermoscopic image and medical history. Don't write any further information."]
        })
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_2 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_2 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_2]})
        res2 = f"Current diagnosis: {diagnosis_2}"
        print(res2)

        image_message_2 = {
            "role": "user",
            "parts": [clinical_img, ("This is the patient's clinical image. Please provide a single, most likely diagnosis by combining the provided dermoscopic image, medical history, and clinical image. Don't write any further information.")]
        }
        messages.append(image_message_2)
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_3 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_3 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_3]})
        res3 = f"Final diagnosis: {diagnosis_3}"
        print(res3)

        res.append("[%d]%s" % (i, res1+res2+res3))
        print("[%d]%s" % (i, res1+res2+res3))
        
    result = ' '.join([str(item)for item in res])
    return result

def run_sequence_mcd_en(model, clinical_img, dermoscopy_img, medical_hist):
    
    clinical_img = Image.open(clinical_img)
    dermoscopy_img = Image.open(dermoscopy_img)
    
    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_en}]}
        ]

        message_1 = [
            {"role": "user",
            "parts": [f"This is the patient's medical history: {medical_hist}. Please provide a single, most likely diagnosis based on this medical history. Don't write any further information."]}
        ]
        messages.append(message_1)

        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_1 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_1 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_1]})
        res1 = f"Current diagnosis: {diagnosis_1}"
        print(res1)

        image_message_1 = {
            "role": "user",
            "parts": [clinical_img, ("This is the patient's clinical image. Please provide a single, most likely diagnosis by combining the provided medical history and clinical image. Don't write any further information.")]
        }
        messages.append(image_message_1)
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_2 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_2 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_2]})
        res2 = f"Current diagnosis: {diagnosis_2}"
        print(res2)

        image_message_2 = {
            "role": "user",
            "parts": [dermoscopy_img, ("This is the patient's dermoscopic image. Please provide a single, most likely diagnosis by combining the provided medical history, clinical image, and dermoscopic image. Don't write any further information.")]
        }
        messages.append(image_message_2)
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_3 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_3 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_3]})
        res3 = f"Final diagnosis: {diagnosis_3}"
        print(res3)

        res.append("[%d]%s" % (i, res1+res2+res3))
        print("[%d]%s" % (i, res1+res2+res3))
        
    result = ' '.join([str(item)for item in res])
    return result

def run_sequence_mdc_en(model, clinical_img, dermoscopy_img, medical_hist):
    
    clinical_img = Image.open(clinical_img)
    dermoscopy_img = Image.open(dermoscopy_img)
    
    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_en}]}
        ]

        message_1 = [
            {"role": "user",
            "parts": [f"This is the patient's medical history: {medical_hist}. Please provide a single, most likely diagnosis based on this medical history. Don't write any further information."]}
        ]
        messages.append(message_1)
        
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_1 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_1 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_1]})
        res1 = f"Current diagnosis: {diagnosis_1}"
        print(res1)

        image_message_1 = {
            "role": "user",
            "parts": [dermoscopy_img, ("This is the patient's dermoscopic image. Please provide a single, most likely diagnosis by combining the provided medical history and dermoscopic image. Don't write any further information.")]
        }
        messages.append(image_message_1)
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_2 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_2 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_2]})
        res2 = f"Current diagnosis: {diagnosis_2}"
        print(res2)

        image_message_2 = {
            "role": "user",
            "parts": [clinical_img, ("This is the patient's clinical image. Please provide a single, most likely diagnosis by combining the provided medical history, dermoscopic image, and clinical image. Don't write any further information.")]
        }
        messages.append(image_message_2)
        while True:
            try:
                response = dashscope.MultiModalConversation.call(api_key=api_keys, model=model,  messages=messages)                
                break
            except Exception as e:
                print(f"    %s，retry------------------------------------"% e)
                time.sleep(1)
        try:
            diagnosis_3 = re.sub("[\r\n]", "", str(response.text))
        except Exception as e:
            diagnosis_3 = re.sub("[\r\n]", "", str(sys.exc_info()))
        messages.append({'role':'model',
                        'parts':[diagnosis_3]})
        res3 = f"Final diagnosis: {diagnosis_3}"
        print(res3)

        res.append("[%d]%s" % (i, res1+res2+res3))
        print("[%d]%s" % (i, res1+res2+res3))
        
    result = ' '.join([str(item)for item in res])
    return result

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取前10KB数据用于检测
    result = chardet.detect(raw_data)
    return result['encoding']

if __name__ == "__main__":
    TEST_TIMES = 10

    TEST_DATASET = "Private_dataset"        # "Private_dataset" or "Derm7pt_dataset"
    TEST_SEQUENCE = "C-D-M"                 # "C-D", "C-M", "C-D-M", "C-M-D", "D-C-M", "D-M-C", "M-C-D", or "M-D-C"
    LANGUAGE = "en"                         # "en" (gemini and llama32) or "zh" (qwenvlmax and deepseekvl)
    TXT_FILE = f'results/multimodal_llama32_{TEST_DATASET}_{TEST_SEQUENCE}.txt'
    DATASET_FILE = f'private_mmdata_en.txt'

    file_paths_clinical = []
    correspond_paths_dermoscopy = []
    correspond_paths_medical = []
    
    with open(DATASET_FILE, 'r', encoding=detect_file_encoding(DATASET_FILE), errors='replace') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            clinical_path = line.split("---")[0].strip()
            dermoscopy_path = line.split("---")[1].strip()
            medical_path = line.split("---")[2].strip()

            if len(TEST_SEQUENCE.split("-")) == 3:
                if clinical_path and dermoscopy_path and medical_path:
                    file_paths_clinical.append(clinical_path)
                    correspond_paths_dermoscopy.append(dermoscopy_path)
                    correspond_paths_medical.append(medical_path)
            elif TEST_SEQUENCE == "C-D":
                if clinical_path and dermoscopy_path:
                    file_paths_clinical.append(clinical_path)
                    correspond_paths_dermoscopy.append(dermoscopy_path)
            elif TEST_SEQUENCE == "C-M":
                if clinical_path and medical_path:
                    file_paths_clinical.append(clinical_path)
                    correspond_paths_medical.append(medical_path)
                
    print(f"Test on the {TEST_DATASET}, finding a total of {len(file_paths_clinical)} test samples")
  
    img_cnt = 0
    
    for i in range(len(file_paths_clinical)):
        clinical_path = file_paths_clinical[i]
        dermoscopy_path = correspond_paths_dermoscopy[i]
        medical_history = correspond_paths_medical[i]
        
        img_cnt += 1
        
        print("-------------Processing [%d]"%img_cnt, clinical_path, dermoscopy_path, medical_history)

        api_keys = 'sk-cc842e903f4b4eb78bdfe349418e1a66'  # Replace with your API key of Gemini
        system_message_en = """You are a medical artificial intelligence assistant. You can diagnose patients based on the provided information. Your goal is to correctly diagnose the patient. Based on the provided information you will provide the main diagnosis. Don't write any further information. Give only a single diagnosis."""

        model = 'llama3.2-11b-vision'

        if TEST_SEQUENCE == "C-D":
            result = run_sequence_cd_en(model, clinical_path, dermoscopy_path)
        elif TEST_SEQUENCE == "C-M":
            result = run_sequence_cm_en(model, clinical_path, medical_history)
        elif TEST_SEQUENCE == "C-D-M":
            result = run_sequence_cdm_en(model, clinical_path, dermoscopy_path, medical_history)
        elif TEST_SEQUENCE == "C-M-D":
            result = run_sequence_cmd_en(model, clinical_path, dermoscopy_path, medical_history)
        elif TEST_SEQUENCE == "D-C-M":
            result = run_sequence_dcm_en(model, clinical_path, dermoscopy_path, medical_history)
        elif TEST_SEQUENCE == "D-M-C":
            result = run_sequence_dmc_en(model, clinical_path, dermoscopy_path, medical_history)
        elif TEST_SEQUENCE == "M-C-D":
            result = run_sequence_mcd_en(model, clinical_path, dermoscopy_path, medical_history)
        elif TEST_SEQUENCE == "M-D-C":
            result = run_sequence_mdc_en(model, clinical_path, dermoscopy_path, medical_history)


        new_content = "%d. %s %s %s %s\n" % (img_cnt, clinical_path, dermoscopy_path, medical_history, result)
        print()
        with open(TXT_FILE, 'a', encoding='utf-8') as file:
            file.writelines(new_content)
    