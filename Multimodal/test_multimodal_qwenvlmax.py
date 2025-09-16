import os
import re
import sys
import time
from PIL import Image
import chardet
import dashscope


def run_sequence_cd_zh(model, clinical_img, dermoscopy_img):
    
    clinical_img = Image.open(clinical_img)
    dermoscopy_img = Image.open(dermoscopy_img)

    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_zh}]}
        ]
                
        image_message_1 = [
            {"role": "user",
            "parts": [clinical_img, ("这是患者的临床影像。请根据该影像提供最可能的诊断。")]}
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
        res1 = f"诊断：{diagnosis_1}"
        print(res1)

        image_message_2 = {
            "role": "user",
            "parts": [dermoscopy_img, ("这是患者的皮肤镜影像。请结合提供的临床影像和皮肤镜影像，给出最可能的最终诊断。")]
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
        res2 = f"最终诊断：{diagnosis_2}"
        print(res2)

        res.append("[%d]%s" % (i, res1+res2))
        print("[%d]%s" % (i, res1+res2))
        
    result = ' '.join([str(item)for item in res])
    return result

def run_sequence_cm_zh(model, clinical_img, medical_hist):
    
    clinical_img = Image.open(clinical_img)
    
    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_zh}]}
        ]
                
        image_message_1 = [
            {"role": "user",
            "parts": [clinical_img, ("这是患者的临床影像。请根据该影像提供最可能的诊断。")]}
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
        res1 = f"诊断：{diagnosis_1}"
        print(res1)

        messages.append({
            "role": "user",
            "parts": [f"这是患者的病史：{medical_hist}。请结合提供的临床影像和病史，给出最可能的最终诊断。"]
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
        res2 = f"最终诊断：{diagnosis_2}"
        print(res2)

        res.append("[%d]%s" % (i, res1+res2))
        print("[%d]%s" % (i, res1+res2))
        
    result = ' '.join([str(item)for item in res])
    return result

def run_sequence_cdm_zh(model, clinical_img, dermoscopy_img, medical_hist):
    
    clinical_img = Image.open(clinical_img)
    dermoscopy_img = Image.open(dermoscopy_img)

    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_zh}]}
        ]
                
        image_message_1 = [
            {"role": "user",
            "parts": [clinical_img, ("这是患者的临床影像。请根据该影像提供最可能的诊断。")]}
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
        res1 = f"诊断：{diagnosis_1}"
        print(res1)

        image_message_2 = {
            "role": "user",
            "parts": [dermoscopy_img, ("这是患者的皮肤镜影像。请结合提供的临床影像和皮肤镜影像，给出最可能的诊断。")]
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
        res2 = f"诊断：{diagnosis_2}"
        print(res2)

        messages.append({
            "role": "user",
            "parts": [f"这是患者的病史：{medical_hist}。请结合提供的临床影像、皮肤镜影像和病史，给出最可能的最终诊断。"]
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
        res3 = f"最终诊断：{diagnosis_3}"
        print(res3)

        res.append("[%d]%s" % (i, res1+res2+res3))
        print("[%d]%s" % (i, res1+res2+res3))
        
    result = ' '.join([str(item)for item in res])
    return result

def run_sequence_cmd_zh(model, clinical_img, dermoscopy_img, medical_hist):
    
    clinical_img = Image.open(clinical_img)
    dermoscopy_img = Image.open(dermoscopy_img)
    
    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_zh}]}
        ]
                
        image_message_1 = [
            {"role": "user",
            "parts": [clinical_img, ("这是患者的临床影像。请根据该影像提供最可能的诊断。")]}
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
        res1 = f"诊断：{diagnosis_1}"
        print(res1)

        messages.append({
            "role": "user",
            "parts": [f"这是患者的病史：{medical_hist}。请结合提供的临床影像和病史，给出最可能的诊断。"]
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
        res2 = f"诊断：{diagnosis_2}"
        print(res2)

        image_message_2 = {
            "role": "user",
            "parts": [dermoscopy_img, ("这是患者的皮肤镜影像。请结合提供的临床影像、病史和皮肤镜影像，给出最可能的最终诊断。")]
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
        res3 = f"最终诊断：{diagnosis_3}"
        print(res3)

        res.append("[%d]%s" % (i, res1+res2+res3))
        print("[%d]%s" % (i, res1+res2+res3))
        
    result = ' '.join([str(item)for item in res])
    return result

def run_sequence_dcm_zh(model, clinical_img, dermoscopy_img, medical_hist):
    
    clinical_img = Image.open(clinical_img)
    dermoscopy_img = Image.open(dermoscopy_img)

    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_zh}]}
        ]
                
        image_message_1 = [
            {"role": "user",
            "parts": [dermoscopy_img, ("这是患者的皮肤镜影像。请根据该影像提供最可能的诊断。")]}
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
        res1 = f"诊断：{diagnosis_1}"
        print(res1)

        image_message_2 = {
            "role": "user",
            "parts": [clinical_img, ("这是患者的临床影像。请结合提供的皮肤镜影像和临床影像，给出最可能的诊断。")]
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
        res2 = f"诊断：{diagnosis_2}"
        print(res2)

        messages.append({
            "role": "user",
            "parts": [f"这是患者的病史：{medical_hist}。请结合提供的皮肤镜影像、临床影像和病史，给出最可能的最终诊断。"]
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
        res3 = f"最终诊断：{diagnosis_3}"
        print(res3)

        res.append("[%d]%s" % (i, res1+res2+res3))
        print("[%d]%s" % (i, res1+res2+res3))
        
    result = ' '.join([str(item)for item in res])
    return result

def run_sequence_dmc_zh(model, clinical_img, dermoscopy_img, medical_hist):
    
    clinical_img = Image.open(clinical_img)
    dermoscopy_img = Image.open(dermoscopy_img)
    
    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_zh}]}
        ]
                
        image_message_1 = [
            {"role": "user",
            "parts": [dermoscopy_img, ("这是患者的皮肤镜影像。请根据该影像提供最可能的诊断。")]}
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
        res1 = f"诊断：{diagnosis_1}"
        print(res1)

        messages.append({
            "role": "user",
            "parts": [f"这是患者的病史：{medical_hist}。请结合提供的皮肤镜影像和病史，给出最可能的诊断。"]
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
        res2 = f"诊断：{diagnosis_2}"
        print(res2)

        image_message_2 = {
            "role": "user",
            "parts": [clinical_img, ("这是患者的临床影像。请结合提供的皮肤镜影像、病史和临床影像，给出最可能的最终诊断。")]
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
        res3 = f"最终诊断：{diagnosis_3}"
        print(res3)

        res.append("[%d]%s" % (i, res1+res2+res3))
        print("[%d]%s" % (i, res1+res2+res3))
        
    result = ' '.join([str(item)for item in res])
    return result

def run_sequence_mcd_zh(model, clinical_img, dermoscopy_img, medical_hist):
    
    clinical_img = Image.open(clinical_img)
    dermoscopy_img = Image.open(dermoscopy_img)
    
    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_zh}]}
        ]

        message_1 = [
            {"role": "user",
            "parts": [f"这是患者的病史：{medical_hist}。请根据该病史提供最可能的诊断。"]}
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
        res1 = f"诊断：{diagnosis_1}"
        print(res1)

        image_message_1 = {
            "role": "user",
            "parts": [clinical_img, ("这是患者的临床影像。请结合提供的病史和临床影像，给出最可能的诊断。")]
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
        res2 = f"诊断：{diagnosis_2}"
        print(res2)

        image_message_2 = {
            "role": "user",
            "parts": [dermoscopy_img, ("这是患者的皮肤镜影像。请结合提供的病史、临床影像和皮肤镜影像，给出最可能的最终诊断。")]
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
        res3 = f"最终诊断：{diagnosis_3}"
        print(res3)

        res.append("[%d]%s" % (i, res1+res2+res3))
        print("[%d]%s" % (i, res1+res2+res3))
        
    result = ' '.join([str(item)for item in res])
    return result

def run_sequence_mdc_zh(model, clinical_img, dermoscopy_img, medical_hist):
    
    clinical_img = Image.open(clinical_img)
    dermoscopy_img = Image.open(dermoscopy_img)
    
    res = []

    for i in range(TEST_TIMES):
        messages = [
            {'role': 'system', 'content': [{'text': system_message_zh}]}
        ]

        message_1 = [
            {"role": "user",
            "parts": [f"这是患者的病史：{medical_hist}。请根据该病史提供最可能的诊断。"]}
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
        res1 = f"诊断：{diagnosis_1}"
        print(res1)

        image_message_1 = {
            "role": "user",
            "parts": [dermoscopy_img, ("这是患者的皮肤镜影像。请结合提供的病史和皮肤镜影像，给出最可能的诊断。")]
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
        res2 = f"诊断：{diagnosis_2}"
        print(res2)

        image_message_2 = {
            "role": "user",
            "parts": [clinical_img, ("这是患者的临床影像。请结合提供的病史、皮肤镜影像和临床影像，给出最可能的最终诊断。")]
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
        res3 = f"最终诊断：{diagnosis_3}"
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
    LANGUAGE = "zh"                         # "en" (gemini and llama32) or "zh" (qwenvlmax and deepseekvl)
    TXT_FILE = f'results/multimodal_qwenvlmax_{TEST_DATASET}_{TEST_SEQUENCE}.txt'
    DATASET_FILE = f'private_mmdata_zh.txt'

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

        api_keys = 'sk-cc842e903f4b4eb78bdfe349418e1a66'  # Replace with your API key of Qwen-VL-Max
        system_message_zh = """您是一名医疗人工智能助手。您可以根据提供的信息为患者诊断。您的目标是通过多轮对话为同一位患者准确诊断。请记住并整合此前所有对话中的信息，然后给出主要诊断。无需附加其他信息，仅提供一个诊断结果。"""

        model = 'qwen-vl-max'

        if TEST_SEQUENCE == "C-D":
            result = run_sequence_cd_zh(model, clinical_path, dermoscopy_path)
        elif TEST_SEQUENCE == "C-M":
            result = run_sequence_cm_zh(model, clinical_path, medical_history)
        elif TEST_SEQUENCE == "C-D-M":
            result = run_sequence_cdm_zh(model, clinical_path, dermoscopy_path, medical_history)
        elif TEST_SEQUENCE == "C-M-D":
            result = run_sequence_cmd_zh(model, clinical_path, dermoscopy_path, medical_history)
        elif TEST_SEQUENCE == "D-C-M":
            result = run_sequence_dcm_zh(model, clinical_path, dermoscopy_path, medical_history)
        elif TEST_SEQUENCE == "D-M-C":
            result = run_sequence_dmc_zh(model, clinical_path, dermoscopy_path, medical_history)
        elif TEST_SEQUENCE == "M-C-D":
            result = run_sequence_mcd_zh(model, clinical_path, dermoscopy_path, medical_history)
        elif TEST_SEQUENCE == "M-D-C":
            result = run_sequence_mdc_zh(model, clinical_path, dermoscopy_path, medical_history)

        new_content = "%d. %s %s %s %s\n" % (img_cnt, clinical_path, dermoscopy_path, medical_history, result)
        print()
        with open(TXT_FILE, 'a', encoding='utf-8') as file:
            file.writelines(new_content)
    