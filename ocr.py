import pytesseract
from pytesseract import Output
from PIL import Image
import pandas as pd
import cv2
import numpy as np
import os
import fastwer
import time
from code_union import code_correct
# get grayscale image

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,3)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

def thresholding_test(image):
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    if thresh <= 145:
        return cv2.threshold(image, thresh - 15, 255, cv2.THRESH_BINARY_INV)
    else:
        return cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    

def new_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11 ,0)

#dilation 팽창 (글씨가 더 팽창)
def dilate(image, iter):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = iter)
    
#erosion 침식 (글씨가 더 얇아짐)
def erode(image, iter):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = iter)

#opening - erosion followed by dilation ( 침식 후 팽창)
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    image = dilate(erode(image,3),1)
    return image
    # return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def closing(image): #팽창 후 침식
    kernel = np.ones((5,5),np.uint8)
    image = erode(dilate(image,3),1)
    return image
    # return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction 기울시 수정
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

def resize(image):
    width = image.shape[1]
    height = image.shape[0]
    por = height / width
    size = (10000, round(10000 * por))
    image = cv2.resize(image,  dsize=size, interpolation=cv2.INTER_LINEAR)
    return image
    
def return_smallsize(image):
    width = image.shape[1]
    height = image.shape[0]
    por = height / width
    size = (2048, round(2048 * por))
    image = cv2.resize(image,  dsize=size, interpolation=cv2.INTER_LINEAR)
    return image
    
def img_Contrast(img): 
    #-----Converting image to LAB Color model----------------------------------- 
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) 
    # -----Splitting the LAB image to different channels------------------------- 
    l, a, b = cv2.split(lab) 
    # -----Applying CLAHE to L-channel------------------------------------------- 
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) 
    cl = clahe.apply(l) 
    # -----Merge the CLAHE enhanced L-channel with the a and b channel----------- 
    limg = cv2.merge((cl, a, b)) 
    # -----Converting image from LAB Color model to RGB model-------------------- 
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) 
    return final
def blur(img):
    return cv2.bilateralFilter(img, 5, 75, 75)


def save_img(image, folder_name, file_name): # 변환된 이미지 저장을 위해
    path = os.getcwd()
    new_dir = os.path.join(path, folder_name)
    # print(new_dir)
    if os.path.isdir(new_dir):
        pass
    else:
        os.mkdir(new_dir)

    new_path = folder_name + '/' + file_name
    new_path = os.path.join(path, new_path)
    # print(new_path)
    cv2.imwrite(new_path, image)

def save_txt(text, folder_name, file_name): # ocr 결과 저장 
    path = os.getcwd()
    new_dir = os.path.join(path, folder_name)
    # print(new_dir)
    if os.path.isdir(new_dir):
        pass
    else:
        os.mkdir(new_dir)

    new_path = folder_name + '/' + file_name
    new_path = os.path.join(path, new_path)
    # print(new_path)
    file = open(new_path, 'w', encoding='utf8')
    file.write(text)
    file.close()

def get_text_ffile(filename):
    text = ''
    file = open(filename, mode= 'r', encoding="utf-8")
    lines= file.readlines()
    for i in range(len(lines)):
        text += lines[i]
    file.close()
    return text

def ocr(image, lang): # ocr
    whitelist_op = '''-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;-_+=\\'\\"<>(){}[]~`!@#$%^&*?|/\\ '''
    # blacklist_op = '''-c tessedit_char_blacklist=¥®©¢¢*'''
    custom_config = r'-c preserve_interword_spaces=1 '+  whitelist_op + ' --oem 3 --psm 6 -l ' + lang
    d = pytesseract.image_to_data(image, config=custom_config, output_type=Output.DICT)
    df = pd.DataFrame(d)
    result_text = ''
    # clean up blanks
    df1 = df[(df.conf!='-1')&(df.text!=' ')&(df.text!='')]
    # sort blocks vertically
    sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
    for block in sorted_blocks:
        curr = df1[df1['block_num']==block]
        sel = curr[curr.text.str.len()>3]
        char_w = (sel.width/sel.text.str.len()).mean()
        prev_par, prev_line, prev_left = 0, 0, 0
        text = ''
        for ix, ln in curr.iterrows():
            # add new line when necessary
            if prev_par != ln['par_num']:
                text += '\n'
                prev_par = ln['par_num']
                prev_line = ln['line_num']
                prev_left = 0
            elif prev_line != ln['line_num']:
                text += '\n'
                prev_line = ln['line_num']
                prev_left = 0

            added = 0  # num of spaces that should be added
            if ln['left']/char_w > prev_left + 1:
                added = int((ln['left'])/char_w) - prev_left
                text += ' ' * added 
            text += ln['text'] + ' '
            prev_left += len(ln['text']) + added + 1
        text += '\n'
        # print(text)
        result_text += text
    return result_text


def error_rate(name, text, ref):
    print(name + " : ")
    cer = fastwer.score_sent(text,ref, char_level = True) # character error rate
    wer = fastwer.score_sent(text,ref, char_level = False) # word error rate

    return cer, wer

def blur_and_gray(image): # blur + gray
    return get_grayscale(blur(image))

def blur_gray_threshold(image):
    return thresholding(blur_and_gray(image))

def opening_b_g_th(image):
    return opening(blur_gray_threshold(image))

def closing_b_g_th(image):
    return closing(blur_gray_threshold(image))

# test 부분
# for i in range(1,12):
#     start = time.time()
#     img_name = 'ex'+ str(i) # image file name
#     img = cv2.imread(img_name + '.JPG') # 이미지 불러오고
#     # 이미지 처리
#     # for i in range(n): # 이미지 처리 n 개 case
#     img = resize(img)
#     img = blur_gray_threshold(img)
#     #ocr
#     result = ocr(img, 'eng_b+kor_b')
#     ref = get_text_ffile(img_name+'_ref.txt')    
#     wer, cer = error_rate(img_name, result, ref)
#     # 이미지 저장
#     save_img(img, 'blur_gray_threshold', img_name + '.JPG')
#     # ocr 텍스트 저장
#     save_txt(result, 'blur_gray_threshold', img_name + '_ocr.txt')
#     # wer, cer 도 이미지변환별로 list에 잘 넣어서 종합 txt로 만들기.
#     print(f'wer : {wer}  cer : {cer}')
#     print("time : ", time.time() - start) 
    

foldername = './test' #폴더명을 인자로 받아서
newfolder = './result_resize_thresh'
result = ''
temp = []
if not os.path.exists(newfolder):
        os.makedirs(newfolder)
frame_count = len(next(os.walk(foldername))[2])
start = time.time()
for i in range(0,frame_count):
    
    filename = foldername +'/frame' + str(i) + '.png'
    img = cv2.imread(filename) # 이미지 불러오고
    # img = get_grayscale(img)
    # img = thresholding_test(img)
    img = resize(img)
    img = get_grayscale(img)
    ret,img = thresholding_test(img)
    # print(ret,end=' ')
    # # if ret <= 140:
    # #     img = closing(img)
    # # else:
    #     # img = opening(img)
    img = opening(img)
    img = return_smallsize(img)
    # img = closing_b_g_th(img)
    temp_result = ocr(img, 'eng_b')
    temp.append(temp_result)
    br = '\n-------------------------------'+str(i)+'-----------------------------------\n'
    result = result + br + temp_result
    res_file = newfolder +'/frame' + str(i) + '.png'
    cv2.imwrite(res_file, img)
code_corrector = code_correct(temp)
code_corrector.ocr_code_union()
temp_result = code_corrector.get_result()
result += temp_result
print()
print(result, end='')
print(f'time : {time.time() - start}')
save_txt(result, newfolder, 'ocr_result.txt')





# img = blur(resize(img))
# cv2.imshow('test', img_r)
# cv2.waitKey(0)   # 5초 후에 자동 close, 빈 칸이면 아무 키 누르면 close
# cv2.destroyAllWindows()