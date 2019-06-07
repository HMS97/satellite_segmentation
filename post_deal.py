import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from PIL import Image
from path import Path
from utils.core import denoise, grow, erode_and_dilate, dilate_and_erode
# 形态学处理
def post_deal(temp):
    kernel_size_denoise = 20
    kernel_size_grow = 20
    simplify_threshold = 0.01
    denoised = denoise(temp, kernel_size_denoise)
    grown = grow(denoised, kernel_size_grow)
#     Image.fromarray(grown*255)
    return grown

def  iou(input, target,classes = 1):
    metric = []
    input = input[:target.shape[0],:target.shape[1]]
    intersection = np.logical_and(target == classes, input == classes)
    union = np.logical_or(target == classes , input == classes)
    iou_score = np.sum(intersection) / np.sum(union)
    metric.append(iou_score)
    return  iou_score

def dict_union(d1, d2):
    keys = d1.keys() | d2.keys()
    temp = {}
    for key in keys:
        temp[key] = sum([d.get(key,0) for d in (d1, d2)])
    return temp


def vote(post_deal_stage = False):
    """
    note: here can generate the images by voting
          the method is only for binary vote for now
    return : a dict consist of images
    """"

    image = {}
    model_list = {}
    for item in ['segnet','unet','pspnet']:
        model_list[item] = {}
        for i in range(9,14):
            if post_deal_stage == True:
                model_list[item][f'picture_{i}']= post_deal(cv2.imread(f'{item}_predict/0517predict{i}.png',0))
            else:
                model_list[item][f'picture_{i}']= cv2.imread(f'{item}_predict/0517predict{i}.png',0)

    models = ['segnet','unet','pspnet']
    vote_image = model_list[models[0]]
    for item in models[1:]:
        vote_image = dict_union(vote_image,model_list[item])
    
    for i in vote_image.keys():
        vote_image[i][vote_image[i] < 2] = 0
        vote_image[i][vote_image[i] >= 2] = 1
    return vote_image

def remove_small_objects(img):
    img = erode_and_dilate(img)

    labels = measure.label(img)  # 返回打上标签的img数组
    jj = measure.regionprops(labels)  # 找出连通域的各种属性。  注意，这里jj找出的连通域不包括背景连通域

    if len(jj) == 1:
        out = img
    else:
    # 通过与质心之间的距离进行判断
        num = labels.max()  #连通域的个数
        del_array = np.array([0] * (num + 1))#生成一个与连通域个数相同的空数组来记录需要删除的区域（从0开始，所以个数要加1）
        for k in range(num):#TODO：这里如果遇到全黑的图像的话会报错

            k_area = jj[k].area  # 将元组转换成array
            ## you can try to change the 600 to see the result
            if 600 < k_area:
                del_array[k + 1] = 1

        del_mask = del_array[labels]
        out = img * del_mask
        out = out.astype('uint8')
#         out = post_deal(out)
        return out
## here can generate the images by voting
vote_image = vote ()
## you can do the nosiy by using remove_small_objects
temp = remove_small_objects(vote_image[0])