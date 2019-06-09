# satellite_segmentaion



#  Introduce: 

### My final thsis is about satellite segmentaion.  I need to extract road、builidings、water or plants from the images. 

#  About the dataset:

### The dataset must conation images and masks. The dataset is shared below about five kinds of objects. 

baidu cloud：https://pan.baidu.com/s/1TcQuMAY2aEiVrFHJd5YIAA 
code：vfps

google dirve：https://drive.google.com/open?id=1gE6WeoSiXTPEr-mYH8uggXiJEISgBHYZ 

#  About the mothod:

### 	I use three models to predict the result. Unet、PSPnet and Segnet. And then I vote for a final result by combining the models' result. Then I do some post_processing for voted images. And then there can be some little improvement.



### 	![image](https://github.com/wuchangsheng951/satellite_segmentaion/tree/master/images/1.png)





#  About the Result:

### 	I just extract road from the models for now. So now I just show the Iou result about the road. I train them for 50 epochs and learning rate is1e-4. Adam optimizer. Here are the result.

| Model_name | Train_miou  | Val_miou    | Train loss  | Loss loss   |
| ---------- | ----------- | ----------- | ----------- | ----------- |
| U-net      | 0.87640     | 0.74904     | 0.07287     | 0.10107     |
| Seg-net    | **0.90533** | 0**.78413** | 0.**06660** | **0.09993** |
| Psp-net    | 0.90404     | *0.75175*   | *0.06693*   | *0.11606*   |

### ![image](https://github.com/wuchangsheng951/satellite_segmentaion/tree/master/images/2.png)



###  	And here is the performance in the different validation area which contaion a large area. As we can see the performance after voting can imporve a lot.

| Model  | Valid_1   （city）% | Valid_2   （city）% | Valid_3   （city）% | Valid_4   （town）% | Valid_5   （town）% |
| ------ | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| Unet   | `0.72285`           | `0.74068`           | `0.73932`           | `0.65743`           | `0.31698            |
| Segnet | `0.75509`           | `0.76256`           | `0.73416`           | `0.65141`           | `0.22854`           |
| Pspnet | `0.75183`           | `0.75508`           | `0.71053`           | `0.62960`           | `0.27243`           |
| Vote   | **0.77340**         | 0.78339*            | 0.75360*            | 0.67627             | `0.28731`           |

![image](https://github.com/wuchangsheng951/satellite_segmentaion/tree/master/images/3.png)



#  	If you want use this code. you can just clone it . I will add the requirments.txt soon.

### 





##  First step:  prepare you own dataset:

###  	You need to give the right path for images and  masks in dataset_processing.py

```python
image_path = Path('./BDCI2017-seg/CCF-training-Semi')/f'{i}.png'
img_class_path = Path('./BDCI2017-seg/CCF-training-Semi')/ f'{i}_class_vis.png'
```

### Second step: train

 ###  	give the path of dataset and alter the params you want to use.

```
device = 'cuda'
path = '/home/shiyi/beshe/kinds_dataset/'
learning_rate = 5e-3
num_epochs = 50
num_classes = 5
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
```

###  Third step:  Prediction

###  	The code in predict.py maybe hard to understand.  You can just use it. You need use the model you trained in step two.

```
model = torch.load(f'model/0514pspnet_50_epoch.pth')
# give the picture you want to predict
file_name = f'/home/shiyi/beshe/gaoxin_map/second_dataset/part1_500.png'
# give the name you want to store
save_dir = '0514predict1.png' 
```

###  Fourth Step: vote and post_processing

###  	The post_processing progress can be interesting. You can different kinds of combanation.

```
# you can just run 
python post_deal.py
```

​	



