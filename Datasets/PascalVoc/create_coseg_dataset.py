import glob
import os
import cv2
import numpy as np

labelfiles=glob.glob("*.png")

def get_cat_from_label(label):
    cat=[]
    for i in range(len(label)):
        for j in range(len(label[0])):
            if label[i][j] not in cat and label[i][j]!=0:
                cat.append(label[i][j])
    return cat


cats=[]
count=0
for labelfile in labelfiles:
    cats.append(get_cat_from_label(cv2.imread(labelfile,0)))
    count+=1
    print("{}/{}".format(count,len(labelfiles)))

def turn_label_01(label,cat):
    for i in range(len(label)):
        for j in range(len(label[0])):
            if label[i][j] not in cat:
                label[i][j]=0
            else:
                label[i][j]=255
    return label


pairs_num=0

def get_similar_cat(cat1,cat2):
    global pairs_num
    same=false
    similar_cat=[]
    for cat in cat1:
        if cat in cat2:
            same=True
            similar_cat.append(cat)
    pairs_num+=1
    return similar_cat





def save_coseg(cat1,cat2,l1,l2,name1,name2):
    similar_cat=get_similar_cat(cat1,cat2)
    if len(similar_cat)==0:
        return
    colabel1=turn_label_01(l1,similar_cat)
    colabel2=turn_label_01(l2,similar_cat)
    cv2.imwrite("../../colabel/train/{}_{}.png".format(name1,name2),colabel1)
    cv2.imwrite("../../colabel/train/{}_{}.png".format(name2,name1),colabel2)
    f=open("../../colabel/train.txt","a")
    f.write("{} {} {} {}\n".format(name1,name2,name1+"_"+name2,name2+"_"+name1))
    f.close()


count=0
for index1 in range(len(labelfiles)):
    count+=1
    print("{}/{}".format(count,len(labelfiles)))
    for index2 in range(index1+1,len(labelfiles)):
        get_similar_cat(cats[index1],cats[index2])
        save_coseg(cats[index1],cats[index2],cv2.imread(labelfiles[index1],0),cv2.imread(labelfiles[index2],0),labelfiles[index1][:-4],labelfiles[index2][:-4])

print("pair numbers : ",pairs_num)

