import os
from sklearn.model_selection import train_test_split

txt = open('F:\\数字图像处理\\DIP 2\\images_labels.txt', 'w')
path = 'F:\\数字图像处理\\DIP 2\\dataset'
for root, dirs, files in os.walk(path):
    if len(dirs) == 0:
        label = os.path.split(root)[1]
        for f in files:
            txt.write(os.path.join(label, f)+' '+str(int(label)-102)+'\n')
txt.close()

txt = open('F:\\数字图像处理\\DIP 2\\images_labels.txt')
lines = txt.readlines()
txt.close()
x = [line.strip().split()[0] for line in lines]
y = [line.strip().split()[1] for line in lines]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)

train = open('F:\\数字图像处理\\DIP 2\\images_labels_train.txt', 'w')
for i, xi in enumerate(x_train):
    train.write(xi+' '+y_train[i]+'\n')
train.close()
test = open('F:\\数字图像处理\\DIP 2\\images_labels_test.txt', 'w')
for i, xi in enumerate(x_test):
    test.write(xi+' '+y_test[i]+'\n')
test.close()