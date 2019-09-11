import cv2
import os
# from matplotlib import pyplot as plt
from skimage import feature, exposure
import numpy as np
from skimage import color
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

hog_images = []
hog_features =[]

# img_path = "results/snapshots/GP010016/GP010016_right_defender"
img_path = "results/snapshots/SVM_LR"
left = np.array(("left"))
left = np.repeat(left, 990)
right = np.array(("right"))
right = np.repeat(right, 1389)
labels = np.hstack((left, right))

for img in os.listdir(img_path):
    # print(img)
    frame = cv2.imread(os.path.join(img_path, img))
    frame = cv2.resize(frame, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)
    one, two, three = cv2.split(hsl)
    new_sat = exposure.adjust_sigmoid(two, 0.75, inv=True)
    # new_img = cv2.merge([new_sat, two, three])
    canny = cv2.Canny(new_sat, 200, 255)
    new_fd, new_hog = feature.hog(canny, orientations=5, pixels_per_cell=(2, 2), block_norm="L1",
                                  cells_per_block=(3, 3), transform_sqrt=False, visualise=True,
                                  feature_vector=True)
    hog_images.append(new_hog)
    hog_features.append(new_fd)

    # np.savetxt("array_hog.txt", new_fd, fmt="%s", delimiter=";")
    # os.makedirs(img_res, exist_ok=True)
    # fullpath = os.path.join(img_res, 'hog_' + img)
    # misc.imsave(fullpath, new_hog)
    # new_hog = exposure.rescale_intensity(new_hog, in_range=(0, 20))
    # cv2.imshow("Original", frame)
    # cv2.imshow("HSL", hsl)
    # cv2.imshow("Sat", two)
    # cv2.imshow("NewSat", new_sat)
    # cv2.imshow("Canny", canny)
    # cv2.imshow("HOG", new_hog.astype("uint8") * 255)
    # cv2.imwrite(fullpath, new_hog.astype("uint8") * 255)
    # cv2.waitKey(6)
    # key = cv2.waitKey(1) & 0xFF
    # if key == 27:
    #     break

clf = svm.SVC(gamma="scale")
hog_features = np.array(hog_features)
labels = labels.reshape(len(labels),1)

print(labels.shape)
print(hog_features.shape)
data_frame = np.hstack((hog_features, labels))
np.random.shuffle(data_frame)

percentage = 80
partition = int(len(hog_features)*percentage/100)
print(data_frame.shape)
x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

print("Accuracy: "+ str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))
print('########### ############')
print(confusion_matrix(y_test, y_pred))