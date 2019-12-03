import argparse
import cv2
import os
# from matplotlib import pyplot as plt
from skimage import feature, exposure
import numpy as np
from skimage import color
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--folder', default="VIRB0037-3", help='Provide path to video file')
args = vars(ap.parse_args())

time_start = datetime.now()
hog_images = []
hog_features =[]
model_name = "scoop_feed_model.sav"

img_path = "train_data/scoop_feed/" + args["folder"]
claw_down = np.array(("claw_down"))
claw_down = np.repeat(claw_down, 86)
claw_up = np.array(("claw_up"))
claw_up = np.repeat(claw_up, 133)
labels = np.hstack((claw_down, claw_up))

for img in os.listdir(img_path):
    # print(img)
    frame = cv2.imread(os.path.join(img_path, img))
    frame = cv2.resize(frame, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)
    one, two, three = cv2.split(hsl)
    new_sat = exposure.adjust_sigmoid(two, 0.75, inv=True)
    # new_img = cv2.merge([new_sat, two, three])
    canny = cv2.Canny(new_sat, 200, 255)
    new_fd, new_hog = feature.hog(canny, orientations=9, pixels_per_cell=(20, 20), block_norm="L1",
                                  cells_per_block=(3, 3), transform_sqrt=False, visualize=True, multichannel=False,
                                  feature_vector=True)
    print(new_fd.shape)
    new_hog = exposure.rescale_intensity(new_hog, in_range=(0, 20))
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

print("Calculation of HOG images finalized.\n"
      "Starting SVM training")
print("Time elapsed: {}".format(datetime.now()-time_start))

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
print("Training finished. Saving the model.")
print("Time elapsed: {}".format(datetime.now()-time_start))
pickle.dump(clf, open(model_name, 'wb'))

print("Starting prediction")
y_pred = clf.predict(x_test)

print("Accuracy: "+ str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))
print('########### ############')
print(confusion_matrix(y_test, y_pred))
print("Time elapsed: {}".format(datetime.now()-time_start))