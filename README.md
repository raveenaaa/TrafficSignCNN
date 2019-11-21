# TrafficSignCNN
This is a repo for recognition of Traffic sign images using a Convolutional Neural Network.
How to run this code:
1. Clone this repo
2. Download the data from this link: https://drive.google.com/file/d/1AaBbNpHfLHAbRkazMt6XRlGbLbqqpgsU/view?usp=sharing and extract to the root repo folder. The original data files are obtained from this link: https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed, however we have combined all these files into a zip folder for convenience). Make sure that the 'data' folder is present in the root folder otherwise the code might give an error.
3. Run one of the files titled traffic_signs_classification_data*.py (this runs the corresponding experiments for each of the data files. Keep in mind that in our testing we used a Nvidia Quadro GV100 GPU, and each runs takes ~45 mins to complete, so please run these on high end GPUs only!!)

## Data
Each of the data files is a pickle file that has been preprocessed from the original Traffic sign images database. Each of these files has a corresponding training, testing and a validation set to make the testing easier.

The preprocessed nine files are as follows:
- data0.pickle - Shuffling
- data1.pickle - Shuffling, /255.0 Normalization
- data2.pickle - Shuffling, /255.0 + Mean Normalization
- data3.pickle - Shuffling, /255.0 + Mean + STD Normalization
- data4.pickle - Grayscale, Shuffling
- data5.pickle - Grayscale, Shuffling, Local Histogram Equalization
- data6.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 Normalization
- data7.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 + Mean Normalization
- data8.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 + Mean + STD Normalization

Datasets data0 - data3 have RGB images and datasets data4 - data8 have Gray images.


Shapes of data0 - data3 are as following (RGB):
- x_train: (86989, 3, 32, 32)
- y_train: (86989,)
- x_validation: (4410, 3, 32, 32)
- y_validation: (4410,)
- x_test: (12630, 3, 32, 32)
- y_test: (12630,)


Shapes of data4 - data8 are as following (Gray):
- x_train: (86989, 1, 32, 32)
- y_train: (86989,)
- x_validation: (4410, 1, 32, 32)
- y_validation: (4410,)
- x_test: (12630, 1, 32, 32)
- y_test: (12630,)

The original dataset was obtained from here: http://benchmark.ini.rub.de/?section=gtsrb&subsection=news
