## Vehicle Dectection Project

###### Udacity Self Driving Car Engineer Nanodegree - Project 5 - Yi CHEN - 25/02/2017
--

The goal of this project is to develop a pipeline to detect vehicles on a video taken from a forward-facing center camera mounted on the front of a car:

The pipeline of processing frame image of the video is:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images, also apply color transform and append binned color features, as well as histograms of color, to the HOG feature vector. Then train a classifier on extracted features.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles. Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_examples.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/bboxes_and_heat.png
[image5]: ./output_images/labels_map.png
[image6]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

---

###Histogram of Oriented Gradients (HOG)

The code for this section is contained in the first part `1. Feature Extraction & Classfier Training` in the IPython notebook.  

####1. Explain how you extracted HOG features from the training images.

I started by defining small functions that allowed me to extract HOG features, color histogram features and spatial binned color features, as well to a color space converstion function.

Then I read in all the `vehicle` and `non-vehicle` images. The details of how images were read in will be mentioned in the classifier training part. Below is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `grayscale` and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

To settle HOG parameters, I tried various combinations of parameters with a Linear SVM classifer, and I found:

* setting `hog_channel=ALL` could have a better prediction accuray as we have thus more features to train than a single channel
* the color space `YCrCb` peformed the best among all tested color spaces: `RGB`, `HSV`, `LUV`, `HLS`, `YUV`, `YCrCb`
* the orientation value of 8 or 16 seemed to be better than 32 as the later proved to be producing more false postive detections when searching cars in an image. In order to get more features for training, I chose an orientation = 16.
* `pix_per_cell=8 `and `cell_per_block=2` seemed to perform well. 


####3. Describe how you trained a classifier using your selected HOG features (and color features if you used them).

I encountered multiple problems in this classifer training part:

* First, I used `train_test_split` function from scikit learn library to split my training & testing data. Then I tested 2 classifers (SVM and RandomForest) with multiple combinations of HOG parameters. The testing accuracies were always very high(around 0.94-0.98). But it turned out to not work well when searching vehicles on image: a lot of false postive detection for SVM while losing vehicles for RandomForest.
* Then I changed my train & test strategy to manual data split. For cars, I took KITTI images as training images, and GTI images as testing images as GTI contain time series images and are quite different from KITTI images. For not cars, I took Extras images for training for GTI images for testing. The testing accuracy dropped significantly(SVM: 0.84, RandomForest: 0.66) and I could only found a part of cars when doing sliding window search. So the model was definitly under-fitting.
* I noticed that the classifers tended to loose white cars, with hope to get more training data for white cars, I looked into Udaicty's labeled dataset (object-detection-crowdai) and extracted around 1900 car images to add into car dataset for training. But the result turned out to be even worse as there were many front-view cars presented in the extracted car images.
* So finally, I returned back to the vehicle and not-vehicle dataset and decided to take 20% bottom images from each folder to constitute my testing dataset.
* I tried to use `GridSearchCV` to find the optimal parameters for SVM and RandomForest, while it turned out that the optimal parameters were: 
		
		 LinearSVC:
		 {'loss': 'hinge', 'C': 0.1}
		 
		 RandomForestClassifier:
		 {'max_depth': 10, 'max_features': 3,
         'min_samples_leaf': 5, 'n_estimators': 500} 
The classifiers still gave very high testing accuracy. However, when I tested the optimized RandomForest on images, it either only detected a part of car body, or loosed the whole car. Thus, I decided to continue with SVM even it produced false postive detections.

To remind, a standard feature scaling (removing the mean and scaling to unit variance) was performed on training & testing data before fitting the model.

###Sliding Window Search

The code for this section is contained in the second part `2. Sliding Window Search` in the IPython notebook.  

####1. Describe how you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I started by observing the size of cars and their positions in the test image. Cars were presented between 400 and 550 on y-axis and their sizes varied from 64 pixels to 200 pixels. So I decided to start with **64** pixels as original window, and increased little by little the scale, as well as the y-axis value to see how the detection will behave. And I found a scale of **1.5** between 400 and 650 on y-axis with an overlapping of **0.75** could detect well cars (still some false postives occurred).

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on just 1 scale on test images using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a moderately good result with false postives.  Here are some example images:

![alt text][image3]

I tried many things to optimize the classifier's performance (see above 'how you trained a classifier' section) but without much improvements...

---

### Video Implementation

The code for this section is contained in the third part `3. Video Pipeline` in the IPython notebook.  

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In order to get rid of false postives, I combined two techniques:

* **Heatmap creation**: for each frame in the video, I recorded the positions of positive detections and created a heatmap for that frame.  

* **Frames smoothing**: I implemented a `VideoHeatmap` class to receive the heatmap of each frame, and summed heatmaps over 6 continuous frames. I then thresholded that summed map with a threshold value of **10** to identify vehicle positions.  After thresholding, I used `scipy.ndimage.measurements.label()` to identify individual blobs in the summed heatmap. I then assumed each blob corresponded to a vehicle and I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame:

#### Here are six frames and their corresponding heatmaps:

![alt text][image4]

#### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image5]

#### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image6]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust? 

In this project, the major problem for me was how to train classifier. As I detailed in the corresponding section, I had difficulties in how to split training & test data as :

* testing accuracy stayed always very high (0.94-0.98) either I performed a random train & test split, or selecting bottom 20% images as testing data from each image folder
* testing accuracy dropped too much if I trained on KITTI and Extra data and tested on GTI data.
* Udacity's addtional dataset didn't help much as images inside were mixed of car orientations (many front-view cars and not enough white cars)

Thus, it also caused problems in selecting an appropriate classifier:

* SVM always produced false postitve detections
* RandomForest had difficulty in detecting white cars

After filtering out false postives for SVM detections, there were still weired boxes during a very short of time in the final video and for few frames, we lost the white car's track, especially when it drove over a section of road whose color was close to white and when the black car came into the view.

To improve the pipeline, I think we could :

* make the test data well different from training data in order to better selecting & tuning a classifer.
* get more data for training, e.g. cars of different colors, distances and angles, cars under diffierent light & weather conditions...
* try more sophisticated algorithms like xgboost, neural networks or convolutional neural networks.
