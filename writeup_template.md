## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./test_undist.jpg "Undistorted"
[image2]: ./Undistorted.PNG "Road Transformed"
[image3]: ./Color_Combined.PNG "Binary Example"
[image4]: ./Perspective_Transform.PNG "Warp Example"
[image5]: ./Lane_Line.PNG "Fit Visual"
[image6]: ./Image_Lane.PNG "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./Advanced_Lane_Line.ipynb" .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objpoint` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at 3 rd code cell in `Advanced_Lane_Line.ipynb`).
As for Gradient threshold I used:
sobelx of the gray image with doing AND operation with sobelx of r-channel to get binary output as given below.

**(sobel_X == 1)&(scaled_sobel_R==1)**

As for Color Threshold I used:
S-channel of HSL color space as because to detect the lane lines under extreme light,R-Channel of RGB for better lane line and LAB color space for detecting yellow lane lines.As I applied AND operation for all the color channel given below.

**(s_binary==1)&((s_binary_R==1)&(s_binary_LV==1))**

And at last I have used OR operation to combine and form image as given below

Here's an example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `corners_unwarp()`, which appears in 4th code cell  in the file `Advanced_Lane_Line.ipynb` (./Advanced_Lane_Line.ipynb) .  The `corners_unwarp()()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner which I drawn manually:

```python
    src=np.float32([(565, 470),(260, 674),(1048, 674),(720, 470)])

    dst=np.float32([(260, 100),(260, 674), (1048, 674), 
                                 (1048, 100)
                                 ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 565, 470      | 260, 100        | 
| 260, 674      | 260, 674      |
| 1048, 674     | 1048, 674      |
| 720, 470      | 1048, 100        |


**My previous points were as source and destination points:**

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # 570-600 # in my code in `Advanced_Lane_Line.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # 446-464 # in my code in `Advanced_Lane_Line.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

1.As my pipeline previously failed for correctly rectify perspective transform for project video but after correcting it became more stable.As one of my observation was wobbling lines due which it lane line was getting fluctuated to combat that I used sobelx of r-channel which made my lane detection more stable. My pipeline fails only for very sharp shadow and light change which can be modified by a better robust color thresholds.

2.Averaging of lane line will also help lane line to detect image better for other challenging video basically where lane lines are not present.
