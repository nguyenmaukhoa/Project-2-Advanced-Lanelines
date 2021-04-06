## Advanced Lane Finding Project

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

[image1]: ./output_images/callibration/calibration2.jpg
[image1b]: ./output_images/callibration/calibration3.jpg
[image2]: ./output_images/callibration/test1.jpg
[image2b]: ./output_images/callibration/test3.jpg
[image3]: ./output_images/color_and_gradient/test1.jpg
[image3b]: ./output_images/color_and_gradient/test3.jpg
[image4]: ./output_images/perspective_transform/straight_lines1.jpg
[image5]: ./output_images/perspective_transform/test2.jpg
[image6]: ./output_images/lane_findings/test1.jpg
[image7]: ./output_images/lane_findings/test2.jpg
[image8]: ./output_images/lane_search_around/test1.jpg
[image9]: ./output_images/lane_search_around/test2.jpg
[image10]: ./output_images/draw_lane_area/Detected_lanes_straight_lines1.jpg
[image11]: ./output_images/draw_lane_area/Detected_lanes_test2.jpg
[image12]: ./output_images/draw_lane_area/Detected_lanes_test6.jpg
[video1]: ./output_videos/project_video_result.mp4
[video2]: ./output_videos/challenge_video_result.mp4
[video3]: ./output_videos/harder_challenge_video_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 4th and 5th code cell of the IPython notebook located in "./Advanced_LaneLines_Project.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners 9x6 in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image1b]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
![alt text][image2b]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (at 5th code cell of the IPython notebook).  Here's an example of my output for this step. 

![alt text][image3]
![alt text][image3b]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `corners_unwarp()`, which appears at the 7th code cell of the IPython notebook.  The `corners_unwarp()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
# Choose offet to calculate for destination image
offset = 100 #

# Get the 4 points in the source image
src = np.float32([[585, 455], 
                   [705, 455], 
                   [1130, 720], 
                   [190, 720]])

# Calculate destination points
dst = np.float32([[offset, 0],
                 [img_size[0] - offset, 0],
                 [img_size[0] - offset, img_size[1]],
                 [offset, img_size[1]]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions `find_lane_pixels()` and `fit_poly()` which identify lane lines and fit a second-order polynomial to both right and left lane lines are clearly labeled in the Jupyter notebook (code cell 10th and 11th). 

The first of these computes a histogram of the bottom half of the image and finds the bottom-most x position (or "base") of the left and right lane lines. Originally these locations were identified from the local maxima of the left and right halves of the histogram.

The function then identifies 9 windows from which to identify lane pixels, each one centered on the midpoint of the pixels from the window below. This effectively "follows" the lane lines up to the top of the binary image, and speeds processing by only searching for activated pixels over a small portion of the image. 

Pixels belonging to each lane line are identified and the Numpy polyfit() method fits a second-order polynomial to each set of pixels. The images below demonstrate how this process works:

![alt text][image6]
![alt text][image7]

The `search_around_poly()` function performs basically the same task, but searching for lane pixels within a certain range. The later frames in the videos are found within this range to avoid noises and speed up the searching task.

![alt text][image8]
![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated in the code cell 17th by the function `measure_curvature_real()`. Sample images are included in session 6 below.

curve_radius = ((1 + (2*fit[0]*y_0*y_meters_per_pixel + fit[1])**2)**1.5) / np.absolute(2*fit[0])

 - Fit[] is the array of coefficients of the second-order polynomial. 
 - y_0 is the y position within the image upon which the curvature calculation is based
 - ym_per_pix = 30/720 is meters per pixel in y dimension

The position of the vehicle with respect to the center of the lane is calculated:

lane_center_position = (r_fit_x_int + l_fit_x_int) /2
center_dist = (car_position - lane_center_position) * xm_per_pix

 - xm_per_pix = 3.7/700
 - r_fit_x_int and l_fit_x_int are the x-intercepts of the right and left fits, respectively. 
 - The car position is the difference between these intercept points and the image midpoint 
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code cell 16th in the function `draw_lane_area()`.  Basically, this function is the combination of all the functions mentioned above
Here is an example of my result on a test image:

![alt text][image10]
![alt text][image11]
![alt text][image12]
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

 - Here's a [link to my main project video result](./output_videos/project_video.mp4)
 - Here's a [link to my challenge video result](./output_videos/challenge_video_result.mp4)
 - Here's a [link to my harder chalenge video result](./output_videos/harder_challenge_video_result.mp4)
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems I had during this project are lighting conditions, shadows, lane confusions caused by lines other the lanes, different kinds of curves... Moreover, I don't know how to use average curvature when lines are not detected. That's why I failed in the challenge video and the harder challenge video. And I don't

I've considered a few possible approaches for making my algorithm more efficient:
 - More dynamic thresholding to adapt to complicated lighting conditions
 - Develop a confidence level for fits and rejecting new fits that deviate beyond a certain amount
 - Avoid horizontal lines and redundant lines in the middle of the lanes