
## Udacity self driving car nanodegree 

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

[image1]: ./writeup_imgs/chess.png "Undistorted chessboard"
[image2]: ./writeup_imgs/undistort.png "Road Transformed"
[image3]: ./writeup_imgs/tresholds.png "Binary Example"
[image4]: ./writeup_imgs/warped.png "Warp Example"
[image5]: ./writeup_imgs/fit.png "Fit Visual"
[image6]: ./writeup_imgs/drawed.png "Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first three code cells of the IPython notebook  "Advanced_lane_finding" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and the obtained distortion coeficients sved in the "mtx" and "dist" arrays. The undistorted chessboard images look like the following result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

The image was undistorted calling the OpenCV function "undistort(img,mtx,dist,None,mtx)" using the previously obtained distortion coeficents as explained in the previous point.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cell 9 in `Avanced_line_finding.pynb`).  Here's an example of my output for this step. 

![alt text][image3]

First the image was converted to HLS and LAB color spaces and after try error loops S and B channels were found as the critical ones to use. Once isolated the S and B channels, x and y sobel gradient tresholds where applyed to the image loosing less information than with the usual grayscaled image. The undiscarded noise of trees and other objects is later discarded correctl aplying the perspective transform. 

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in the cell 10 in the file `Avanced_line_finding.pynb`.  The `warp()` function takes as inputs an image (`img`), as well as M transformation matrix calculated in the cell number 9 from the defined source and destination points.  I chose the hardcode the source and destination points in the following manner:

```python
#SOURCE RECTANGLE POINTS
src = np.float32([(565,460),
                  (728,460), 
                  (1195,685), 
                  (150,685)])

#WARPED RECTANGLE POINTS    
dst = np.float32([(350,0),  
                  (img.shape[1]-350,0), 
                  (img.shape[1]-350,img.shape[0]),
                  (350,img.shape[0])])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 565, 460      | 350, 0        | 
| 728, 460      | 930, 0        |
| 1197, 685     | 930, 720      |
| 150, 685      | 350, 720      |

I verified that my perspective transform was working as expected by drawing the rectangle formed by `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Using the code provided by Udacity, I implemented the sliding window search method `polyfit_sliding_window` (cell 12) backed by the histogram peak first aproximation. Once detected the pixel point belonging to the lines, a secon order polynomial was fitted to each right and left lines.

Following the class advices I created a Line class (cell 16)  to save the 10 previous best fits obtained and checked with a brief sanity check that compares the coeficients of the recently obtained fit with the previous saved average. If there is a fit saved and considerated best fit, the other explained method is used.

This other method `polyfit_next_frame` (cell 12) is quite faster, as it searches for the line within a 40px margin from the previous fit.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the function `calc_curv_rad_center`, in the cell number 14 of the notebook, converting from pixel to metric world using the Udacity provided conversions, I calculated the distance to the center comparing the image center point with the center of the two fitted polynomials in the bottom of the picture.

```python
car_position = (bin_img.shape[1])/2
l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
lane_center_position = (r_fit_x_int + l_fit_x_int) /2
center_dist = (car_position - lane_center_position) * xm_per_pix
```

For the curvature, I used the code provided in the classroom as follows

```python
 Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the cell  14, in the function `draw_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

As can be seen the function, appends the curvature and center to distance values using the values obtained by the `calc_curv_rad_center` function and draws the identifyed lane portion using the classroom code 

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline may fail in very extreme curvature radius, as the perspective transform discards the sourroundings maybe too agresively. 

On the other hand, it could also fail in different lighting conditions, as the tresholds have been adapted to the project video data to perform better, in a certain sense the pipeline is overfitted to the project video and a more robust pipeline should be developed based in more invariant data in the future 
