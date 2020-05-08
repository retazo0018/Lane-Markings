import cv2

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image) # always make a copy when working with image arrays

canny_image = canny(lane_image)

cropped_image = region_of_interest(canny_image)

# to detect lines in cropped gradiant image
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

# optimization
averaged_lines =  average_slope_intercept(lane_image, lines) 

line_image = display_lines(lane_image, averaged_lines)

# weighted sum of color image and line image to make lines appear on original image.
# 2nd and 4th arg - pixel intensities! 4th arg is 1 for lines to appear more clearer.
# last arguement is gamma arguement
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
