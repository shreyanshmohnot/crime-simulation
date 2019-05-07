import cv2
import os

image_folder = 'images'
video_name = 'B_n_TC_evo.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

#video = cv2.VideoWriter(video_name, 0, 2, (width,height))
video = cv2.VideoWriter(video_name, 0, 2, (width,height))
cv2.VideoWriter_fourcc('M','J','P','G')
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()