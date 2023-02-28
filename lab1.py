import cv2
import numpy as np

ix, iy =  -1, -1
end_x, end_y = -1, -1
creating = False

def create_rectangle(event, x, y, flags, pra):
    global ix, iy, creating,  img, end_x, end_y
    if event == cv2.EVENT_LBUTTONDOWN:
        creating  = True
        ix = x
        iy = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if creating == True:
            cv2.rectangle(back_ground,(ix, iy), (x, y), (20,20, 255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
            creating = False
            end_x , end_y = x, y
            cv2.imwrite('orginal.png',back_ground )

back_ground = np.ones((500, 500,3), np.uint8)*255
cv2.namedWindow("Rectangle Window")
cv2.setMouseCallback("Rectangle Window", create_rectangle)

while True:
   cv2.imshow("Rectangle Window", back_ground)
   if cv2.waitKey(10) == 27:
      break
cv2.destroyAllWindows()


center = ((end_x+ ix)/2, (end_y + iy)/2)
rotate_matrix = cv2.getRotationMatrix2D(center = center, angle = 45,  scale = 1)
rotated_image = cv2.warpAffine(src=back_ground, M=rotate_matrix, dsize=(500,500),borderValue = (255, 255,255))
cv2.imwrite('rotated image.png',rotated_image)

tx, ty = 50, 50
M = np.array([
    [1,0,tx],
    [0,1,ty]
], dtype = np.float32)
translated_image = cv2.warpAffine(src = back_ground, M = M, dsize = (500, 500), borderValue = (255, 255,255))
cv2.imwrite('translated_image.png', translated_image)


back_ground = np.ones((500, 500, 3), np.uint8)* 255
N =1.5
width = end_x - ix
height = end_y - iy
scale_image = cv2.rectangle(back_ground, (ix, iy), (int(ix+width*N), int(iy+height*N)),(20,20, 255),-1)
cv2.imwrite('scale.png', scale_image)
