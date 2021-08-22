import imgviz
import numpy as np
import cv2
import matplotlib.pyplot as plt

color = np.load("/home/demo/catkin_ws/src/deep_grasping_ros/color.npy")
segmap = np.load("/home/demo/catkin_ws/src/deep_grasping_ros/segmap.npy")

masks = []
for id in np.unique(segmap):
    if id == 0.0:
        continue
    mask = np.zeros_like(segmap)
    mask = np.where(segmap==id, True, False)
    masks.append(mask)
masks = np.array(masks)

insviz = imgviz.instances2rgb(
    image=color,
    masks=masks,
    labels=np.uint8(np.unique(segmap))[1:]
)
plt.figure(dpi=200)
plt.plot()
plt.imshow(insviz)
img = imgviz.io.pyplot_to_numpy()
plt.close()
print(img.shape, np.max(img))
# mg = imgviz.io.pyplot_to_numpy()

# plt.imshow(insviz)