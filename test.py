import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 4))
plt.axis('off')
ims = [[plt.imshow(img[:, :, ::-1], animated=True)]
       for img in sf.output_img_list[0:40]]
ani = animation.ArtistAnimation(fig, ims, interval=50)

HTML(ani.to_jshtml())
