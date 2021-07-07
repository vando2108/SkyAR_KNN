import cv2
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from cv2.ximgproc import guidedFilter

def convertColor(imgHD):
  imgHD = cv2.cvtColor(imgHD, cv2.COLOR_BGR2RGB)
  imgHD = np.array(imgHD / 255., dtype=np.float32)
  return imgHD

def removeOutliers(prev_pts, curr_pts):
  d = np.sum((prev_pts - curr_pts)**2, axis=-1)**0.5

  d_ = np.array(d).reshape(-1, 1)
  kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(d_)
  density = np.exp(kde.score_samples(d_))

  prev_pts = prev_pts[np.where((density >= 0.1))]
  curr_pts = curr_pts[np.where((density >= 0.1))]

  return prev_pts, curr_pts

def build_transformation_matrix(transform):
  transform_matrix = np.zeros((2, 3))
  transform_matrix[0, 0] = np.cos(transform[2])
  transform_matrix[0, 1] = -np.sin(transform[2])
  transform_matrix[1, 0] = np.sin(transform[2])
  transform_matrix[1, 1] = np.cos(transform[2])
  transform_matrix[0, 2] = transform[0]
  transform_matrix[1, 2] = transform[1]

  return transform_matrix

def estimate_partial_transform(matched_keypoints):
	prev_matched_kp, cur_matched_kp = matched_keypoints
	transform = cv2.estimateAffinePartial2D(np.array(prev_matched_kp), np.array(cur_matched_kp))[0]

	if transform is not None:
	    # translation x
	    dx = transform[0, 2]
	    # translation y
	    dy = transform[1, 2]
	    # rotation
	    da = np.arctan2(transform[1, 0], transform[0, 0])
	else:
	    dx = dy = da = 0
	
	return [dx, dy, da]

def update_transformation_matrix(M, m):
	M_ = np.concatenate([M, np.zeros([1,3])], axis=0)
	M_[-1, -1] = 1
	m_ = np.concatenate([m, np.zeros([1,3])], axis=0)
	m_[-1, -1] = 1

	M_new = np.matmul(m_, M_)
	return M_new[0:2, :]

class SkyBox():
	def __init__(self):
		self.load_skybox()
		self.M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
		self.frame_id = 0
		self.auto_light_matching = False
		self.relighting_factor = 0.8
		self.recoloring_factor = 0.5
		self.halo_effect = True

	def load_skybox(self):
		skybox_img = cv2.imread('data/background.jpg', cv2.IMREAD_COLOR)
		skybox_img = cv2.cvtColor(skybox_img, cv2.COLOR_BGR2RGB)
		skybox_img = cv2.resize(skybox_img, (640, 360))

		skybox_center_crop = 0.5
		cc = 1. / skybox_center_crop
		imgtile = cv2.resize(skybox_img, (int(cc * 640), int(cc * 360)))
		self.skybox_imgx2 = self.tile_skybox_img(imgtile)
		self.skybox_imgx2 = np.expand_dims(self.skybox_imgx2, axis=0)

	def tile_skybox_img(self, imgtile):
	  screen_y1 = int(imgtile.shape[0] / 2 - 360 / 2)
	  screen_x1 = int(imgtile.shape[1] / 2 - 640 / 2)
	  imgtile = np.concatenate(
	            [imgtile[screen_y1:,:,:], imgtile[0:screen_y1,:,:]], axis=0)
	  imgtile = np.concatenate(
	            [imgtile[:,screen_x1:,:], imgtile[:,0:screen_x1,:]], axis=1)
	
	  return imgtile

	def get_mask(self, frame):
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		l_green = np.array([32, 94, 132])
		r_green = np.array([179, 255, 255])
		mask = cv2.inRange(hsv, l_green, r_green)
		for i in range(len(mask)):
		  for j in range(len(mask[i])):
		    if (mask[i][j] == 0):
		      hsv[i][j][0] = 0
		      hsv[i][j][1] = 0
		      hsv[i][j][2] = 0
		    else:
		      hsv[i][j][0] = 255
		      hsv[i][j][1] = 255
		      hsv[i][j][2] = 255
		return hsv
	
	def skymask_refinement(self, G_pred, img):
		r, eps = 20, 0.01
		refined_skymask = guidedFilter(img[:,:,2], G_pred[:,:,0], r, eps)

		refined_skymask = np.stack(
		    [refined_skymask, refined_skymask, refined_skymask], axis=-1)

		return np.clip(refined_skymask, a_min=0, a_max=1)

	def get_skybg_from_box(self, m):
		self.M = update_transformation_matrix(self.M, m)

		nbgs, bgh, bgw, c = self.skybox_imgx2.shape
		fetch_id = self.frame_id % nbgs
		skybg_warp = cv2.warpAffine(
		    self.skybox_imgx2[fetch_id, :,:,:], self.M,
		    (bgw, bgh), borderMode=cv2.BORDER_WRAP)

		skybg = skybg_warp[0:360, 0:640, :]

		self.frame_id += 1

		return np.array(skybg, np.float32)/255.

	def skybox_tracking(self, frame, frame_prev, skymask):
	    if np.mean(skymask) < 0.05:
	        print('sky area is too small')
	        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

	    prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
	    prev_gray = np.array(255*prev_gray, dtype=np.uint8)
	    curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	    curr_gray = np.array(255*curr_gray, dtype=np.uint8)

	    mask = np.array(skymask[:,:,0] > 0.99, dtype=np.uint8)

	    template_size = int(0.05*mask.shape[0])
	    mask = cv2.erode(mask, np.ones([template_size, template_size]))

	    # ShiTomasi corner detection
	    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

	    if prev_pts is None:
	        print('no feature point detected')
	        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

	    # Calculate optical flow (i.e. track feature points)
	    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
	    # Filter only valid points
	    idx = np.where(status == 1)[0]
	    if idx.size == 0:
	        print('no good point matched')
	        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

	    prev_pts, curr_pts = removeOutliers(prev_pts, curr_pts)

	    if curr_pts.shape[0] < 10:
	        print('no good point matched')
	        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

	    # limit the motion to translation + rotation
	    dxdyda = estimate_partial_transform((np.array(prev_pts), np.array(curr_pts)))
	    m = build_transformation_matrix(dxdyda)

	    return m

	def relighting(self, img, skybg, skymask):
		step = int(img.shape[0]/20)
		skybg_thumb = skybg[::step, ::step, :]
		img_thumb = img[::step, ::step, :]
		skymask_thumb = skymask[::step, ::step, :]
		skybg_mean = np.mean(skybg_thumb, axis=(0, 1), keepdims=True)
		img_mean = np.sum(img_thumb * (1-skymask_thumb), axis=(0, 1), keepdims=True) \
		           / ((1-skymask_thumb).sum(axis=(0,1), keepdims=True) + 1e-9)
		diff = skybg_mean - img_mean
		img_colortune = img + self.recoloring_factor*diff

		if self.auto_light_matching:
		    img = img_colortune
		else:
		    img = self.relighting_factor*(img_colortune + (img.mean() - img_colortune.mean()))

		return img

	def halo(self, syneth, skybg, skymask):
		# reflection
		halo = 0.5*cv2.blur(
		    skybg*skymask, (int(640/5),
		                    int(640/5)))
		# screen blend 1 - (1-a)(1-b)
		syneth_with_halo = 1 - (1-syneth) * (1-halo)

		return syneth_with_halo

	def skyblend(self, img, img_prev, skymask):
		m = self.skybox_tracking(img, img_prev, skymask)
		skybg = self.get_skybg_from_box(m)
		img = self.relighting(img, skybg, skymask)
		syneth = img * (1 - skymask) + skybg * skymask
		if self.halo_effect:
		    # halo effect brings better visual realism but will slow down the speed
		    syneth = self.halo(syneth, skybg, skymask)

		return np.clip(syneth, a_min=0, a_max=1)

	def write_video(self, img_HD, syneth):
		frame = np.array(255.0 * syneth[:, :, ::-1], dtype=np.uint8)

		frame_cat = np.concatenate([img_HD, syneth], axis=1)
		frame_cat = np.array(255.0 * frame_cat[:, :, ::-1], dtype=np.uint8)

if __name__ == '__main__':
	sb = SkyBox()
	cap = cv2.VideoCapture('data/output.mp4')
	ret, frame = cap.read()
	mask = sb.get_mask(frame)
	frame = convertColor(frame)
	mask = convertColor(mask)
	temp = sb.skyblend(frame, frame, mask)
	cv2.imshow('mask', mask)
	cv2.waitKey(0)
	cv2.destroyWindow()