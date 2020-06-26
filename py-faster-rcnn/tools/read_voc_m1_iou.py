import matplotlib
matplotlib.use('Agg') 
import _init_paths
import pickle
import json
from collections import defaultdict
from datasets.factory import get_imdb
import csv


class Box():
	def __init__(self, cls, bb, score=None):
		self.cls = cls
		if score is not None:
			self.score = round(float(score),4)
		else:
			self.score = score
		self.x = round(float(bb[0]),4)
		self.y = round(float(bb[1]),4)
		self.xx = round(float(bb[2]),4)
		self.yy = round(float(bb[3]),4)
		self.w = self.xx - self.x
		self.h = self.yy - self.y

	def __str__(self):
		if self.score == None:
			return "X:{} Y:{} XX:{} YY:{} W:{} H:{} Class:{}".format(self.x,self.y,self.xx,self.yy, self.w, self.h, self.cls)
		else:
			return "X:{} Y:{} XX:{} YY:{} W:{} H:{} Class:{} Score:{}".format(self.x,self.y,self.xx,self.yy, self.w, self.h, self.cls, self.score)

def overlap(x1, w1, x2, w2):
	l1 = x1 - (w1 / 2.0)
	l2 = x2 - (w2 / 2.0)
	left = l1 if l1 > l2 else l2
	r1 = x1 + (w1 / 2.0)
	r2 = x2 + (w2 / 2.0)
	right = r1 if r1 < r2 else r2
	return right - left

def box_intersection(a, b):
	w = overlap(a.x, a.w, b.x, b.w)
	h = overlap(a.y, a.h, b.y, b.h)
	if w < 0 or h < 0: return 0
	area = w * h
	return area

def box_union(a, b):
	i = box_intersection(a, b)
	u = a.w * a.h + b.w * b.h - i
	return u

def box_iou(a, b):
	return round(box_intersection(a, b) / box_union(a, b), 4)

def box_iou_old(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA.x, boxB.x)
	yA = max(boxA.y, boxB.y)
	xB = min(boxA.xx, boxB.xx)
	yB = min(boxA.yy, boxB.yy)
 
	# compute the area of intersection rectangle
	if ((xB - xA + 1 < 0 ) or ( yB - yA + 1 < 0)) :
		return 0
	
	interArea = (xB - xA + 1) * (yB - yA + 1)
	
	#print('InterArea {0:0.2f}'.format(interArea))

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA.w + 1) * (boxA.h + 1)
	boxBArea = (boxB.w + 1) * (boxB.h + 1)

	#print('Union {0:0.2f}'.format(float(boxAArea + boxBArea - interArea))) 
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	
 
	# return the intersection over union value
	return round(iou,4)


imdb = get_imdb('voc_2007_test')

#path = '../output/faster_rcnn_end2end/voc_2007_test/'

places = defaultdict(list)
with open('places_voc.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in spamreader:
		places[row[0]] = row[1] 

with open('%s/vgg16_context_concat_4fc_img224_faster_rcnn_iter_70154/faster/detections.pkl'%(path), 'rb') as f:
	data_faster = pickle.load(f)

with open('%s/vgg16_context_concat_4fc_img224_faster_rcnn_iter_70154/context/detections.pkl'%(path), 'rb') as k:
	data_context = pickle.load(k)

gt = imdb.get_annotation()

with open('gt_voc_iou.csv', 'w') as csvfile:
    	spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(['img', 'bb' , 'obj_gt', 'obj_faster', 'obj_context', 'score_faster', 'score_context', 'iou_faster', 'iou_context', 'place'])

	# index images
	for i in xrange(imdb.num_images):

		image_id = imdb.image_path_at(i).split('/')[-1].split('.jpg')[0]
		place_img = image_id.zfill(6)
	
		gt_bb = gt[i]['boxes']
		gt_cls = gt[i]['gt_classes']
	
		# index classes to GT
		for j in xrange(0, len(gt_cls)):
			classe = gt_cls[j]
			gt_box = Box(classe, gt_bb[j])
			
			best_iou_faster = 0.0
			best_iou_context = 0.0
			best_iou_bb_faster = None
			best_iou_bb_context = None


			for cls in xrange(1, 21):
				
				# faster predictions
				for x in xrange(0,len(data_faster[cls][i])):
									
					pred_box = Box(cls, data_faster[cls][i][x][:-1], data_faster[cls][i][x][4])
					iou = box_iou(gt_box,pred_box)
					if best_iou_faster < iou:
						best_iou_faster = iou
						best_iou_bb_faster = pred_box

				# our model prediction using context
				for y in xrange(0,len(data_context[cls][i])):

					pred_box2 = Box(cls, data_context[cls][i][y][:-1], data_context[cls][i][y][4])
					iou = box_iou(gt_box,pred_box2)
					if best_iou_context < iou:
						best_iou_context = iou
						best_iou_bb_context = pred_box2
					
			# Writing resuls into a CSV file		

			if best_iou_faster == 0 and best_iou_context == 0:
				spamwriter.writerow([image_id, j , classe, 0, 0, 0, 0, 0, 0, places[place_img]])
			elif best_iou_faster == 0 and best_iou_context != 0:
				spamwriter.writerow([image_id, j , classe, 0, best_iou_bb_context.cls , 0, best_iou_bb_context.score ,0, best_iou_context, places[place_img]])
			elif best_iou_faster != 0 and best_iou_context == 0:
				spamwriter.writerow([image_id, j , classe, best_iou_bb_faster.cls  , 0, best_iou_bb_faster.score, 0, best_iou_faster, 0, places[place_img]])
			else:
				spamwriter.writerow([image_id, j , classe, best_iou_bb_faster.cls, best_iou_bb_context.cls, best_iou_bb_faster.score, best_iou_bb_context.score, best_iou_faster, best_iou_context, places[place_img]])
		
