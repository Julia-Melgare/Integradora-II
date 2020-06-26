#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import matplotlib
#matplotlib.use('Agg') 
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.test import im_detect2
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from os import listdir
import pickle
import cPickle

colors = matplotlib.cm.prism(np.linspace(0, 1, 21))

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'vgg16_concat': ('VGG16_Context_concat_4fc_img224',
                  'vgg16_context_concat_4fc_img224_faster_rcnn_iter_70154.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'context': ('contexr-rcnn', 'context-rcnn_iter_70154.caffemodel')}


def vis_detections(cls_ind, im, ax, class_name, dets, txt_file, top5, thresh=0.5):
	"""Draw detected bounding boxes."""
	inds = np.where(dets[:, -1] >= thresh)[0]
	if len(inds) == 0:
		return

	im = im[:, :, (2, 1, 0)]

	# fig, ax = plt.subplots(figsize=(12, 12))

	
	ax.imshow(im, aspect='equal')
   
	for i in inds:
		bbox = dets[i, :4]
		score = dets[i, -1]
		
		left = bbox[0]
		bottom = bbox[1]
		width = bbox[2] - bbox[0]
		height = bbox[3] - bbox[1]
		top = bottom + height
		right = left + width

		txt_file.write(str(int(cls_ind)) + ' ' + str(int(left)) + ' ' + str(int(bottom)) + ' ' + str(int(width)) + ' ' + str(int(height))+'\n')

		ax.add_patch(
			plt.Rectangle((bbox[0], bbox[1]),
						  bbox[2] - bbox[0],
						  bbox[3] - bbox[1], fill=False,
						  edgecolor=colors[cls_ind], linewidth=3.5)
			)
		#print 'left {} botton {} right {} top {}'.format(bbox[0],bbox[1], right, top)
		#print 'name: {}  bottom {} left {}, top: {}, right: {}, width {}, height {}'.format(class_name, bottom, left, top, right, width, height)


		if (im.shape[1]) <= right+1:
			#print 'right'
			ax.text(right, bottom-2,
					'{:s} {:.3f}'.format(class_name, score),
					 horizontalalignment='right',
					 verticalalignment='top',
					 bbox=dict(facecolor='white', edgecolor=colors[cls_ind],linewidth=5.0, alpha=0.9),
					 fontsize=16, color='black') 
		elif (bottom<=0):
			#print 'bottom'
			ax.text(right, top-height+5,
					'{:s} {:.3f}'.format(class_name, score),
					 horizontalalignment='left',
					 verticalalignment='top',
					 bbox=dict(facecolor='white', edgecolor=colors[cls_ind],linewidth=5.0, alpha=0.9),
					 fontsize=16, color='black')
		else:
			ax.text(left, bottom-2,
					'{:s} {:.3f}'.format(class_name, score),
					 horizontalalignment='left',
					 verticalalignment='top',
					 bbox=dict(facecolor='white', edgecolor=colors[cls_ind],linewidth=5.0, alpha=0.9),
					 fontsize=16, color='black')

	if len(top5)!=0:
		with open('resources/labels.pkl', 'rb') as f:
			labels = cPickle.load(f)		
			#ax.set_title(('Ambiente: {} ').format(labels[top5[0]],  y=0.87), fontsize=14)

	ax.text(0.5, 0.98, 'Ambiente: {} '.format(labels[top5[0]]),
        verticalalignment='top', horizontalalignment='center',
        transform=ax.transAxes,
        color='white', fontsize=20, bbox=dict(facecolor='black', alpha=0.2))

	plt.axis('off')
	plt.tight_layout()
	plt.draw()

def demo(net, image_name):
	"""Detect object classes in an image using pre-computed object proposals."""
	if cfg.TEST.CONTEXT:
		txt_file = open('test/context/leannet/'+im_name.split('.')[0]+'.txt','w')
	else:
		txt_file = open('test/context/faster/'+im_name.split('.')[0]+'.txt','w')
	# Load the demo image
	im_file = os.path.join('/content/py-faster-rcnn/data/demo_voc/', image_name)
	im = cv2.imread(im_file)
	height, width, channels = im.shape
	# Detect all object classes and regress object bounds
	timer = Timer()
	timer.tic()
	
	if cfg.TEST.PLACES:
		scores, boxes, top5 = im_detect2(net, im)
	else:
		scores, boxes = im_detect(net, im)
		top5=[]

	timer.toc()
	print ('Detection took {:.3f}s for '
		   '{:d} object proposals').format(timer.total_time, boxes.shape[0])

	# Visualize detections for each class
	CONF_THRESH = 0.55
	NMS_THRESH = 0.3
	
	fig, ax = plt.subplots(figsize=(12, 12))

	for cls_ind, cls in enumerate(CLASSES[1:]):
		cls_ind += 1 # because we skipped background
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes,
						  cls_scores[:, np.newaxis])).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep, :]
		vis_detections(cls_ind, im, ax, cls, dets, txt_file, top5, thresh=CONF_THRESH)
		

	
	ax.imshow(im[:, :, (2, 1, 0)], aspect='equal')
	#ax.set_title(('Probability of detection >= {:.1f}').format(CONF_THRESH),fontsize=14)
	plt.axis('off')
	plt.tight_layout()
	plt.draw()
	txt_file.close()

def parse_args():
	"""Parse input arguments."""
	parser = argparse.ArgumentParser(description='Faster R-CNN demo')
	parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
						default=0, type=int)
	parser.add_argument('--cpu', dest='cpu_mode',
						help='Use CPU mode (overrides --gpu)',
						action='store_true')
	parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
						choices=NETS.keys(), default='vgg16')
	parser.add_argument('--place', dest='place', help='Use place of the scene',
						default=1, type=int)
	parser.add_argument('--context', dest='context', help='Use context layer prob',
						default=1, type=int)

	args = parser.parse_args()

	return args

if __name__ == '__main__':
	cfg.TEST.HAS_RPN = True  # Use RPN for proposals

	args = parse_args()
	
	if args.place:
		cfg.TEST.PLACES = True
	else:
		cfg.TEST.PLACES = False

	if args.context:
		cfg.TEST.CONTEXT = True
	else:
		cfg.TEST.CONTEXT = False

	prototxt = r'/content/drive/My Drive/Integradora II/caffe/context-rcnn/test.prototxt'#os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
							#'faster_rcnn_end2end', 'test.prototxt')
	caffemodel = r'/content/drive/My Drive/Integradora II/caffe/context-rcnn_iter_70154.caffemodel'#os.path.join('/content/py-faster-rcnn', 'faster_rcnn_models',
							  #NETS[args.demo_net][1])

	if not os.path.isfile(caffemodel):
		raise IOError(('{:s} not found.\nDid you run ./data/script/'
					   'fetch_faster_rcnn_models.sh?').format(caffemodel))

	if args.cpu_mode:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu_id)
		cfg.GPU_ID = args.gpu_id
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)

	print '\n\nLoaded network {:s}'.format(caffemodel)

	# Warmup on a dummy image
	#im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
	im = 128 * np.ones((720, 1280, 3), dtype=np.uint8)
	for i in xrange(2):
		_, _= im_detect(net, im)

	caminho = '/content/py-faster-rcnn/data/demo_voc/'#/home/leandro/faster_pascalvoc/data/demo/'
	im_names = listdir(caminho)
	
	for im_name in im_names:
		
			
		
		if cfg.TEST.CONTEXT:
			if (os.path.isfile('test/context/leannet/'+ im_name.split('.')[0]+'.txt')==False):
				print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
				print 'Demo for data/demo/{}'.format(im_name)
				demo(net, im_name)
				plt.savefig( 'test/context/leannet/'+ im_name.split('.')[0]+'.pdf', format='pdf', bbox_inches='tight')
		else:
			if (os.path.isfile('test/context/faster/'+ im_name.split('.')[0]+'.txt')==False):
				print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
				print 'Demo for data/demo/{}'.format(im_name)
				demo(net, im_name)
				plt.savefig( 'test/context/faster/'+ im_name.split('.')[0]+'.pdf', format='pdf', bbox_inches='tight')

		#plt.show()
		plt.close()

		

   

	#
