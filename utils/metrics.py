import numpy as np
from torchmetrics.classification.jaccard import JaccardIndex

import torchmetrics
import torch


from torchmetrics.classification.dice import Dice
from torchmetrics.functional import dice_score

import torch.nn.functional as F

SMOOTH = 1e-6
from pprint import pprint

def batch_iou(pred, gt):
	print(pred)
	pred, gt = prepro(pred, gt)
	print(pred)
	print(pred.shape)
	iou_scores = []
	batch_size = pred.shape[0]
	for i in range(batch_size):
		label_class = gt[i].cpu().detach().numpy()
		label_class_predicted = pred[i].cpu().detach().numpy()
		# IOU score
		intersection = np.logical_and(label_class, label_class_predicted)
		union = np.logical_or(label_class, label_class_predicted)
		iou_score = np.sum(intersection) / np.sum(union)
		iou_scores.append(iou_score)
		print(iou_scores)
		# axes[i, 0].imshow(landscape)
		# axes[i, 0].set_title("Landscape")
		# axes[i, 1].imshow(label_class)
		# axes[i, 1].set_title("Label Class")
		# axes[i, 2].imshow(label_class_predicted)
		# axes[i, 2].set_title("Label Class - Predicted")

		# plt.show()
	return sum(iou_scores)/len(iou_scores)


def get_iou_vector(pred, gt):
	pred, gt = prepro(pred, gt)
	batch_size = pred.shape[0]
	jaccard = JaccardIndex(num_classes=2)
	metric = []
	for batch in range(batch_size):
		iou = jaccard(pred, gt)
		metric.append(iou)
	print(metric)
	return np.mean(metric).detach().numpy()

# def get_iou_vector(pred, gt):
# 	pred = pred.squeeze()
# 	gt = gt.squeeze()
# 	print(pred.shape)
# 	batch_size = pred.shape[0]
# 	pred = torch.sigmoid(pred)
# 	print(batch_size)
# 	print(pred)
# 	metric = []
# 	for batch in range(batch_size):
# 		p, t = pred[batch]>0, gt[batch]>0
# 		print(p)
# 		intersection = np.logical_and(p, t)
# 		union = np.logical_or(p, t)
# 		iou = (np.sum(intersection) + SMOOTH) / (np.sum(union) + SMOOTH)
# 		thresholds = np.arange(0.5, 1, 0.05) # 이건 왜이렇게 할까,,
# 		s = []
# 		for thresh in thresholds:
# 			s.append(iou > thresh)
# 		metric.append(np.mean(s))
# 	print(metric)
# 	return np.mean(metric)

def prepro(prd_, gt_):
	prd = prd_.squeeze().clone()
	gt = gt_.squeeze()
	# prd = torch.logsumexp(prd,0,True)
	# print("logsumexp")
	# print(prd)
	# prd = torch.sigmoid(prd)
	# print("sigmoid")
	# print(prd_)
	prd = torch.sigmoid(prd)
	print("sig")
	print(prd)
	thr = 0.4720
	prd[prd >= thr] = 1
	prd[prd < thr] = 0
	# # prd = (prd_>thr).long()
	# np.set_printoptions(threshold=448*448+1, linewidth=np.inf)

	# if len(prd_.unique() )> 2:
	# 	if len(prd.unique()) >2:
	# 		print('prd 줄어들지 않았다.')
	# 	else :
	# 		# print(prd)
	# 		# print(prd_)
	# 		print("prd 0과 1이 아니였는데 바이너리로 줄어들었다.   {} >> {:.0f}, {}//gt:{}".format(prd_.unique(), torch.sum(prd), prd.unique(), torch.sum(gt)))
	# # else :
	# # 	print("원래 바이너리")
	return prd.type(torch.int64), gt.type(torch.int64)

def np_iou(pred, gt): # 안씀
	pred[pred > 0.5] = 1
	pred[pred <= 0.5] = 0
	intersection = np.logical_and(gt, pred)
	union = np.logical_or(gt, pred)
	return (np.sum(intersection) + SMOOTH) / (np.sum(union) + SMOOTH)


def jcacard_iou(pred, gt):
	pred, gt = prepro(pred, gt)
	jaccard = JaccardIndex(num_classes=2)
	metric = jaccard(pred, gt)
	return metric.detach().numpy()

def torch_dice(pred, gt):
	pred, gt = prepro(pred, gt)
	intersection = torch.sum(torch.logical_and(pred, gt))
	union = torch.sum(pred) + torch.sum(gt)
	dice = 2.0 * (intersection + SMOOTH) / (union + SMOOTH)
	return dice.detach().numpy()


def torch_iou_uj(pred, gt):
	pred, gt = prepro(pred, gt)

	# if len(pred.unique() )> 2:
	# 	if len(pred.unique()) >2:
	# 		print('prd 줄어들지 않았다.')
	# 	else :
	# 		print("prd 0과 1이 아니였는데 바이너리로 줄어들었다.   {} >> {:.0f}, {}//gt:{}".format(len(pred.unique()), torch.sum(pred), len(pred.unique()), torch.sum(gt)))
	# else :
	# 	print("원래 바이너리")
	intersection = torch.logical_and(gt, pred)
	union = torch.logical_or(gt, pred)
	return ((torch.sum(intersection) + SMOOTH) / (torch.sum(union) + SMOOTH)).detach().numpy()


def torch_iou_fast(pred: torch.Tensor, gt: torch.Tensor):
	pred, gt = prepro(pred, gt)
	# print(gt[0][0][0])
	intersection = (pred & gt).sum((1,2)) # 한쪽이 0이면 0
	union = (pred | gt).sum((1,2)) # 양쪽다 0 이면 0
	iou = (intersection + SMOOTH) / (union + SMOOTH)
	# threshold = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10 #0.5와 비교
	return iou.mean().detach().numpy()


def uj_dice(pred: torch.Tensor, gt: torch.Tensor):
	pred, gt = prepro(pred, gt)
	intersection = (pred & gt).sum((1,2))
	union = pred.sum((1,2)) + gt.sum((1,2))
	iou = 2 * (intersection + SMOOTH) / (union + SMOOTH)
	return iou.mean().detach().numpy()

