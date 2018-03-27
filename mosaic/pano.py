#-*- conding:utf-8 -*-
import cv2
import sys
from matchers import matchers
import time
from matplotlib import pyplot as plt
from numpy import *

class Stitch:
	def __init__(self):
		#可以使用文件记录好图像路径，再读取。
		# self.path = args
		# fp = open(self.path, 'r')
		# filenames = [each.rstrip('\r\n') for each in  fp.readlines()]

		IMGFILEPATH='Mosaic_image/'
		filenames = [('img01/image-001-0'+str(i+1)+'.bmp' ) for i in range(4)]
		self.images = [cv2.resize(cv2.imread(IMGFILEPATH+each,0),(192, 192)) for each in filenames ] #480,320
		# self.images = [cv2.imread(each,0) for each in filenames[:2]] 
	
		self.count = len(self.images)
		self.img_list = []
		self.matcher_obj = matchers()
		self.prepare_lists()

		self.call_times=0  #匹配函数imgshift被调用的次数
		self.mach_number=0 #可以匹配上的图片数量

	def prepare_lists(self):
		for i in range(self.count): self.img_list.append(self.images[i])
			

	def imgshift(self):
		no_match_list=[] #不能匹配的图像列表
		a = self.img_list[0]
		tmp=a
		for b in self.img_list[1:]:
			H = self.matcher_obj.match(a, b, 'left')
			# print ('a,b size:',a.shape,b.shape)
			if  H is None :
				no_match_list.append(b)
				# print ('H:',H)
				continue
			#归一化
			xh = linalg.inv(H)  #矩阵求逆
			ds = dot(xh, array([a.shape[1], a.shape[0], 1]))
			ds = ds/ds[-1]
			f1 = dot(xh, array([0,0,1]))
			f1 = f1/f1[-1]
			xh[0][-1] += abs(f1[0])
			xh[1][-1] += abs(f1[1])
			ds = dot(xh, array([a.shape[1], a.shape[0], 1]))
			offsety = abs(int(f1[1]))
			offsetx = abs(int(f1[0]))
			dsize = (abs(int(ds[0]))+offsetx, abs(int(ds[1])) + offsety)
			# print ('f1:',f1)
			print("image dsize =>", dsize)
			tmp = cv2.warpPerspective(a, xh, dsize)
			if (b.shape[0]+offsety) > tmp.shape[0] or (b.shape[1]+offsetx) >tmp.shape[1]:
				print('Not matching!!!')
				no_match_list.append(b)
				continue

			#拼接
			tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b
			#平滑接缝处。高斯金字塔
			# tmp=self.reconstruct(tmp,b,offsetx,offsety)

			print('Matching!!!')
			self.mach_number +=1
			a = tmp

		#去掉大部分黑色背景
		u,d,l,r=self.imgCutting(a)
		self.leftImage = a[u:d,l:r]
		print ('nomatchimg len:',len(no_match_list))
		#如若有剩余没匹配上的，继续匹配，最多循环匹配5次
		if len(no_match_list) and self.call_times<5:  
			# print ('times:',self.call_times)
			self.call_times +=  1
			self.img_list=[]
			self.img_list.append(self.leftImage)
			[ self.img_list.append(img) for img in no_match_list ]
			self.imgshift()


	#去掉大部分背景
	def imgCutting(self,img):
		h,w=img.shape
		#求每行每列像素和，背景区域像素和为0
		per_row_sum = sum(img,axis=1)
		per_col_sum = sum(img,axis=0)
		id_up=per_row_sum.tolist().index(per_row_sum[per_row_sum>0][0])
		id_dw=per_row_sum.tolist().index(per_row_sum[per_row_sum>0][::-1][0])
		id_lef=per_col_sum.tolist().index(per_col_sum[per_col_sum>0][0])
		id_rig=per_col_sum.tolist().index(per_col_sum[per_col_sum>0][::-1][0])

		#调整 ：
		if id_up > 0  : id_up  -= 1
		if id_dw < h-1: id_dw  += 1
		if id_lef> 0  : id_lef -= 1
		if id_rig< w-1: id_rig += 1

		return id_up,id_dw,id_lef,id_rig


	#进行融合 平滑接缝处，金字塔方法  不太理想,可能处理方法不对 ，融合内容可以参看opencv 官方文档，苹果和橘子融合
	def reconstruct(self,A,B,offsetx,offsety):
		print ('offsetx,offsety:',offsetx,offsety)
		#generate Gaussian pyramid for A
		G=A.copy()
		gpA=[G]
		for i in range(6):
			G = cv2.pyrDown(G)
			gpA.append(G)

		#generate Gaussian pyramid for B
		G = B.copy()
		gpB=[G]
		for i in range(6):
			G = cv2.pyrDown(G)
			gpB.append(G)

		#generate Laplacian Pyraid for B
		lpA=[gpA[5]]
		for i in range(5,0,-1):
			GE = cv2.pyrUp(gpA[i])
			minx,miny=min(gpA[i-1].shape[0],GE.shape[0]),min(gpA[i-1].shape[1],GE.shape[1])
			L=cv2.subtract(gpA[i-1][:minx,:miny],GE[:minx,:miny])
			lpA.append(L)

		#generate Laplacian Pyraid for B
		lpB=[gpB[5]]
		for i in range(5,0,-1):
			GE = cv2.pyrUp(gpB[i])
			minx,miny=min(gpA[i-1].shape[0],GE.shape[0]),min(gpA[i-1].shape[1],GE.shape[1])
			L=cv2.subtract(gpB[i-1][:minx,:miny],GE[:minx,:miny])
			lpB.append(L)

		# Now add left and right halves of images in each level
		#numpy.hstack(tup)
		#Take a sequence of arrays and stack them horizontally
		#to make a single array.
		LS = []
		l=array([1,2,4,8,16,32,64][::-1][:6])
		ofx_list= offsetx//l
		ofy_list= offsety//l

		for la,lb,ofy,ofx in zip(lpA,lpB,ofx_list,ofy_list):
			la[ofy:lb.shape[0]+ofy,ofx:lb.shape[1]+ofx] = lb
			LS.append(la)

		#now reconstruct
		ls_=LS[0]
		for i in range(1,6):
			ls_ = cv2.pyrUp(ls_)
			minx,miny=min(ls_.shape[0],LS[i].shape[0]),min(ls_.shape[1],LS[i].shape[1])
			ls_ = cv2.add(ls_[:minx,:miny],LS[i][:minx,:miny])

		return ls_



if __name__ == '__main__':
	now=time.time()
	# try:
	# 	args = sys.argv[1]  #sys.argv 用来获取命令行参数，sys.argv[0]代表本身文件路径，说一参数从1开始
	# except:
	# 	args = "txtlists/files4.txt"
	# finally:
	# 	print("Parameters : ", args)
	# s = Stitch(args)

	s = Stitch()
	s.imgshift()
	print ('running time:',time.time()-now)
	
	if s.mach_number ==0:
		print ('No matchers')
	else:
		# s.leftImage=cv2.resize(s.leftImage,(192,192))
		cv2.imwrite("mosic.bmp", s.leftImage)

		print('Have %d images to mosic'%s.mach_number)
		plt.imshow(cv2.imread('mosic.bmp',0),cmap='gray')
		plt.show()
	

	