from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.uic import loadUi
from butterworth import Butter
import sys
import cv2
import numpy as np
import scipy.signal as sig
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import math   
import numpy as np
import scipy.misc
from scipy.fftpack import dct,idct
from skimage.color import rgb2ycbcr,ycbcr2rgb
import huffman
from collections import Counter
import time

w=8 #chieu rong cua 1 block size, size co the thay doi nhung max la 8 vi bang luong tu la 8*8
w=max(2,min(8,w))
h=w #chieu cao cua 1 block size
# xLen = img.shape[1]//w #chia theo kich thuoc chieu ngang cua anh
# yLen = img.shape[0]//h #chia theo kich thuoc chieu doc cua anh
runBits=1 #modify it if you want
bitBits=3  #modify it if you want
rbBits=runBits+bitBits ##(run,bitSize of coefficient)
useYCbCr=False #co the thay doi
useHuffman=True #co the thay doi
# quantizationRatio=1 #co the thay doi, quantization table=default quantization table * quantizationRatio

def mse(ori, re):
    rows=ori.shape[1]
    cols=ori.shape[0]
    return 1/(rows*cols)*(np.sum((ori - re)**2))

def psnr(ori, re):
    mse_value = mse(ori, re)
    return 10*(np.log10(255*255/mse_value))

def myYcbcr2rgb(ycbcr):
  return (ycbcr2rgb(ycbcr).clip(0,1)*255).astype(np.uint8) #clip: nho hon 0 thi bang 0 lon hon 1 thi bang 1

def toBlocks(img): #chuyen hinh anh thanh cac khoi
  xLen = img.shape[1]//w
  yLen = img.shape[0]//h
  blocks = np.zeros((yLen,xLen,h,w,3),dtype=np.int16)
  for y in range(yLen):
    for x in range(xLen):
      blocks[y][x]=img[y*h:(y+1)*h,x*w:(x+1)*w]
  return np.array(blocks)
# blocks = toBlocks(img)

def dctOrDedctAllBlocks(blocks,xLen,yLen,type="dct"):
  f=dct if type=="dct" else idct
  dedctBlocks = np.zeros((yLen,xLen,h,w,3))
  for y in range(yLen):
    for x in range(xLen):
      d = np.zeros((h,w,3))
      for i in range(3):
        block=blocks[y][x][:,:,i]
        d[:,:,i]=f(f(block.T, norm = 'ortho').T, norm = 'ortho')
        if (type!="dct"):
          d=d.round().astype(np.int16)
      dedctBlocks[y][x]=d
  return dedctBlocks

def blocks2img(blocks,xLen,yLen): #chuyển đổi các khối DCT trở lại hình ảnh
  W=xLen*w
  H=yLen*h
  img = np.zeros((H,W,3))
  for y in range(yLen):
    for x in range(xLen):
      img[y*h:y*h+h,x*w:x*w+w]=blocks[y][x]
  return img

def quantization(qDctBlocks,quantizationRatio):
  QY=np.array([[16,11,10,16,24,40,51,61],
      [12,12,14,19,26,58,60,55],
      [14,13,16,24,40,57,69,56],
      [14,17,22,29,51,87,80,62],
      [18,22,37,56,68,109,103,77],
      [24,35,55,64,81,104,113,92],
      [49,64,78,87,103,121,120,101],
      [72,92,95,98,112,100,103,99]])
  QC=np.array([[17,18,24,47,99,99,99,99],
      [18,21,26,66,99,99,99,99],
      [24,26,56,99,99,99,99,99],
      [47,66,99,99,99,99,99,99],
      [99,99,99,99,99,99,99,99],
      [99,99,99,99,99,99,99,99],
      [99,99,99,99,99,99,99,99],
      [99,99,99,99,99,99,99,99]])
  QY=QY[:w,:h]
  QC=QC[:w,:h]
  Q3 = np.moveaxis(np.array([QY]+[QC]+[QC]),0,2)*quantizationRatio if useYCbCr else np.dstack([QY*quantizationRatio]*3)#Tất cả các kênh màu cần lượng tử hoá
  Q3=Q3*((11-w)/3)
  qDctBlocks=(qDctBlocks/Q3).round().astype('int16')    
  return qDctBlocks, Q3

def zigZag(block):
  lines=[[] for i in range(h+w-1)] 
  for y in range(h): 
    for x in range(w): 
      i=y+x 
      if(i%2 ==0): 
          lines[i].insert(0,block[y][x]) 
      else:  
          lines[i].append(block[y][x]) 
  return np.array([coefficient for line in lines for coefficient in line])

def huffmanCounter(zigZagArr):
  rbCount=[]
  run=0
  for AC in zigZagArr[1:]:
    if(AC!=0):
      AC=max(AC,1-2**(2**bitBits-1)) if AC<0 else min(AC,2**(2**bitBits-1)-1)
      if(run>2**runBits-1):
        runGap=2**runBits
        k=run//runGap
        for i in range(k):
          rbCount.append('1'*runBits+'0'*bitBits)
        run-=k*runGap
      run=min(run,2**runBits-1) 
      bitSize=min(int(math.ceil(math.log(abs(AC)+0.000000001)/math.log(2))),2**bitBits-1)
      rbCount.append(format(run<<bitBits|bitSize,'0'+str(rbBits)+'b'))
      run=0
    else:
      run+=1
  rbCount.append("0"*(rbBits))
  return Counter(rbCount)


# hiển thị run-length theo cách có thể đọc được
def runLengthReadable(zigZagArr,lastDC):
  rlc=[]
  run=0
  newDC=min(zigZagArr[0],2**(2**bitBits-1)-1)
  DC=newDC-lastDC
  bitSize=max(0,min(int(math.ceil(math.log(abs(DC)+0.000000001)/math.log(2))),2**bitBits-1))
  rlc.append([np.array(bitSize),DC])
  code=format(bitSize, '0'+str(bitBits)+'b')+"\n"
  if (bitSize>0):
    code=code[:-1]+","+(format(DC,"b") if DC>0 else ''.join([str((int(b)^1)) for b in format(abs(DC),"b")]))+"\n"
  for AC in zigZagArr[1:]:
    if(AC!=0):
      AC=max(AC,1-2**(2**bitBits-1)) if AC<0 else min(AC,2**(2**bitBits-1)-1)
      if(run>2**runBits-1):
        runGap=2**runBits
        k=run//runGap
        for i in range(k):
          code+='1'*runBits+'0'*bitBits+'\n'
          rlc.append([runGap-1,0])
        run-=k*runGap
      bitSize=min(int(math.ceil(math.log(abs(AC)+0.000000001)/math.log(2))),2**bitBits-1)
      #VLI encoding (next 2 lines of codes)
      code+=format(run<<bitBits|bitSize,'0'+str(rbBits)+'b')+','
      code+=(format(AC,"b") if AC>=0 else ''.join([str((int(b)^1)) for b in format(abs(AC),"b")]))+"\n"
      rs=np.zeros(1,dtype=object)
      rs[0]=np.array([run,bitSize])
      rs= np.append(rs,AC)
      rlc.append(rs)
      run=0
    else:
      run+=1
  rlc.append([0,0])
  code+="0"*(rbBits)#end
  return np.array(rlc),code,newDC

def runLength(zigZagArr,lastDC,hfm=None):
  rlc=[]
  run=0
  newDC=min(zigZagArr[0],2**(2**bitBits-1))
  DC=newDC-lastDC
  bitSize=max(0,min(int(math.ceil(math.log(abs(DC)+0.000000001)/math.log(2))),2**bitBits-1))
  code=format(bitSize, '0'+str(bitBits)+'b')
  if (bitSize>0):
   code+=(format(DC,"b") if DC>0 else ''.join([str((int(b)^1)) for b in format(abs(DC),"b")]))
  for AC in zigZagArr[1:]:
    if(AC!=0):
      AC=max(AC,1-2**(2**bitBits-1)) if AC<0 else min(AC,2**(2**bitBits-1)-1)
      if(run>2**runBits-1):
        runGap=2**runBits
        k=run//runGap
        for i in range(k):
          code+=('1'*runBits+'0'*bitBits)if hfm == None else  hfm['1'*runBits+'0'*bitBits]#end
        run-=k*runGap
      run=min(run,2**runBits-1) 
      bitSize=min(int(math.ceil(math.log(abs(AC)+0.000000001)/math.log(2))),2**bitBits-1)
      rb=format(run<<bitBits|bitSize,'0'+str(rbBits)+'b') if hfm == None else hfm[format(run<<bitBits|bitSize,'0'+str(rbBits)+'b')]
      code+=rb+(format(AC,"b") if AC>=0 else ''.join([str((int(b)^1)) for b in format(abs(AC),"b")]))
      run=0
    else:
      run+=1
  code+="0"*(rbBits) if hfm == None else  hfm["0"*(rbBits)]#end
  return code,newDC

def runLength2bytes(code):
  return bytes([len(code)%8]+[int(code[i:i+8],2) for i in range(0, len(code), 8)])

def huffmanCounterWholeImg(blocks,xLen,yLen):
  rbCount=np.zeros(xLen*yLen*3,dtype=Counter)
  zz=np.zeros(xLen*yLen*3,dtype=object)
  for y in range(yLen):
    for x in range(xLen):
      for i in range(3):
        zz[y*xLen*3+x*3+i]=zigZag(blocks[y, x,:,:,i])
        rbCount[y*xLen*3+x*3+i]=huffmanCounter(zz[y*xLen*3+x*3+i])
  return np.sum(rbCount),zz

def savingQuantizedDctBlocks(blocks,xLen,yLen,img):
  rbCount,zigZag=huffmanCounterWholeImg(blocks,xLen,yLen)
  hfm=huffman.codebook(rbCount.items())
  sortedHfm=[[hfm[i[0]],i[0]] for i in rbCount.most_common()]
  code=""
  DC=0
  for y in range(yLen):
    for x in range(xLen):
      for i in range(3):
        codeNew,DC=runLength(zigZag[y*xLen*3+x*3+i],DC,hfm if useHuffman else None)
        code+=codeNew
  savedImg=runLength2bytes(code)
  # print(str(code[:100])+"......")
  # print(str(savedImg[:20])+"......")
  # print("Image original size:    %.3f MB"%(img.size/(2**20)))
  # print("Compression image size: %.3f MB"%(len(savedImg)/2**20))
  # print("Compression ratio:      %.2f : 1"%(img.size/2**20/(len(savedImg)/2**20)))
  return bytes([int(format(xLen,'012b')[:8],2),int(format(xLen,'012b')[8:]+format(yLen,'012b')[:4],2),int(format(yLen,'012b')[4:],2)])+savedImg,sortedHfm

class UI(QMainWindow):
  def __init__(self):
    super(UI, self).__init__()
    loadUi("demo.ui", self)
    self.setWindowIcon(QtGui.QIcon("python-icon.png"))

    self.pre_img = None
    self.origin_img = None
    self.compress_img = None
    self.dctBlocks = None
    self.qDctBlocks = None
    self.numberqtz = 0
    self.image = None
    self.xLen = 0
    self.yLen = 0
    self.quantizationRatio = 1

    #set input button
    # self.setQratio.setText(str(self.quantizationRatio))
    self.chooseImage.clicked.connect(self.open_img)
    self.DCT.clicked.connect(self.computeDCT)
    self.quantization.clicked.connect(self.computeqDCT)
    self.Decompress.clicked.connect(self.computeDecompress)
    self.Reset.clicked.connect(self.computeReset)
    self.buttonOK.clicked.connect(self.setRatio)

  @pyqtSlot()
  def loadImage(self, fname):
    self.image = cv2.imread(fname)
    self.xLen = self.image.shape[1]//w
    self.yLen = self.image.shape[0]//h
    self.origin_img = cv2.resize(self.image,(411,391))
    self.tmp = self.image
    self.displayImage()
  
  def displayImage(self, window=1):
    qformat = QImage.Format_Indexed8

    if len(self.origin_img.shape) == 3:
        if(self.origin_img.shape[2]) == 4:
            qformat = QImage.Format_RGBA8888
        else:
            qformat = QImage.Format_RGB888
    img = QImage(self.origin_img, self.origin_img.shape[1], self.origin_img.shape[0], self.origin_img.strides[0], qformat)
    # image.shape[0] là số pixel theo chiều Y
    # image.shape[1] là số pixel theo chiều X
    # image.shape[2] lưu số channel biểu thị mỗi pixel
    img = img.rgbSwapped() # chuyển đổi hiệu quả một ảnh RGB thành một ảnh BGR.
    if window == 1:
        self.pre_frame.setPixmap(QPixmap.fromImage(img))
        self.pre_frame.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)# căn chỉnh vị trí xuất hiện của hình trên lable
    if window == 2:
        self.aft_frame.setPixmap(QPixmap.fromImage(img))
        self.aft_frame.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

  def displayPreImage(self, window=1):
    qformat = QImage.Format_Indexed8

    if len(self.pre_img.shape) == 3:
        if(self.pre_img.shape[2]) == 4:
            qformat = QImage.Format_RGBA8888
        else:
            qformat = QImage.Format_RGB888
    img = QImage(self.pre_img, self.pre_img.shape[1], self.pre_img.shape[0], self.pre_img.strides[0], qformat)
    # image.shape[0] là số pixel theo chiều Y
    # image.shape[1] là số pixel theo chiều X
    # image.shape[2] lưu số channel biểu thị mỗi pixel
    img = img.rgbSwapped() # chuyển đổi hiệu quả một ảnh RGB thành một ảnh BGR.
    if window == 1:
        self.pre_frame.setPixmap(QPixmap.fromImage(img))
        self.pre_frame.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)# căn chỉnh vị trí xuất hiện của hình trên lable
    if window == 2:
        self.aft_frame.setPixmap(QPixmap.fromImage(img))
        self.aft_frame.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


  def open_img(self):
    fname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'This PC', "Image Files (*)")
    if fname:
        self.loadImage(fname)
    else:
        print("Invalid Image")   

  def computeDCT(self):
    img = self.image
    ycbcr=rgb2ycbcr(img) #chuyen hinh anh sang YCbCr
    rgb=myYcbcr2rgb(ycbcr) #chuyen hinh anh ve rgb
    if (useYCbCr):
      img=ycbcr
    blocks = toBlocks(img)
    self.dctBlocks=dctOrDedctAllBlocks(blocks,self.xLen, self.yLen,"dct")
    newImg=blocks2img(self.dctBlocks,self.xLen,self.yLen)
    newImg = cv2.resize(newImg,(411,391))
    cv2.imwrite("newImg.jpg",newImg)
    self.pre_img = cv2.imread('newImg.jpg')
    self.displayPreImage(2)

  def computeqDCT(self):
    self.qDctBlocks, self.numberqtz = quantization(self.dctBlocks,self.quantizationRatio) 
    qDctImg=blocks2img(self.qDctBlocks,self.xLen,self.yLen).astype('int16')  
    newQdct = cv2.resize(qDctImg,(411,391))
    cv2.imwrite("newQdct.jpg",newQdct)
    self.pre_img = cv2.imread('newQdct.jpg')
    self.displayPreImage(2)

  def computeDecompress(self):
    dedctBlocks=dctOrDedctAllBlocks(self.qDctBlocks*self.numberqtz,self.xLen, self.yLen,"idct")
    pre_img = myYcbcr2rgb(blocks2img(dedctBlocks,self.xLen, self.yLen)) if useYCbCr else blocks2img(dedctBlocks,self.xLen, self.yLen).astype(np.int16)
    self.compress_img = pre_img
    pre_img = cv2.resize(pre_img,(411,391))
    cv2.imwrite("pre_img.jpg",pre_img)
    self.pre_img = cv2.imread('pre_img.jpg')
    self.displayPreImage(2)

    t1 = time.time()
    savedImg,sortedHfmForDecode=savingQuantizedDctBlocks(self.qDctBlocks,self.xLen, self.yLen, self.image)
    t2=time.time()

    save = open("img.bin", "wb")
    save.write(savedImg)
    save.close()

    origin_img = self.image.size/(2**20)
    textOriginImg = "{:.3f} MB".format(origin_img)
    self.disSizePre.setText(textOriginImg)

    compress_img = len(savedImg)/2**20
    textCompressImg = "{:.3f} MB".format(compress_img)
    self.disSizeCompress.setText(textCompressImg)

    ratio = self.image.size/2**20/(len(savedImg)/2**20)
    textRatio = "{:.2f} : 1".format(ratio)
    self.disRatio.setText(textRatio)

    Encoding = t2-t1
    textEncoding = "{:.3f}".format(Encoding)
    self.disTime.setText(textEncoding)

    MSE = mse(self.image, self.compress_img)
    textMSE = "{:.3f}".format(MSE)
    self.disMSE.setText(textMSE)

    PSRN = psnr(self.image, self.compress_img)
    textPSRN = "{:.3f}".format(PSRN)
    self.disPSNR.setText(textPSRN)

  def setRatio(self):
    self.quantizationRatio = int(self.setQratio.text())

  def computeReset(self):
    self.pre_frame.clear()
    self.aft_frame.clear()
    self.disSizePre.setText("")
    self.disSizeCompress.setText("")
    self.disRatio.setText("")
    self.disTime.setText("")
    self.disMSE.setText("")
    self.disPSNR.setText("")

app = QApplication(sys.argv)
win = UI()
win.show()
sys.exit(app.exec())