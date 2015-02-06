from numpy import *
from pylab import *
import util

def StripeData(dataMatrix, numFramesPerData, append=False):
   if numFramesPerData == 1:
      return dataMatrix.copy()

   if append:
      left = dataMatrix[:,:(numFramesPerData/2)].copy()
      right = dataMatrix[:,-(numFramesPerData/2):].copy()
      dataMatrix = concatenate((left, dataMatrix, right), axis=1)
   
   dataMatrixStriped = zeros((dataMatrix.shape[0]*numFramesPerData, dataMatrix.shape[1]-numFramesPerData+1))
   for i in arange(0,numFramesPerData):
      dataMatrixStriped[arange(i*dataMatrix.shape[0], (i+1)*dataMatrix.shape[0]), :] = dataMatrix[:, arange(i, i+dataMatrixStriped.shape[1])].copy()
   return dataMatrixStriped

def UnStripeData(stripedMatrix, numFramesPerData):
   if numFramesPerData == 1:
      return stripedMatrix.copy()
   dataDim = stripedMatrix.shape[0]/numFramesPerData
   dataMatrix = zeros((dataDim, stripedMatrix.shape[1]+numFramesPerData-1))
   #dataMatrix = -Inf * ones((dataDim, stripedMatrix.shape[1]+numFramesPerData-1))
   dataMatrixCount = zeros((dataDim, stripedMatrix.shape[1]+numFramesPerData-1))
   for i in arange(0,numFramesPerData): 
      dataMatrix[:, arange(i, i+stripedMatrix.shape[1])] += stripedMatrix[arange(i*dataMatrix.shape[0], (i+1)*dataMatrix.shape[0]), :]
      #dataMatrix[:, arange(i, i+stripedMatrix.shape[1])] = maximum(dataMatrix[:, arange(i, i+stripedMatrix.shape[1])],stripedMatrix[arange(i*dataMatrix.shape[0], (i+1)*dataMatrix.shape[0]), :])
      dataMatrixCount[:, arange(i, i+stripedMatrix.shape[1])] += 1
   dataMatrix = dataMatrix/dataMatrixCount
   return dataMatrix

def CollateAndStripeData(dataMatrix, numFramesPerData):
   dataMatrixStriped = zeros((dataMatrix.shape[0]*numFramesPerData, dataMatrix.shape[1]-numFramesPerData+1))
   for startIndex in range(0,513,100):
      endIndex = min(513, startIndex+100)
      IDataDim = arange(startIndex, endIndex)
      dataMatrixStriped[arange(startIndex*numFramesPerData, endIndex*numFramesPerData),:]  = StripeData(dataMatrix[IDataDim,:], numFramesPerData)
   return dataMatrixStriped

def CollateAndUnStripeData(stripedMatrix, numFramesPerData):
   dataDim = stripedMatrix.shape[0]/numFramesPerData
   dataMatrix = zeros((dataDim, stripedMatrix.shape[1]+numFramesPerData-1))
   for startIndex in range(0,513,100):
      endIndex = min(513, startIndex+100)
      IDataDim = arange(startIndex, endIndex)
      dataMatrix[IDataDim,:]  = UnStripeData(stripedMatrix[arange(startIndex*numFramesPerData, endIndex*numFramesPerData),:], numFramesPerData)
   return dataMatrix

def StripeAndStrideData(dataMatrix, numFramesPerData, moveBy):
   startIndices = arange(0,dataMatrix.shape[1]-numFramesPerData+1, moveBy)
   dataMatrixStriped = zeros((dataMatrix.shape[0]*numFramesPerData, startIndices.size))
   for i in arange(0,numFramesPerData):
      dataMatrixStriped[arange(i*dataMatrix.shape[0], (i+1)*dataMatrix.shape[0]), :] = dataMatrix[:, startIndices+i]
   return dataMatrixStriped

def UnStripeAndUnStrideData(stripedMatrix, numFramesPerData, moveBy, blnAverage=True):
   dataDim = stripedMatrix.shape[0]/numFramesPerData
   numColumnsIn = stripedMatrix.shape[1]
   numColumnsOut = moveBy*(numColumnsIn-1) + numFramesPerData

   dataMatrix = zeros((dataDim, numColumnsOut))
   dataMatrixCount = zeros((dataDim, numColumnsOut))

   targetIndices = arange(0,numColumnsIn*moveBy, moveBy)
   print targetIndices.size
   print numColumnsIn

   for i in arange(0,numFramesPerData): 
      dataMatrix[:, targetIndices+i] = dataMatrix[:, targetIndices+i] + stripedMatrix[arange(i*dataDim, (i+1)*dataDim), :]
      dataMatrixCount[:, targetIndices+i] = dataMatrixCount[:, targetIndices+i] + 1
   if blnAverage == True:
      dataMatrix = dataMatrix/dataMatrixCount
   return dataMatrix


