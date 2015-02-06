import array as ARR
import struct
import numpy
from pylab import *
import struct, os

def  CreateDeltaAndAccelerationVector(dataMatrix, deltaWindow=2):   
    ''' 
    Coefficients are eventually going to be log transformed. The deltas are computed in log space.
    '''
    sequenceLen = dataMatrix.shape[1]
    appendedVector = concatenate((tile(dataMatrix[:,0].reshape(-1,1,order='F'), (1,2*deltaWindow)), \
                       dataMatrix.copy(), \
                       tile(dataMatrix[:, -1].reshape(-1,1, order='F'), (1,deltaWindow*2))),1)

    deltaVec = zeros(appendedVector.shape)
    denom = 2*sum(arange(1,deltaWindow+1)**2)
    for i in range(1, deltaWindow+1):
        indices = arange(deltaWindow, sequenceLen+deltaWindow*3)

        deltaVecCur = appendedVector[:, indices+i] - appendedVector[:,indices-i]
        deltaVec[:,indices] = deltaVec[:,indices] + deltaVecCur * i 

    deltaVec = deltaVec/denom

    deltadeltaVec = zeros(appendedVector.shape)
    for i in range(1, deltaWindow+1):
        indices = arange(2*deltaWindow, sequenceLen+deltaWindow*2)
        deltadeltaVecCur = deltaVec[:, indices+i] - deltaVec[:,indices-i]
        deltadeltaVec[:,indices] = deltadeltaVec[:,indices] + deltadeltaVecCur * i 

    deltadeltaVec = deltadeltaVec/denom

    deltaVec = deltaVec[:,arange(2*deltaWindow,sequenceLen+2*deltaWindow)]
    deltadeltaVec = deltadeltaVec[:,arange(2*deltaWindow,sequenceLen+2*deltaWindow)]

    return deltaVec, deltadeltaVec


def ReadHTKWithDeltas(file):
    data = ReadHTK(file)
    data_d, data_a = CreateDeltaAndAccelerationVector(data)
    data = concatenate((data, data_d, data_a), axis=0)
    return data

def ReadHTK(file): 
   ''' CONVERTED VOICEBOX MATLAB CODE TO PYTHON
   READHTK  read an HTK parameter file [D,FP,DT,TC,T]=(FILE)
   
   Input:
   FILE = name of HTX file
    Outputs:
          D = data: column vector for waveforms, one row per frame for other types
         FP = frame period in seconds
         DT = data type (also includes Voicebox code for generating data)
                0  WAVEFORM     Acoustic waveform
                1  LPC          Linear prediction coefficients
               2  LPREFC       LPC Reflection coefficients:  -lpcar2rf([1 LPC]);LPREFC(1)=[];
                3  LPCEPSTRA    LPC Cepstral coefficients
                4  LPDELCEP     LPC cepstral+delta coefficients (obsolete)
                5  IREFC        LPC Reflection coefficients (16 bit fixed point)
                6  MFCC         Mel frequency cepstral coefficients
                7  FBANK        Log Fliter bank energies
                8  MELSPEC      linear Mel-scaled spectrum
                9  USER         User defined features
               10  DISCRETE     Vector quantised codebook
               11  PLP          Perceptual Linear prediction
               12  ANON
         TC = full type code = DT plus (optionally) one or more of the following modifiers
                  64  _E  Includes energy terms
                 128  _N  Suppress absolute energy
                 256  _D  Include delta coefs
                 512  _A  Include acceleration coefs
                1024  _C  Compressed
                2048  _Z  Zero mean static coefs
                4096  _K  CRC checksum (not implemented yet)
                8192  _0  Include 0'th cepstral coef
               16384  _V  Attach VQ index
               32768  _T  Attach delta-delta-delta index
          T = text version of type code e.g. LPC_C_K
  
      Thanks to Dan Ellis (ee.columbia.edu) for sorting out decompression.
   
         Copyright (C) Mike Brookes 2005
         Version: $Id: readhtk.m,v 1.9 2011/02/21 17:34:37 dmb Exp $
   
      VOICEBOX is a MATLAB toolbox for speech processing.
      Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
   
   ################################################################################
      This program is free software; you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation; either version 2 of the License, or
      (at your option) any later version.
   
      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.
   
      You can obtain a copy of the GNU General Public License from
      http://www.gnu.org/copyleft/gpl.html or by writing to
      Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
   ################################################################################
   '''
 
   fid=open(file,'rb')
   if fid < 0:
     raise Exception('Cannot read from file ' + file)


   nf=struct.unpack('>L', fid.read(4))[0]        # number of frames
   fp=struct.unpack('>L', fid.read(4))[0]*1.0e-7 # frame interval
   by=struct.unpack('>h', fid.read(2))[0]        # bytes per frame
   tc=struct.unpack('>h', fid.read(2))[0]        # type code (see comments above for interpretation)
   tc=tc+65536*(tc<0)

   cc='ENDACZK0VT'                                       #list of suffix codes
   nhb=len(cc)                                        # number of suffix codes
   ndt=6                                                 # number of bits for base type
   hb=numpy.array(floor(tc*(2.0**arange(-(ndt+nhb),(-ndt+1)))), dtype='int')
   hd=hb[arange(nhb,0,-1)]-2*hb[arange(nhb-1,-1,-1)]   # extract bits from type code
   dt=tc-hb[-1]*(2**ndt)                               # low six bits of tc represent data type
 
   #hd(7)=1 CRC check
   #hd(5)=1 compressed data
   if (dt==5):
      # hack to fix error in IREFC files which are sometimes stored as compressed LPREFC
      fid.seek(0,2)
      flen=fid.tell()        # find length of file
      fid.seek(12,0)
      if flen > 14+by*nf:
         # if file is too long (including possible CRCC) then assume compression constants exist
         dt=2               # change type to LPREFC
         hd[4]=1            # set compressed flag
         nf=nf+4            # frame count doesn't include compression constants in this case
 
   if sum(dt == 0) != 0 or sum(dt == 5) != 0 or sum(dt == 10) != 0:
      #16 bit data for waveforms, IREFC and DISCRETE
      
      binvalues = ARR.array('h')
      binvalues.read(fid, by/2 * nf)
      #d = Numeric.array(binvalues, typecode=N.Short)
      #d = N.reshape(d, (by/2, nf))
      d = numpy.array(binvalues).reshape([by/2, nf])

      if (dt == 5):
         d=d/32767                    # scale IREFC
   else:
     if hd[4] != 0:                      
        # compressed data - first read scales
        nf = nf - 4 # frame count includes compression constants
        ncol = by / 2
        binvalues = ARR.array('f')
        binvalues.read(fid, 2*ncol)
        binvalues.byteswap()
        scales = numpy.array(binvalues[0:ncol]).reshape(-1,1)
        biases = numpy.array(binvalues[ncol:]).reshape(-1,1)

        binvalues = ARR.array('h')
        binvalues.read(fid, ncol*nf)
        binvalues.byteswap()
        d = ((numpy.array(binvalues).reshape([ncol, nf], order='F')+tile(biases,[1,nf]))* tile(1./scales,[1,nf]))
     else:                            
        # uncompressed data
        binvalues = ARR.array('f')
        binvalues.read(fid, by/4*nf)
        binvalues.byteswap()
        d= numpy.array(binvalues).reshape([by/4,nf],order='F')

   fid.close()
   ns=sum(hd)                 # number of suffixes
   kinds=['WAVEFORM', 'LPC', 'LPREFC', 'LPCEPSTRA', 'LPDELCEP', 'IREFC', 'MFCC', 'FBANK', 'MELSPEC', 'USER', 'DISCRETE', 'PLP', 'ANON', '???']
   #t=kinds[min(dt,len(kinds))] 
   #print hd
   #relevant = cc[find(hd > 0)]
   #for i in range(ns):
      #t = t + '_' + relevant[i]
   #t=kinds[min(dt,len(kinds))]
   return d


def ReadBinary(fName, num_pts_per_frame=1):
    b= os.path.getsize(fName)
    #print "Size of file = ", str(b)

    fid = open(fName, 'rb')
    num_frames = b/num_pts_per_frame

    binvalues = ARR.array('f')
    binvalues.read(fid, b/4)
    fid.close()

    spec = numpy.array(binvalues).reshape((num_pts_per_frame, -1), order='F')

    return spec

def WriteHTK(file_name, data):
    dim, num_frames = data.shape
    h = struct.pack('>LLhh', # the beginning '>' says write big-endian
          num_frames,# number of frames
          100000, #samplePeriod
          4*dim,
          9) # user features
    assert len(h) == 12
    fh = file(file_name, 'wb')
    fh.write(h)
    numpy.array(data.reshape(-1,order='F'), dtype='float32').byteswap().tofile(fh)
    fh.close()

