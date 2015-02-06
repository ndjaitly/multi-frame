This code is from the work in the paper: "Autoregressive product of multi-frame
predictions can improve the accuracy of hybrid models", Interspeech 2014. 
Please cite the above paper if you use this code. 

The current instructions on using this code are quite informal, and probably
not enough to get the system running. If you run into any issues, please email
me at ndjaitly@gmail.com and I will try and resolve them for you. 

In order to use this model you will need:

1. To generate the data from Kaldi. For this you will need TIMIT data, and
the code in the folder s6 provided. Place this code in <kaldi folder>/egs/timit/
and run run.sh - after changing run.sh to point to the curret TIMIT location
on your computer. After that export the alignments using 
local/export_log_fbanks_to_htk.sh in the s6 recipe folder. Remember to change
the output path in this script. Now the data is ready.

2. Add the folders cudamat, and cudamat_ext in the attached code to your PYTHON_PATH
and LD_LIBRARY_PATH. Note that cudamat is now available in Google Code. I'm 
putting the version of cudamat I used in my thesis here. If someone wants to
remove this dependency and point to the proper place, I would encourage their
contribution.

3. Train a DBN using the provided code. For this run train_dbn.py in the provided
folder with an appropriate parameter file in the params folder (there are several).

4. Train the neural network using train_nnet_mult.py using appropriate pointers
to the trained DBN model, and parameter file in the params folder.

5. Decode the results using perform_decoding_multi.py (there are several variants
in the folder). 

Cheers! 
