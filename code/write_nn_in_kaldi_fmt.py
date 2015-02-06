import argparse, os
import nnet_train, nnet_layers
import array
def WriteLayer(fid, nnet, layer_num):
    nn_layer, layer_type = nnet._lst_layers[layer_num], \
                                       nnet._lst_layer_type[layer_num]
    if layer_type == nnet_layers.SIGMOID_LAYER:
        fid.write("<sigmoid>")
    elif layer_type == nnet_layers.SOFTMAX_LAYER:
        fid.write("<softmax>")
    else:
        raise Exception, "Unknown layer type"

    wts = nn_layer._wts.asarray()
    biases = nn_layer._b.asarray()

    num_dims, num_hid = wts.shape
    arr = array.array('l')
    arr.append(num_hid)
    arr.append(num_dims)
    arr.tofile(fid)
    arr_biases = array.array('l')
    arr_biases.append(num_hid)

    if wts.dtype == 'float32':
        fid.write('FM ')
        # need to write it twice
        arr.tofile(fid)
        wts.transpose().tofile(fid)
    else:
        fid.write('FM ')
        # need to write it twice
        arr.tofile(fid)
        array(wts.transpose(), 'float32').tofile(fid)

    if biases.dtype == 'float32':
        fid.write('FV ')
        arr_biases.tofile(fid)
        biases.tofile(fid)
    else:
        fid.write('FV ')
        arr_biases.tofile(fid)
        array(biases, 'float32').tofile(fid)

parser = argparse.ArgumentParser(description='Write my neural network' + \
                         ' parameters into Kaldi format. Will be saved' + \
                         ' in model folder')
parser.add_argument('model_fldr', help='my neural network file')

arguments = parser.parse_args()

model_file = os.path.join(arguments.model_fldr, "model.dat")
output_file = os.path.join(arguments.model_fldr, "kaldi_model.dat")

nnet_model = nnet_train.nn()
nnet_model.load(model_file)
import pdb
fid = open(output_file, 'wb')
for layer_num in range(nnet_model.get_num_layers()):
    WriteLayer(fid, nnet_model, layer_num)
fid.close()
