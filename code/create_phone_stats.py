from numpy import *

def get_phone_states(phone_file, num_states_per_phone=3):
    f = open(phone_file, 'r')
    state_to_phone = [] 
    lst_phones = []
    for index, line in enumerate(f):
        phone = line.rstrip()
        for i in range(num_states_per_phone): state_to_phone.append(phone)
        lst_phones.append(phone)
    f.close()
    return state_to_phone, lst_phones


phone_file = '/u/ndjaitly/workspace/KALDI/kaldi-trunk/egs/timit/s5/data/local/phones.txt'
ali_file = '/ais/gobi3/u/ndjaitly/workspace/Thesis/final/alignments/TIMIT/FBANKS_E/train/ali'


state_to_phone, lst_phones = get_phone_states(phone_file)
phone_to_lengths = {} 
for phone in lst_phones: phone_to_lengths[phone] = []

f = open(ali_file)

for line in f:
    parts = line.rstrip().split()
    indices = array([int(x)/3 for x in parts[1:]],'int')
    I = flatnonzero(indices[1:] != indices[:-1])+2
    I = concatenate(([0], I, [len(parts)-1]))

    for i in range(I.size-1):
        s, e = I[i], I[i+1]
        phone_to_lengths[lst_phones[indices[s]]].append(e-s)

f.close()

for phone in lst_phones:
    phone_lengths = array(phone_to_lengths[phone])
    min_val, max_val, median_val, std_val = phone_lengths.min(), phone_lengths.max(), median(phone_lengths), phone_lengths.std()

    print "%s\t%d\t%d\t%.2f\t%.2f"%(phone, min_val, max_val, median_val, std_val)
