
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
        
def get_phone_to_class(class_phones, lst_phones, state_to_phone):
    phone_to_new_index = {}
    for index, phone in enumerate(class_phones):
        phone_to_new_index[phone] = index

    other_phones_index = len(class_phones)

    state_to_new_index = []
    for i in range(len(state_to_phone)):
        phone = state_to_phone[i]
        if phone_to_new_index.has_key(phone):
            state_to_new_index.append(phone_to_new_index[phone])
        else:
            state_to_new_index.append(other_phones_index)
        
    return state_to_new_index

phone_class_file = 'phone_code.txt'
phone_file = '/u/ndjaitly/workspace/KALDI/kaldi-trunk/egs/timit/s5/data/local/phones.txt'


state_to_phone, lst_phones = get_phone_states(phone_file)

f = open(phone_class_file)
for line in f:
    parts = line.rstrip().split()
    class_name = parts[0]
    phone_map = get_phone_to_class(parts[1:], lst_phones, state_to_phone)

    map_file_name = class_name + "_map.txt"
    print "Writing map for class %s to file %s"%(class_name, map_file_name)
    f_map = open(map_file_name, 'w')
    for ph in phone_map: f_map.write("%d\n"%ph)
    f_map.close()

f.close()
