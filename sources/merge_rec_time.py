import pickle as pk
import os
import re

def extract_matching_files(files):
    # Regular expression pattern to match files ending with _[number].rec
    pattern = re.compile(r'.*_[0-9]+\.rec$')

    # Filter files matching the specific pattern
    matched_files = [f for f in files if pattern.match(f)]
    return matched_files

all_files = os.listdir('rec_fed_sample_time/')
# print(all_files)

matched_files = extract_matching_files(all_files)

for f in matched_files:
    mf = f[:-6] + f[-4:]
    add_num = 0
    if not os.path.exists("./rec_fed_sample_time/"+mf):
        print("now we create the " + "./rec_fed_sample_time/" + mf)
        with open("./rec_fed_sample_time/"+ mf, 'wb') as fout: 
            blank_dict = {}
            pk.dump(blank_dict, fout)

    with open("./rec_fed_sample_time/"+f, 'rb') as finS:
        Sres = pk.load(finS)
    with open("./rec_fed_sample_time/"+mf, 'rb')as finM:
        Mres = pk.load(finM)

    print(f,'-->' ,mf, '({})'.format(len(Mres)))
    print(Sres)

    # print(Sres)
    for k, v in Sres.items():
        # print()
        if Mres.get(k) == None:
            Mres[k] = v
            add_num += 1
            print("add new k-v:", k, v)

    if add_num>0:
        with open("./rec_fed_sample_time/"+mf, 'wb')as finM:
            pk.dump(Mres, finM)
        print(add_num)
        # print(Mres)
    
    print(len(Mres))
        # print(Mres)

    # print(Mres)
