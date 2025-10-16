import os
import sys

output= open('output.txt','r').readlines()
if len(output) != 253:
    print("[ERROR] Output file should contain 253 lines")
    sys.exit(1)


d={fname.strip():[int(euc),int(dtw)] for fname,euc,dtw in [line.split('-') for line in output]}
if len(d) != 253:
    print("[ERROR] check the format of output file. couldn't parse it.")
    sys.exit(1)
ok=1
if d['sample1.wav'] != [4,4]:
    print("[ERROR] sample1.wav prediction should be [4,4]")
    ok=False
if d['sample2.wav'] != [1,1]:
    print("[ERROR] sample2.wav prediction should be [1,1]")
    ok=False
if d['sample3.wav'] != [3,3]:
    print("[ERROR] sample3.wav prediction should be [3,3]")
    ok=False
if ok:
    print ("output.txt is in the correct format \nGood Luck!")