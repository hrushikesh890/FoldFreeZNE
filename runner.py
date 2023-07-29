import rzne as rzne
from datetime import date
import csv, os

today = date.today()
today = str(today)
today = "%s-%s-%s"%(today.split('-')[0],today.split('-')[1],today.split('-')[2])
dirname = "Test_runs_decoh_depol_readout_paper_" + today
fname = dirname + "/noisemodel.pkl"
os.makedirs(os.path.dirname(fname), exist_ok=True)
rzne.save_object(rzne.noise_model, fname)

filename = dirname  + "/Hamsim.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
dirname2 = dirname + "/Hamsim_raw_files" 
for i in range(4, 8):
    for j in range(4, 12, 2):
        data = rzne.execute_hamsimul_test(i, j, dirname2)
        file = open(filename, 'a+', newline ='')
        with file:   
            write = csv.writer(file)
            write.writerow(data)

filename = dirname  + "/VQE.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
dirname2 = dirname + "/VQE_raw_files" 
for i in range(4, 8):
    for j in range(4, 12, 2):
        data = rzne.execute_VQE(i, j, dirname2)
        file = open(filename, 'a+', newline ='')
        with file:   
            write = csv.writer(file)
            write.writerow(data)

filename = dirname  + "/QAOA.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
dirname2 = dirname + "/QAOA_raw_files" 
for i in range(4, 8):
    for j in range(4, 12, 2):
        data = rzne.execute_QAOA(i, j, dirname2)
        file = open(filename, 'a+', newline ='')
        with file:   
            write = csv.writer(file)
            write.writerow(data)


filename = dirname  + "/QAOASwap.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
dirname2 = dirname + "/QAOASwap_raw_files" 
for i in range(4, 8):
    for j in range(4, 12, 2):
        data = rzne.execute_QAOA_Swap(i, j, dirname2)
        file = open(filename, 'a+', newline ='')
        with file:   
            write = csv.writer(file)
            write.writerow(data)

'''filename = dirname  + "/GHZ.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
dirname2 = dirname + "/GHZ_raw_files" 
for i in range(4, 17, 4):
    data = rzne.execute_GHZ(i, 0, dirname2)
    file = open(filename, 'a+', newline ='')
    with file:   
        write = csv.writer(file)
        write.writerow(data)

filename = dirname  + "/MerminBell.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
dirname2 = dirname + "/MerminBell_raw_files" 
for i in range(4, 8):
    data = rzne.execute_MerminBell(i, 0, dirname2)
    file = open(filename, 'a+', newline ='')
    with file:   
        write = csv.writer(file)
        write.writerow(data)'''
