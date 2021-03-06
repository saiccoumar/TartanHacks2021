import os
import pandas as pd 
import shutil 


path = os.path.abspath('')
print("Path: "+path)
INPUT_PATH = path
entries = os.path.join(INPUT_PATH+"/archive/Stocks")
print(os.listdir(entries))
# dummy = pd.read_csv(os.path.join(entries+'/iba.us.txt'), engine='c')
# print(dummy.iloc[-1]['Volume'])
# OUTPUT_PATH = path+"/outputs/lstm_best_7-3-19_12AM/"+iter_changes
for x in os.listdir(entries):
    print("entry:"+x)
    filePath = os.path.join(entries+ "/"+x)
    print(filePath)

    if x == ".DS_Store":
        print("yes")
    #     if os.stat(filePath).st_size > 0:
            # df_ge = pd.read_csv(filePath, engine='c')

            # print("value: ")
            # if df_ge.iloc[-1]['Volume'] == '':
            #     print("empty")
            # else:
            #     date = df_ge.iloc[-1]['Volume']
            # # year = int(date[0:4])
            #     print(date)
            # if (date>12500000):
            #         shutil.move(filePath, os.path.join(INPUT_PATH+ "/ETFInput"))

print("fin")