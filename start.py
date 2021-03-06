from flask import Flask, request, render_template
from matplotlib import pyplot as plt
import numpy as np
import os
import os.path
from os import path
import sys
import time
import pandas as pd 
#from tqdm.notebook import tqdm_notebook
from tqdm import tqdm
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

params = {
    "batch_size": 20,  # 20<16<10, 25 was a bust
    "epochs": 100,
    "lr": 0.00010000,
    "time_steps": 60
}

iter_changes = "dropout_layers_0.4_0.4"

path = os.path.abspath('')
print("Path: "+path)
INPUT_PATH = path + "/archive/data"
OUTPUT_PATH = path
TIME_STEPS = params["time_steps"]
BATCH_SIZE = params["batch_size"]
stime = time.time()

def print_time(text, stime):
    seconds = (time.time()-stime)
    print(text, seconds//60,"minutes : ",np.round(seconds%60),"seconds")
    
def trim_dataset(mat,batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat
templateDir = path+"/templates"
statDir = path +"/static"
print(templateDir)
print(statDir)
app = Flask(__name__,template_folder=templateDir,static_folder=statDir)
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def build_timeseries(mat, y_col_index):
    """
    Converts ndarray into timeseries format and supervised data format. Takes first TIME_STEPS
    number of rows as input and sets the TIME_STEPS+1th data as corresponding output and so on.
    :param mat: ndarray which holds the dataset
    :param y_col_index: index of column which acts as output
    :return: returns two ndarrays-- input and output in format suitable to feed
    to LSTM.
    """
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    print("dim_0",dim_0)
    for i in tqdm(range(dim_0)):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
#         if i < 10:
#           print(i,"-->", x[i,-1,:], y[i])
    print("length of time-series i/o",x.shape,y.shape)
    return x, y
# disable cacheing
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
@app.route('/')
def home():
    return render_template('index.html')


"""@app.route('/ticker', methods=["GET","POST"])
def sliderForm():
    if request.method == "POST": 
        global rangeSlider
        rangeSlider = request.form.get("myRange")"""


@app.route('/ticker', methods=["GET","POST"])
def tickerForm(name="ge"):
    global path
    global INPUT_PATH
    global OUTPUT_PATH
    global TIME_STEPS
    global BATCH_SIZE
    global stime
    global iter_changes
    global params
    if request.method == "POST": 
        rangeVar = request.form.get("rangeValue")
        ticker = request.form.get("tickerID") 
        # ticker = "fxl"
        print("HOT WATER: "+rangeVar)
        print("render")
        rangeVar = float(rangeVar)
        rangeVar = rangeVar/100
        arr = {}
        for i in range (0,10):
            arr[i] = 0
        #############################
        stime = time.time()
        if os.path.isfile(os.path.join(INPUT_PATH, ticker+".us.txt")):
            print(os.path.join(INPUT_PATH, ticker+".us.txt"))
            df_ge = pd.read_csv(os.path.join(INPUT_PATH, ticker+".us.txt"), engine='python')
        else:
            print("failed to find file")

            return render_template('front.html', name = "failed to find image", image =  "/static/safestocks.jpg", infoKeys = list(arr.keys()),infoValues = list(arr.values()),maximum = "none", minimum = "none",mess = "No Ticker Found", volatility = 0)
       
        print(df_ge.shape)
        print(df_ge.columns)
        print(df_ge.head(5))
        #tqdm_notebook.pandas('Processing...')
        # df_ge = process_dataframe(df_ge)
        #print(df_ge.dtypes)
        train_cols = ["Open","High","Low","Close","Volume"]
        df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
        print("Train--Test size", len(df_train), len(df_test))

        x = df_train.loc[:,train_cols].values
        min_max_scaler = MinMaxScaler()
        x_train = min_max_scaler.fit_transform(x)
        x_test = min_max_scaler.transform(df_test.loc[:,train_cols])

        #print("Deleting unused dataframes of total size(KB)",(sys.getsizeof(df_ge)+sys.getsizeof(df_train)+sys.getsizeof(df_test))//1024)

        #del df_ge
        #del df_test
        #del df_train
        #del x

        #print("Are any NaNs present in train/test matrices?",np.isnan(x_train).any(), np.isnan(x_train).any())
        #x_t, y_t = build_timeseries(x_train, 3) ##
        #x_t = trim_dataset(x_t, BATCH_SIZE)
        #y_t = trim_dataset(y_t, BATCH_SIZE)
        #print("Batch trimmed size",x_t.shape, y_t.shape)

        x_temp, y_temp = build_timeseries(x_test, 3)
        print("done with  buildtimeseries")
        x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
        y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)
        print(path)
        print(os.path.join(path+"/outputs/", ticker+'.h5'))
        if os.path.isfile(os.path.join(path+ "/outputs/", ticker+'.h5')):
            print("LOOK HERE:"+ os.path.join(path+ "/outputs/", ticker+'.h5'))
            saved_model = load_model(os.path.join(path+ "/outputs/", ticker+'.h5')) # , "lstm_best_7-3-19_12AM",
        else:
            print("failed to find path")
            return render_template('front.html', name = "failed to load image", image =  "/static/safestocks.jpg", infoKeys = list(arr.keys()),infoValues = list(arr.values()), maximum = "none", minimum = "none",mess = "No Ticker Found",  volatility = 0)

        print(saved_model)

        y_pred = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred = y_pred.flatten()
        y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
        error = mean_squared_error(y_test_t, y_pred)
        print("Error is", error, y_pred.shape, y_test_t.shape)
        print("failed here 1")
        print(len(y_pred-1))
        rangeVar = int(rangeVar * float((len(y_pred-1))))
        print("MNM"+str(rangeVar))
        print(y_pred[0:rangeVar])
        print(y_test_t[0:rangeVar])
        print("failed here 3")
        y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3] -0.5   # min_max_scaler.inverse_transform(y_pred)
        y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3] # min_max_scaler.inverse_transform      (y_test_t)
        print(y_pred_org[0:rangeVar])
        print(y_test_t_org[0:rangeVar])
        print("failed here 2")
        # Visualize the prediction
        import matplotlib
        matplotlib.use('Agg')
        arr = {}
        volatilityIndex = 0
        from matplotlib import pyplot as plt
        # for i in range(0,rangeVar):
            # volatilityIndex = volatilityIndex + (y_pred_org[i])
        # sum(y_pred_org)/len(y_pred_org)
        maxROC = -999
        recordedDate = 0 
        message = ""
        for i in range(0, rangeVar,int(rangeVar/10)):
            arr[i] = y_pred_org[i]
            # temp = (y_pred_org[i+1])-(y_pred_org[i]) 
            # print(temp)
            # volatilityIndex = volatilityIndex + (y_pred_org[i])
            # if maxROC < ((temp)/y_pred_org[i+1]):
            #     maxROC = ((temp)/y_pred_org[i+1])
            #     print(maxROC)
            #     recordedDate = i
        print("all values:"+str(volatilityIndex))
        volatilityIndex = sum(y_pred_org)/len(y_pred_org)
        print("mean:"+str(volatilityIndex))
        squaredDeviations = 0
        for i in range(0, rangeVar,int(rangeVar/10)):
            print((((volatilityIndex)-y_pred_org[i])))
            squaredDeviations = squaredDeviations + (((volatilityIndex)-y_pred_org[i])*((volatilityIndex)-y_pred_org[i]))
        print("sqrt"+str(squaredDeviations))
        volatilityIndex = int(squaredDeviations/9)
        print("volatility:"+str(volatilityIndex))
        message = "We recommend that you buy between days " + str(recordedDate) + " and " + str(recordedDate+100)
        # print(list(arr.keys())[0])
        min = arr[(list(arr.keys())[0])]
        max = arr[(list(arr.keys())[0])]
        print(min)
        for i in list(arr.values()):
            if i > max:
                max = i
            if i < min:
                min = i

        print(max)
        print(min)
        print(arr)
        plt.figure()
        print("failed here 5")
        
        
        plt.plot(y_pred_org[0:rangeVar])
        plt.plot(y_test_t_org[0:rangeVar])
        plt.title('Prediction vs Real Stock Price')
        plt.ylabel('Price($)')
        plt.xlabel('Days')
        plt.legend(['Prediction', 'Real'], loc='upper left')
        print("failed here 6")
        #plt.show()
        # img = plt
        plt.savefig(os.path.join("static/plot"+'.png'))
        print_time("program completed ", stime)
        ##############################
        
        return render_template('front.html', name = ticker,image = "/static/plot.png", infoKeys = list(arr.keys()),infoValues = list(arr.values()),maximum = max, minimum = min,mess = message, volatility = volatilityIndex)
    arr = {}
    for i in range (0,10):
        arr[i] = 0
    return render_template('front.html', name="None selected",image = "/static/safestocks.jpg", infoKeys = list(arr.keys()),infoValues = list(arr.values()), maximum = "none", minimum = "none",mess = "none", volatility = 0)

if __name__ == '__main__':
	app.run(port=8000, debug=True)