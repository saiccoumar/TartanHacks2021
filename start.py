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
INPUT_PATH = path + "/archive/ETFs"
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


@app.route('/')
def home():
    return render_template('index.html')

rangeSlider = 10
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
    global rangeSlider
    print("rangeSlider:" + str(rangeSlider))
    if request.method == "POST": 
        range = request.form.get("rangeValue")
        ticker = request.form.get("tickerID") 
        # ticker = "fxl"
        print(range)
        print("render")
        #############################
        stime = time.time()
        if os.path.isfile(os.path.join(INPUT_PATH, ticker+".us.txt")):
            print(os.path.join(INPUT_PATH, ticker+".us.txt"))
            df_ge = pd.read_csv(os.path.join(INPUT_PATH, ticker+".us.txt"), engine='python')
        else:
            print("failed to find file")
            return render_template('front.html', name = ticker)
       
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
        print(os.path.join(path+"/outputsETF", ticker+'.h5'))
        if os.path.isfile(os.path.join(path+ "/outputsETF", ticker+'.h5')):
            print("LOOK HERE:"+ os.path.join(path+ "/outputsETF", ticker+'.h5'))
            saved_model = load_model(os.path.join(path+ "/outputsETF", ticker+'.h5')) # , "lstm_best_7-3-19_12AM",
        else:
            print("failed to find path")
            return render_template('front.html', name = ticker + "fail")

        print(saved_model)

        y_pred = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred = y_pred.flatten()
        y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
        error = mean_squared_error(y_test_t, y_pred)
        print("Error is", error, y_pred.shape, y_test_t.shape)
        print("failed here 1")
        print(y_pred[0:15])
        print(y_test_t[0:15])
        print("failed here 3")
        y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3] # min_max_scaler.inverse_transform(y_pred)
        y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3] # min_max_scaler.inverse_transform      (y_test_t)
        print(y_pred_org[0:15])
        print(y_test_t_org[0:15])
        print("failed here 2")
        # Visualize the prediction
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        
        plt.figure()
        print("failed here 5")
        plt.plot(y_pred_org[0:150])
        plt.plot(y_test_t_org[0:150])
        plt.title('Prediction vs Real Stock Price')
        plt.ylabel('Price')
        plt.xlabel('Days')
        plt.legend(['Prediction', 'Real'], loc='upper left')
        print("failed here 6")
        #plt.show()
        # img = plt
        plt.savefig(os.path.join(path + "/static/dio"+'.jpg'))
        print_time("program completed ", stime)
        ##############################
        return render_template('front.html', name = ticker,image = "../static/dio.jpg")
    return render_template('front.html', name=name,image = "../static/dio.jpg")

if __name__ == '__main__':
	app.run(port=8000, debug=True)