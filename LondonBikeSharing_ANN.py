import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
import seaborn as sns
from keras.layers import LeakyReLU
from kerastuner.tuners import RandomSearch
import shutil



def merge_and_split_columns(df):
    df['timestamp'] = pd.to_datetime(london['timestamp'], format ="%Y-%m-%d %H:%M:%S")
    
    df['is_non_workday'] = df['is_holiday'] + df['is_weekend']
    df = df.drop(['is_holiday','is_weekend'],axis=1)
         
    df['month'] = df['timestamp'].dt.month 
    df['year'] = df['timestamp'].dt.year 
    df['day']=df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week']=df['timestamp'].dt.dayofweek
         
    df['is_night'] = 0
    df.loc[(df['hour'] < 6) | (df['hour'] > 20), 'is_night'] = 1
    
    df['rush_hour'] = 1
    df.loc[((df['is_non_workday'] == 0) & ((df['hour'] == 8) | (df['hour'] == 17) | (df['hour'] == 18))) | 
            ((df['is_non_workday'] == 1) & ( (df['hour'] == 12) | (df['hour'] == 13) | (df['hour'] == 14) | (df['hour'] == 15) | 
                    (df['hour'] == 16) | (df['hour'] == 17))), 'rush_hour'] = 0
         
    return df
    


def plot_corr(df):
    f, ax = plt.subplots(figsize=(10,10))
    cmap = sns.diverging_palette(220, 110, as_cmap=True)
    sns.heatmap(df.corr(), cmap=cmap, vmax=1.0, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Korelasyon Matrisi')
    

    
def build_model(hp):
    model = keras.Sequential([
        keras.layers.Dense(n_cols, input_shape=(n_cols,)),
        keras.layers.Dense(units = hp.Int('unit1', min_value = 32, max_value = 512, step = 32),activation=LeakyReLU(alpha=0.01)),
        keras.layers.Dense(units = hp.Int('unit2', min_value = 32, max_value = 256, step = 32),activation=LeakyReLU(alpha=0.01)),
        keras.layers.Dense(units = hp.Int('unit3', min_value = 0, max_value = 128, step = 16),activation=LeakyReLU(alpha=0.01)),
        keras.layers.Dense(units = hp.Int('unit3', min_value = 0, max_value = 64, step = 16),activation=LeakyReLU(alpha=0.01)),                              
        keras.layers.Dense(1, activation='relu')        
        ])
    keras.optimizers.Nadam(hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4]))
    #keras.initializers.RandomNormal(mean=0.0, stddev=5, seed=2021)
    model.compile(optimizer='Nadam', loss='mse', metrics=['mae'])
    
    return model


class CustomTuner(RandomSearch):
  def run_trial(self, trial, *args, **kwargs):
    # You can add additional HyperParameters for preprocessing and custom training loops
    # via overriding `run_trial`
    kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 64, 256, step=64)
    kwargs['epochs'] = trial.hyperparameters.Int('epochs', 40, 60 , step=10)#min,max,step
    super(CustomTuner, self).run_trial(trial, *args, **kwargs)

def plotLoss(history):  
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.figure(figsize=(20,20))
    plt.show()

def plotCount():   
    
    Predict = random_model.predict(london_test_X)
    Target = london_test_Y.to_numpy()
    Target = Target.reshape(london_test_X.shape[0],1)
    ErrorDiff = Target - Predict
    ErrorDiff = ErrorDiff.reshape(london_test_X.shape[0],1)
    #ErrorDiffSort = sorted(ErrorDiff)
    
    plt.plot(Target)
    plt.plot(Predict)
    plt.title('CNT')
    plt.ylabel('Bike Count')
    plt.xlabel('Input')
    plt.legend(['Target','Predict'], loc='upper left')
    plt.figure(figsize=(20,20))
    plt.show()   








def split_train_and_test(london):
    splitMonth = 10

    london = shuffle(london)
    
    
    london_test1 = london[((london.month == 3) | (london.month == 6) | 
            (london.month == 9) |  (london.month == 12)) & (london.day <= splitMonth) & (london.year == 2015)]
    london_test2 = london[((london.month == 1) | (london.month == 4) | 
            (london.month == 7) |  (london.month == 10)) & ((london.day <= splitMonth * 2) & (london.day > splitMonth)) & (london.year == 2015)]
    london_test3 = london[((london.month == 2) | (london.month == 5) | 
            (london.month == 8) |  (london.month == 11)) & (london.day > splitMonth * 2) & (london.year == 2015)]
    london_test4 = london_test1.append(london_test2)
    london_test5 = london_test4.append(london_test3)
      
    london_train1 = london[((london.month == 3) | (london.month == 6) | 
            (london.month == 9) |  (london.month == 12)) & (london.day > splitMonth) & (london.year == 2015)]
    london_train2 = london[((london.month == 1) | (london.month == 4) | 
            (london.month == 7) |  (london.month == 10)) & ((london.day > splitMonth * 2) | (london.day <= splitMonth)) & (london.year == 2015)]
    london_train3 = london[((london.month == 2) | (london.month == 5) | 
            (london.month == 8) |  (london.month == 11)) & (london.day <= splitMonth * 2) & (london.year == 2015)]
    london_train4 = london_train1.append(london_train2)
    london_train5 = london_train4.append(london_train3)
    
    
    london_test6 = london[((london.month == 2) | (london.month == 5) | 
            (london.month == 8) |  (london.month == 11)) & (london.day <= splitMonth) & (london.year == 2016)]
    london_test7 = london[((london.month == 1) | (london.month == 4) | 
            (london.month == 7) |  (london.month == 10)) & ((london.day <= splitMonth * 2) & (london.day > splitMonth)) & (london.year == 2016)]
    london_test8 = london[((london.month == 3) | (london.month == 6) | 
            (london.month == 9) |  (london.month == 12)) & (london.day > splitMonth * 2) & (london.year == 2016)]
    london_test9 = london_test6.append(london_test7)
    london_test10 = london_test9.append(london_test8)
      
    london_train6 = london[((london.month == 2) | (london.month == 5) | 
            (london.month == 8) |  (london.month == 11)) & (london.day > splitMonth) & (london.year == 2016)]
    london_train7 = london[((london.month == 1) | (london.month == 4) | 
            (london.month == 7) |  (london.month == 10)) & ((london.day > splitMonth * 2) | (london.day <= splitMonth)) & (london.year == 2016)]
    london_train8 = london[((london.month == 3) | (london.month == 6) | 
            (london.month == 9) |  (london.month == 12)) & (london.day <= splitMonth * 2) & (london.year == 2016)]
    london_train9 = london_train6.append(london_train7)
    london_train10 = london_train9.append(london_train8)
    
    london_test = london_test5.append(london_test10)
    london_train = london_train5.append(london_train10)
    
    return london_train, london_test



def drop_and_categorize(london_train, london_test):
    london_test = london_test.drop(['timestamp'],axis=1)
    london_test = london_test.drop(['t1'],axis=1)
    london_test = london_test.drop(['day_of_week'],axis=1)
    
    london_train = london_train.drop(['timestamp'],axis=1)
    london_train = london_train.drop(['t1'],axis=1)
    london_train = london_train.drop(['day_of_week'],axis=1)
    
    
    london_test = london_test.drop(['year'],axis=1)
    london_train = london_train.drop(['year'],axis=1)
    
    
    london_train.weather_code = pd.Series(london_train.weather_code, dtype="category")
    london_train.month = pd.Series(london_train.month, dtype="category")
    london_train.day = pd.Series(london_train.day, dtype="category")
    london_train.hour = pd.Series(london_train.hour, dtype="category")
    london_train.is_non_workday = pd.Series(london_train.is_non_workday, dtype="bool")
    london_train.is_night = pd.Series(london_train.is_non_workday, dtype="bool")
    london_train.rush_hour = pd.Series(london_train.is_non_workday, dtype="bool")
    
    
    london_train = london_train.drop(['month'],axis=1)
    london_train = london_train.drop(['season'],axis=1)
    
    
    
    london_test.weather_code = pd.Series(london_test.weather_code, dtype="category")
    london_test.month = pd.Series(london_test.month, dtype="category")
    london_test.day = pd.Series(london_test.day, dtype="category")
    london_test.hour = pd.Series(london_test.hour, dtype="category")
    london_test.is_non_workday = pd.Series(london_test.is_non_workday, dtype="bool")
    london_test.is_night = pd.Series(london_test.is_non_workday, dtype="bool")
    london_test.rush_hour = pd.Series(london_test.is_non_workday, dtype="bool")
    
    
    
    london_test = london_test.drop(['month'],axis=1)
    london_test = london_test.drop(['season'],axis=1)
    
    london_test = shuffle(london_test)
    london_train = shuffle(london_train)
    
    return london_train, london_test
    




def split_target(london_train, london_test):
    london_train = london_train.drop(['day'],axis=1)
    london_train_Y = london_train['cnt']
    london_train_X = london_train.drop(['cnt'],axis=1)
    
    
    
    london_test = london_test.drop(['day'],axis=1)
    london_test_Y = london_test['cnt']
    london_test_X = london_test.drop(['cnt'],axis=1)
    
    return london_train_X, london_train_Y, london_test_X, london_test_Y




def scaling(london_train_X,london_test_X):
    london_train_X[['t2','hum','wind_speed']] = preprocessing.scale(london_train_X[['t2','hum','wind_speed']])
    london_test_X[['t2','hum','wind_speed']] = preprocessing.scale(london_test_X[['t2','hum','wind_speed']])       
    return london_train_X, london_test_X




london = pd.read_csv('london_merged.csv')
london = merge_and_split_columns(london)
london_train, london_test = split_train_and_test(london)
london_train, london_test = drop_and_categorize(london_train,london_test)
london_train_X, london_train_Y, london_test_X, london_test_Y = split_target(london_train,london_test)
london_train_X, london_test_X = scaling(london_train_X,london_test_X)



n_cols = london_train_X.shape[1]

shutil.rmtree('./LondonBikeSharing', ignore_errors=True)

tuner = CustomTuner(
    build_model,
    objective='val_loss',
    max_trials=100,
    executions_per_trial=3,
    directory='./',
    project_name='LondonBikeSharing')


tuner.search(london_train_X, london_train_Y,
             validation_split=0.15, callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5)])

models = tuner.get_best_models(num_models=1)



# Get the best hyperparameters from the search
random_params = tuner.get_best_hyperparameters()[0]

# Build the model using the best hyperparameters
random_model = tuner.hypermodel.build(random_params)

# Train the best fitting model
history = random_model.fit(london_train_X.values, london_train_Y.values.flatten(), epochs=75)
test_loss = random_model.evaluate(london_test_X,london_test_Y)



plotLoss(history)
plotCount()   
    
plot_corr(london)
plot_corr(london_train)


