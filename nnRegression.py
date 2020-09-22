from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


def get_test_with_data():
    # get train data
    test_with_data_data_path = 'test_with_answers.csv'
    test_with_data = pd.read_csv(test_with_data_data_path, sep=';')

    return test_with_data


def get_data():
    # get train data
    train_data_path = 'train.csv'
    train = pd.read_csv(train_data_path, sep=';')

    # get test data
    test_data_path = 'test.csv'
    test = pd.read_csv(test_data_path, sep=';')

    return train, test


def get_combined_data():
    # reading train data
    train, test = get_data()

    target = train.Target
    train.drop(['Target'], axis=1, inplace=True)

    combined = train.append(test)
    # combined.reset_index(inplace=True)
    # combined.drop(['index', 'Target'], inplace=True, axis=1)
    return combined, target


def compare_predicted_with_real(predicted, real):
    return 1 - np.absolute((predicted - real) / real)


def under_90_percent(array):
    count = 0
    for i in array:
        if (i < 0.9):
            count = count + 1

    return count / len(array)


# Load train and test data into pandas DataFrames
train_data, test_data = get_data()

# Combine train and test data to process them together
combined, target = get_combined_data()

combined["Cloudiness"] = combined["Cloudiness"].fillna(0)
combined["Cloudiness"] = combined["Cloudiness"]/100


def get_cols_with_no_nans(df, col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type :
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans
    '''

    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else:
        print('Error : choose a type (num, no_num, all)')
        return 0

    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans


num_cols = get_cols_with_no_nans(combined, 'num')
cat_cols = get_cols_with_no_nans(combined, 'no_num')

print('Number of numerical columns with no nan values :', len(num_cols))
print('Number of nun-numerical columns with no nan values :', len(cat_cols))

combined = combined[num_cols + cat_cols]
combined.hist(figsize=(12, 10))
plt.show()

train_data = train_data[num_cols + cat_cols]
train_data['Target'] = target

C_mat = train_data.corr()
fig = plt.figure(figsize=(15, 15))

sb.heatmap(C_mat, vmax=.8, square=True)
plt.show()


def oneHotEncode(df, colNames):
    for col in colNames:
        if (df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)

            # drop the encoded column
            df.drop([col], axis=1, inplace=True)
    return df


print('There were {} columns before encoding categorical features'.format(combined.shape[1]))
combined = oneHotEncode(combined, cat_cols)
print('There are {} columns after encoding categorical features'.format(combined.shape[1]))


def split_combined():
    global combined
    train = combined[:5999]
    test = combined[5999:]

    return train, test


train, test = split_combined()

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal', input_dim=train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# Compile the network:
NN_model.compile(loss='mean_absolute_error', optimizer='Adamax', metrics=['mean_absolute_error'])
NN_model.summary()

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

# NN_model.fit(train, target, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)


# Load file with real consumption results :
test_with_data = get_test_with_data()
consumption = test_with_data["Target"].to_numpy().reshape(-1, 1)

# Load the best model :
# model = callbacks_list.pop()
# NN_model = model.model
# NN_model.compile(loss='mean_absolute_error', metrics=['mean_absolute_error'])
# predictions = NN_model.predict(test)
# result = compare_predicted_with_real(predictions, consumption)

wights_file = 'Weights-Adam-430--11.37813.hdf5'  # choose the best checkpoint
NN_model.load_weights(wights_file)  # load it
NN_model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=['mean_absolute_error'])
predictions_adam = NN_model.predict(test)
result_adam = compare_predicted_with_real(predictions_adam, consumption)

wights_file = 'Weights--Nadam-348--11.30503.hdf5'  # choose the best checkpoint
NN_model.load_weights(wights_file)  # load it
NN_model.compile(loss='mean_absolute_error', optimizer='Nadam', metrics=['mean_absolute_error'])
predictions_Nadam = NN_model.predict(test)
result_Nadam = compare_predicted_with_real(predictions_Nadam, consumption)

wights_file = 'Weights-Adamax-486--10.74177.hdf5'  # choose the best checkpoint
NN_model.load_weights(wights_file)  # load it
NN_model.compile(loss='mean_absolute_error', optimizer='Adamax', metrics=['mean_absolute_error'])
predictions_Adamax = NN_model.predict(test)
np.savetxt('forecast.csv', predictions_Adamax, delimiter=';')
result_Adamax = compare_predicted_with_real(predictions_Adamax, consumption)

point1 = [0, 0.9]
point2 = [len(consumption), 0.9]
x_values = [point1[0], point2[0]]
y_values = [point1[1], point2[1]]

plot1 = plt.figure(1)
plot1.suptitle('Adam', fontsize=16)
plt.plot(result_adam)
plt.plot(x_values, y_values)
plot2 = plt.figure(2)
plot2.suptitle('NAdam', fontsize=16)
plt.plot(result_Nadam)
plt.plot(x_values, y_values)
plot3 = plt.figure(3)
plot3.suptitle('Adamax', fontsize=16)
plt.plot(result_Adamax)
plt.plot(x_values, y_values)
# plot4 = plt. figure(4)
# plot4.suptitle('result', fontsize=16)
# plt. plot(result)
plt.show()

print("Adam result")
print(under_90_percent(result_adam))
print("Nadam result")
print(under_90_percent(result_Nadam))
print("Adamax result")
print(under_90_percent(result_Adamax))
# print("Current result")
# print(under_90_percent(result))
