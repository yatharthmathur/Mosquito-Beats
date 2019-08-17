from preprocess import *
from std import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Dense, MaxPooling2D, Flatten

def get_test_data():
	data = []
	n = list(map(int,input('Enter the list of npy files to load : ').split()))
	for i in n:
		for x in np.load('npy/test/'+str(i)+'.npy', allow_pickle=True):
			data.append(x)
	
	x_test = []
	y_test = []
	
	for x in data:
		x_test.append(x[0])
		y_test.append(x[1])

	del data
	x_test = np.asarray(x_test).reshape((len(x_test),61, 161, 1))
	y_test = np.asarray(y_test)
	
	return x_test, y_test
	

def get_data():
    #loading the features from npy files to train
    data = []
    n = list(map(int,input('Enter the list of npy files to load : ').split()))
    for i in n:
        print('Loading file {}'.format(i))
        for x in tqdm.tqdm(np.load('npy/train/'+str(i)+'.npy',allow_pickle=True)):
            data.append(x)

    #splitting the data into training and testing data
    train, test = train_test_split(data, train_size=0.8)

    #freeing the memory
    del data

    #seperating the npy tuples into x, y for both training and testing
    x_train = []
    y_train = []

    x_test = []
    y_test = []
    for x in train:
        x_train.append(x[0])
        y_train.append(x[1])
    for x in test:
        x_test.append(x[0])
        y_test.append(x[1])

    del train, test

    #converting lists to np.array and reshaping each entry into a 4D tensor with a single channel
    x_train = np.asarray(x_train).reshape((len(x_train),61, 161, 1))
    x_test = np.asarray(x_test).reshape((len(x_test),61, 161, 1))
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return x_train, y_train, x_test, y_test

#to make the CNN model layer by layer
def get_model():
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Conv2D(32, (2,2), input_shape=(len(x_train),61, 161, 1), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))

    model.add(BatchNormalization())
    model.add(Conv2D(64, (2,2), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))

    model.add(BatchNormalization())
    model.add(Conv2D(32, (2,2), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(6, activation='softmax'))
    return model

#to load saved models
def load_model(json_file,weights): #load x model json, y
    #LOAD MODEL
    from tensorflow.keras.models import model_from_json

    json = open(json_file,'r')
    model = json.read()
    model = model_from_json(model)
    model.load_weights(weights)
    print(model,'has been loaded.')
    return model

#to save trained models
def save_model(model_file,weights):
	json = model.to_json()
	with open(model_file, "w") as json_file:
		json_file.write(json)
	model.save_weights(weights)
	print(model, 'saved.')
