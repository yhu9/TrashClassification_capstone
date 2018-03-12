

KEEP_RATE = 0.8                     #Rate of dropping out in the dropout layer
LOG_DIR = "../ops_logs"             #Directory where the logs would be stored for visualization of the training

#Neural network constants
cat1_dir = "../categories/treematter/ingroup/"
cat2_dir = "../categories/plywood/ingroup/"
cat3_dir = "../categories/cardboard/ingroup/"
cat4_dir = "../categories/construction_waste/ingroup/"
SEG_DIR = "../segments/"
CAT1            = "treematter"
CAT2            = "plywood"
CAT3            = "cardboard"
CAT4            = "construction"
CAT1_ONEHOT     = [1,0,0,0]
CAT2_ONEHOT     = [0,1,0,0]
CAT3_ONEHOT     = [0,0,1,0]
CAT4_ONEHOT     = [0,0,0,1]
LEARNING_RATE = 0.01               #Learning rate for training the CNN
CNN_LAYER1 = 4                  #Number of features output for conv layer 1
CNN_LAYER2 = 4                  #Number of features output for conv layer 2
CNN_CLASSES      = 4
CNN_EPOCHS       = 100
CNN_FULL1   = 1024                 #Number of features output for fully connected layer1
IMG_SIZE = 28
IMG_DEPTH   = 3
KEEP_RATE = 0.85
BATCH_SIZE = 100

'''
applies Dropout to the input.

Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting. The units that are kept are scaled by 1 / (1 - rate), so that their sum is unchanged at training time and inference time.
'''

