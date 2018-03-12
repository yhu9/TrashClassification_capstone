

CAT1            = "treematter"
CAT2            = "plywood"
CAT3            = "cardboard"
CAT4            = "construction"
CAT1_ONEHOT     = [1,0,0,0]
CAT2_ONEHOT     = [0,1,0,0]
CAT3_ONEHOT     = [0,0,1,0]
CAT4_ONEHOT     = [0,0,0,1]
LEARNING_RATE = 0.01               #Learning rate for training the CNN
CNN_INPUT        = 784
CNN_LAYER1 = 6                  #Number of features output for conv layer 1
CNN_LAYER2 = 4                  #Number of features output for conv layer 2
CNN_CLASSES      = 4
CNN_EPOCHS       = 300
CNN_FULL1   = 1024                 #Number of features output for fully connected layer1
IMG_WIDTH   = 28
IMG_HEIGHT  = 28
IMG_DEPTH   = 3
