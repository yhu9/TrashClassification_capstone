

CONTAINER_SIZE = 1000               #size of the container
OUTPUT_SIZE = 20                    #size of the image after reading and altering the original image
N = 1                               #Number of instances of brain images
N_CLASSES = 10                       #Number of different classifications
BLUR_COEF = 1.0 / 7.0               #Blurring Coefficient
IMAGE_SIZE = 20                     #This should be the same as output size
IMAGE_DEPTH = 20                    #This should be the same as the output size
IMAGE_CHANNELS = 1                  #Number of color channels
STEPS = 10                      #Number of steps for training
BATCH_SIZE = 2                      #Batch size
LEARNING_RATE = 0.01               #Learning rate for training the CNN
OPTIMIZER = "SGD"                   #Loss function optimization algorithm. other options: {SGD,Adam,Adagrad,Ftrl,Momentum,RMSProp}
N_FEAT_FULL1 = 1024                 #Number of features output for fully connected layer1
KEEP_RATE = 0.8                     #Rate of dropping out in the dropout layer
LOG_DIR = "../ops_logs"             #Directory where the logs would be stored for visualization of the training
DIRECTORY = "../test_data/"         #Directory where the input data is stored
STEPS_RECORD = 2                   #Number of steps to record data

#Neural network constants
MNIST_INPUT        = 784
NN_INPUT        = 584
NN_HIDDEN1      = 256
NN_HIDDEN2      = 256
NN_CLASSES      = 4
NN_EPOCHS       = 1000
NN_BATCHSIZE    = 250
NN_LEARNING_RATE = 0.001               #Learning rate for training the CNN
CAT1            = "cardboard"
CAT2            = "construction"
CAT3            = "plywood"
CAT4            = "treematter"
CAT1_ONEHOT     = [1,0,0,0]
CAT2_ONEHOT     = [0,1,0,0]
CAT3_ONEHOT     = [0,0,1,0]
CAT4_ONEHOT     = [0,0,0,1]

#Neural network constants
CNN_INPUT        = 784
CNN_LAYER1 = 10                  #Number of features output for conv layer 1
CNN_LAYER2 = 4                  #Number of features output for conv layer 2
CNN_CLASSES      = 10
CNN_EPOCHS       = 10000



