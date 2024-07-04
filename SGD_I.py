from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_loss(loss_train,loss_validate,validate_every):
    epochs=range(1,len(loss_train)+1)
    plt.plot(epochs,loss_train,'r',label="train_loss")
    epochs_vali=[epochs*validate_every for epochs in range(1,len(loss_validate)+1)]
    plt.plot(epochs_vali,loss_validate,'b',label="validate_loss")
    plt.title("LOSS OF SGD")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

def mse_loss(X:np.ndarray,Y:np.ndarray,W:np.ndarray):
    return 0.5*np.mean(np.square(X@W-Y))

def loss_grad(X:np.ndarray,Y:np.ndarray,W:np.ndarray):
    return X.T@(X@W-Y)/X.shape[0]

def get_batch(batch_size:int,X:np.ndarray,Y:np.ndarray):
    assert X.shape[0]%batch_size==0 , f'{X.shape[0]}%{batch_size} !=0'
    batch_num=X.shape[0]//batch_size
    X_new=X.reshape((batch_num,batch_size,X.shape[1]))
    Y_new=Y.reshape((batch_num,batch_size,))
    for i in range(batch_num):
        yield X_new[i,:,:],Y_new[i,:]

lr=0.001
num_epochs=1000
batch_size=64
validate_every=4
validate_occupy=0.2

X,Y=fetch_california_housing(return_X_y=True)
ones=np.ones(shape=(X.shape[0],1))
X=np.hstack([X,ones])
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=validate_occupy,shuffle=True)

def train(num_epochs:int,batch_size:int,validate_every:int,W0:np.ndarray,X_train:np.ndarray,X_test:np.ndarray,Y_train:np.ndarray,Y_test:np.ndarray):
   loss_train=[]
   loss_validate=[]
   W=W0
   loop=tqdm(range(num_epochs))
   for epoch in loop:
       loss_train_epoch=0
       for x_batch,y_batch in get_batch(batch_size,X_train,Y_train):
           loss_batch=mse_loss(x_batch,y_batch,W)
           loss_train_epoch+=loss_batch*x_batch.shape[0]/X_train.shape[0]
           grad_t=loss_grad(x_batch,y_batch,W)
           W=W-lr*grad_t

       loss_train.append(loss_train_epoch)
       loop.set_description(f'EPOCH:{epoch},TRAIN_LOSS:{loss_train_epoch}')

       if epoch%validate_every==0:
           loss_validate_epoch=mse_loss(X_test,Y_test,W)
           loss_validate.append(loss_validate_epoch)
           print('=============validate==============')
           print(f'Epoch:{epoch},train loss:{loss_train_epoch},val loss:{loss_validate_epoch}')
           print('===================================')

   plot_loss(loss_train,loss_validate,validate_every)



W0=np.random.random(size=(X.shape[1], ))
train(num_epochs=num_epochs, batch_size=batch_size, validate_every=validate_every, W0=W0, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
