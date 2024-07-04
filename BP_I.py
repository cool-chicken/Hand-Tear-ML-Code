import matplotlib.pyplot as plt
import numpy as np

def load_data():
    x=np.arange(0.0,1.0,0.01)
    y=20*np.sin(2*np.pi*x)
    plt.scatter(x,y)
    return x,y

def init_parameters(layerdim):
    L=len(layerdim)
    parameters={}
    for i in range(1,L):
        parameters['w'+str(i)]=np.random.random((layerdim[i],layerdim[i-1]))
        parameters['b'+str(i)]=np.zeros((layerdim[i],1))
    return parameters

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x)*(1-sigmoid(x))
def forward(x,parameters):
    layers=len(parameters)//2
    a=[]
    z=[]
    catche={}
    a.append(x)
    z.append(x)
    for i in range(1,layers):
        z_temp=parameters['w'+str(i)]@x+parameters['b'+str(i)]
        a.append(z_temp)
        x=sigmoid(z_temp)
        z.append(x)
    z_temp=parameters['w'+str(layers)]@x+parameters['b'+str(layers)]
    a.append(z_temp)
    z.append(z_temp)
    catche['a']=a
    catche['z']=z
    return catche,z_temp

def backward(parameters,y,catche,al):
     m=y.shape[1]
     grads={}
     layers=len(parameters)//2
     grads['dz'+str(layers)]=al-y
     grads['dw'+str(layers)]=grads['dz'+str(layers)]@catche['z'][layers-1].T/m
     grads['db'+str(layers)]=np.sum(grads['dz'+str(layers)],axis=1,keepdims=True)/m
     for i in reversed(range(1,layers)):
         grads['dz' + str(i)] = parameters['w'+str(i+1)].T@grads['dz'+str(i+1)]*sigmoid_grad(
            catche['a'][i])
         grads['dw' + str(i)] = grads['dz' + str(i)]@catche['z'][i- 1].T / m
         grads['db' + str(i)] = np.sum(grads['dz' + str(i)], axis=1, keepdims=True) / m
     return grads

def  update_grads(parameters, grads, learning_rate):
    layers=len(parameters)//2
    for i in range(1,layers+1):
        parameters["w" + str(i)] -= learning_rate * grads["dw" + str(i)]
        parameters["b" + str(i)] -= learning_rate * grads["db" + str(i)]
    return parameters

def compute_loss(al,y):
    return np.mean(0.5*np.square(al-y))
x,y=load_data()
x=x.reshape(1,100)
y=y.reshape(1,100)
parameters=init_parameters([1,12,15,1])
al=0
for i in range(3600):
    catche,al=forward(x,parameters)
    grads=backward(parameters,y,catche,al)
    parameters = update_grads(parameters, grads, learning_rate=0.1)
    if i % 100 == 0:
        print(compute_loss(al, y))
plt.scatter(x,al)
plt.show()