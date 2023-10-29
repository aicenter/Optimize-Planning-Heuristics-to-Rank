import numpy as np
import copy
import random
from astarlstar import Astar_lstar
from astarlrt import Astar_lrt
from astarlgbfs import Astar_lgbfs
from gbfslstar import Gbfs_lstar
from gbfslgbfs import Gbfs_lgbfs
from gbfslrt import Gbfs_lrt
from astarl2 import Astar_l2
from gbfsl2 import Gbfs_l2
from astarbellman import Astar_bell
from gbfsbellman import Gbfs_bell
from nn_struct_new import NN
import tensorflow as tf
from tensorflow.keras.activations import softplus

nn = NN(10)

def expand_search_algs(states,search_alg,train_loss):
    #create mini batches
    if search_alg =="astar":
        for i in range(100):
            create_minibatches_astar(states,train_loss,search_alg)
    if search_alg =="gbfs":
        for i in range(100):
            create_minibatches_gbfs(states,train_loss,search_alg)

def bellman_loss(pred_cost,true_cost,child_states_x,child_states_y):
    f = tf.reshape(pred_cost, [1, -1])
    true_cost = tf.convert_to_tensor(true_cost,dtype = tf.float32)
    reg =  tf.reduce_mean(tf.maximum(0,f - 2*true_cost) + tf.maximum(0,true_cost - f))
    h = 0
    for i in range(len(child_states_x)):
        h+= tf.maximum(0,1+tf.reduce_min(nn.model([child_states_x[i],child_states_y[i]]))- pred_cost[i])
    return reg + h/len(child_states_x)

def l2_loss(y_pred,y_true):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)

def lstar_loss(h_matrix,h_r,path_cost):
    f = tf.reshape(h_r, [1, -1])+path_cost
    tf.convert_to_tensor(h_matrix,dtype = tf.float32)
    o = tf.matmul(f,h_matrix)
    soft = tf.map_fn(softplus,o)
    return tf.reduce_mean(soft, axis=-1)

def lgbfs_lrt_loss(h_matrix,h_r):
    f = tf.reshape(h_r, [1, -1])
    tf.convert_to_tensor(f,dtype = tf.float32)
    o = tf.matmul(f,h_matrix)
    soft = tf.map_fn(softplus,o)
    return tf.reduce_mean(soft, axis=-1)

optimizer = tf.keras.optimizers.Adam()


def train_minibatches(x_train,y_train,train_loss,search_alg,h_matrix=None,path_cost=None,h_val = None,child_states_x=None,child_states_y = None):
    optimizer = tf.keras.optimizers.Adam()
    with tf.GradientTape() as tape:
        logits = nn.model([x_train,y_train], training=True)
        if train_loss == "lstar":
            loss_value = lstar_loss(h_matrix, logits, path_cost)
        if train_loss == "lgbfs" or train_loss == "lrt":
            loss_value = lgbfs_lrt_loss(h_matrix, logits)
        if train_loss == 'l2':
            loss_value = l2_loss(logits,h_val)
        if train_loss == 'bellman':
            loss_value = bellman_loss(logits,h_val,child_states_x,child_states_y)
    grads = tape.gradient(loss_value, nn.model.trainable_weights)
    optimizer.apply_gradients(zip(grads, nn.model.trainable_weights))
    nn.model.save_weights("sok_"+search_alg+"_"+train_loss)

def create_minibatches_astar(coll_states,train_loss,search_alg):
    #Train for 20000 times
    for i in range(20000):
        state = coll_states[i]
        #print(state)
        goal_state=copy.deepcopy(state)
        find_box_tar = np.where(state == 2)
        box_tar=list(zip(find_box_tar[0], find_box_tar[1]))
        
        if train_loss == "lstar":
            X_Train, Y_Train, h_matrix, path_cost = Astar_lstar(state, box_tar)
            train_minibatches(X_Train,Y_Train, train_loss, search_alg,h_matrix=h_matrix, path_cost=path_cost)
        if train_loss == "lgbfs":
            X_Train, Y_Train, h_matrix = Astar_lgbfs(state, box_tar)
            train_minibatches(X_Train,Y_Train,train_loss, search_alg,h_matrix=h_matrix)
        if train_loss == "lrt":
            X_Train, Y_Train, h_matrix = Astar_lrt(state, box_tar)
            train_minibatches(X_Train,Y_Train,train_loss,search_alg,h_matrix=h_matrix)
        if train_loss == "l2":
            X_Train, Y_Train, h_val = Astar_l2(state, box_tar)
            train_minibatches(X_Train,Y_Train,train_loss,search_alg,h_val=h_val)
        if train_loss == "bellman":
            X_Train, Y_Train, child_states_x, child_states_y, h_val = Astar_bell(state, box_tar)
            train_minibatches(X_Train,Y_Train,train_loss,search_alg,child_states_x=child_states_x,child_states_y=child_states_y,h_val=h_val)
           

def create_minibatches_gbfs(coll_states,train_loss,search_alg):
    #Train for 20000 times
    for i in range(20000):
        state = coll_states[i]

        print(state,"\n")
        goal_state=copy.deepcopy(state)
        find_box_tar = np.where(state == 2)
        box_tar=list(zip(find_box_tar[0], find_box_tar[1]))
        
        if train_loss == "lstar":
            X_Train, Y_Train, h_matrix, path_cost = Gbfs_lstar(state, box_tar)
            train_minibatches(X_Train,Y_Train, train_loss, search_alg,h_matrix=h_matrix, path_cost=path_cost)
        if train_loss == "lgbfs":
            X_Train, Y_Train, h_matrix = Gbfs_lgbfs(state, box_tar)
            train_minibatches(X_Train,Y_Train,train_loss, search_alg, h_matrix=h_matrix)
        if train_loss == "lrt":
            X_Train, Y_Train, h_matrix = Gbfs_lrt(state, box_tar)
            train_minibatches(X_Train,Y_Train,train_loss,search_alg, h_matrix=h_matrix)
        if train_loss == "l2":
            X_Train, Y_Train, h_val = Gbfs_l2(state, box_tar)
            train_minibatches(X_Train,Y_Train,train_loss,search_alg,h_val=h_val)
        if train_loss == "bellman":
            X_Train, Y_Train, child_states_x, child_states_y, h_val = Gbfs_bell(state, box_tar)
            train_minibatches(X_Train,Y_Train,train_loss,search_alg,child_states_x=child_states_x,child_states_y=child_states_y,h_val=h_val)
