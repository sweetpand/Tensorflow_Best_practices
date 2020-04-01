import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


batch_size = 4
num_classes = 2
num_steps = 10
state_size = 4
learning_rate = 0.2


def gen_data(size = 1000000):
    X = np.array(np.random.choice(2,size=(size,)))
    Y = []
    for i in range(size):
        threhold = 0.5
        if X[i-3] == 1:
            threhold += 0.5
        if X[i-8] == 1:
            threhold -= 0.25
        if np.random.rand() > threhold:
            Y.append(0)
        else:
            Y.append(1)
    return X,np.array(Y)


def gen_batch(raw_data,batch_size,num_steps):

    raw_x,raw_y = raw_data
    data_x = raw_x.reshape(-1,batch_size,num_steps)
    data_y = raw_y.reshape(-1,batch_size,num_steps)
    for i in range(data_x.shape[0]):
        yield (data_x[i],data_y[i])


def gen_epochs(n):
    for i in range(n):
        yield(gen_batch(gen_data(),batch_size,num_steps))




x = tf.placeholder(tf.int32,[batch_size,num_steps],name='input_placeholder')
y = tf.placeholder(tf.int32,[batch_size,num_steps],name='output_placeholder')

init_state = tf.zeros([batch_size,state_size])

x_one_hot = tf.one_hot(x,num_classes)


rnn_inputs = tf.unstack(x_one_hot,axis=1)

cell = tf.contrib.rnn.BasicRNNCell(state_size)
#final_state
rnn_outputs,final_state = tf.contrib.rnn.static_rnn(cell,rnn_inputs,initial_state=init_state)

with tf.variable_scope("softmax"):
    W = tf.get_variable("W",[state_size,num_classes])
    b = tf.get_variable("b",[num_classes],initializer=tf.constant_initializer(0.0))

logits = [tf.matmul(rnn_output,W)+b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

y_as_list = tf.unstack(y,num=num_steps,axis=1)

loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logit)
        for (logit,label) in zip(predictions,y_as_list)
        ]

total_loss = tf.reduce_mean(loss)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def train_network(num_epochs,num_steps,state_size=4,verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx,epoch in enumerate(gen_epochs(num_epochs)):
            training_loss = 0
            training_state = np.zeros((batch_size,state_size))
            if verbose:
                print("\n EPOCH",idx)
            for step,(X,Y) in enumerate(epoch):
                tr_losses,training_loss_,training_state,_ = sess.run([loss,total_loss,final_state,train_step],
                                                                     feed_dict={x:X,y:Y,init_state:training_state})
                training_loss += training_loss_

                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step",step,"for last 100 steps:",training_loss/100)
                    training_losses.append(training_loss/100)
                training_loss = 0


        return training_losses

training_losses = train_network(5,num_steps)
plt.plot(training_losses)
plt.show()
