import tensorflow as tf
import numpy as np
from collections import deque
import random
import datetime


class DeepQNetwork:
    r_list = np.array([[-100,0,-100,-100,-100,0,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100],
[0,-100,0,-100,-100,-100,0,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100],
[-100,0,-100,100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100],
[-100,-100,0,100,0,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100],
[-100,-100,-100,100,-100,-100,-100,-100,-100,0,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100],
[0,-100,-100,-100,-100,-100,0,-100,-100,-100,0,-100,-100,-100,-100,-100,-100,-100,-100,-100],
[-100,0,-100,-100,-100,0,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100],
[-100,-100,0,-100,-100,-100,0,-100,0,-100,-100,-100,0,-100,-100,-100,-100,-100,-100,-100],
[-100,-100,-100,100,-100,-100,-100,-100,-100,0,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100],
[-100,-100,-100,-100,0,-100,-100,-100,0,-100,-100,-100,-100,-100,0,-100,-100,-100,-100,-100],
[-100,-100,-100,-100,-100,0,-100,-100,-100,-100,-100,-100,-100,-100,-100,0,-100,-100,-100,-100],
[-100,-100,-100,-100,-100,-100,0,-100,-100,-100,0,-100,0,-100,-100,-100,0,-100,-100,-100],
[-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,0,-100,-100],
[-100,-100,-100,-100,-100,-100,-100,-100,0,-100,-100,-100,0,-100,0,-100,-100,-100,0,-100],
[-100,-100,-100,-100,-100,-100,-100,-100,-100,0,-100,-100,-100,-100,-100,-100,-100,-100,-100,0],
[-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,0,-100,-100,-100,-100,-100,0,-100,-100,-100],
[-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,0,-100,0,-100,-100],
[-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,0,-100,-100,-100,0,-100,0,-100],
[-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,0,-100,0],
[-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,0,-100,-100,-100,0,-100]])

    
    def __init__(self):
        self.learning_rate = 0.001  #神经网络的学习速率设为0.001
        self.state_num = 20  #有20个位置状态
        self.action_num = 20  #有20个动作
        self.epsilon = 0.9   #动作选择概率初始值设为0.9
        self.epsilon_final = 0.9999 
        self.state_list = np.identity(self.state_num)  
        self.action_list = np.identity(self.action_num)  
        self.relay_memory_store = deque()  
        self.memory_size = 10000  
        self.observe = 2500  
        self.batch_mini = 200  
        self.gamma = 0.9  
        self.learn_step_counter = 0 
        self.train_step_counter = 0   
        self.bug = 0  
        self.creat_network()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def creat_network(self):
        self.q_eval_input = tf.placeholder(shape=[None,self.state_num],dtype=tf.float32)   
        self.action_input = tf.placeholder(shape=[None,self.action_num],dtype=tf.float32)  
        self.q_target = tf.placeholder(shape=[None],dtype=tf.float32) 
        neuro_layer_1 = 8  #隐含层
        w1 = tf.Variable(tf.random_normal([self.state_num,neuro_layer_1]))  
        b1 = tf.Variable(tf.zeros([1,neuro_layer_1])+0.1)  
        l1 = tf.nn.relu(tf.matmul(self.q_eval_input,w1)+b1)  
        w2 = tf.Variable(tf.random_normal([neuro_layer_1,self.state_num]))  
        b2 = tf.Variable(tf.zeros([1,self.action_num])+0.1)  
        self.q_eval = tf.matmul(l1,w2)+b2   
        self.reward_action = tf.reduce_sum(tf.multiply(self.q_eval,self.action_input),reduction_indices=1)  
        self.loss = tf.reduce_mean(tf.square(self.q_target - self.reward_action))   
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss) 
        

        
    def choose_action(self,state_index):
        current_state = self.state_list[state_index:state_index + 1]  
        if ((np.random.uniform() > self.epsilon ) or  (self.train_step_counter < self.observe)):  
            action_index = np.random.randint(0,self.action_num) 
        else:
            action_q = self.session.run(self.q_eval,feed_dict={self.q_eval_input:current_state})  
            action_index = np.argmax(action_q) 
        return action_index  
    

    def save(self,state0,action0,reward0,next_state0,done0): 
        current_state1 =  self.state_list[state0:state0+1]
        current_action1 = self.action_list[action0:action0+1]
        next_state1 = self.state_list[next_state0:next_state0+1]
        self.relay_memory_store.append((current_state1,current_action1,reward0,next_state1,done0))  #保存
        if len(self.relay_memory_store) > self.memory_size:  
            self.relay_memory_store.popleft()  


    def experience_replay(self):   #经验回放
        batch = random.sample(self.relay_memory_store,self.batch_mini)  
        batch_state = None
        batch_action = None
        batch_reward = None
        batch_next_state = None
        batch_done = None
        for i in range(self.batch_mini):  
            batch_state = batch[i][0] if batch_state is None else np.vstack((batch_state,batch[i][0])) 
            batch_action = batch[i][1] if batch_action is None else np.vstack((batch_action,batch[i][1]))
            batch_reward = batch[i][2] if batch_reward is None else np.vstack((batch_reward,batch[i][2]))
            batch_next_state = batch[i][3] if batch_next_state is None else np.vstack((batch_next_state,batch[i][3]))
            batch_done = batch[i][4] if batch_done is None else np.vstack((batch_done,batch[i][4]))
        q_target = []
        q_next = self.session.run(self.q_eval,feed_dict={self.q_eval_input:batch_next_state})
        for i in range(self.batch_mini):
            R = batch_reward[i][0]  
            q_value = R + self.gamma * np.max(q_next[i]) 
            if R < 0:
                q_target.append(R)
            else:
                q_target.append(q_value)  
        loss3 = self.session.run([self.train_op,self.loss],feed_dict={self.q_eval_input:batch_state,self.action_input:batch_action,self.q_target:q_target})  
        self.learn_step_counter += 1  

        
    def train(self):
        current_state = np.random.randint(0,self.state_num)  
        while True:
            current_action = self.choose_action(current_state)  
            next_state = current_action 
            current_reward = self.r_list[current_state][current_action]   
            done = True  if current_state == 3 else False  
            self.save(current_state,current_action,current_reward,next_state,done)  
            if self.train_step_counter > self.observe:  
                self.experience_replay()  
            if self.train_step_counter > 40000:   
                break
            if done:
                current_state = np.random.randint(0,self.state_num)        
            else:
                current_state = next_state
            if self.train_step_counter > self.observe and self.epsilon < self.epsilon_final:  
                self.epsilon += 0.00001
            self.train_step_counter += 1
            if self.train_step_counter % 1000 ==0:  
                print(self.train_step_counter)
                

    def play(self):
        print('---开始训练---')
        self.train()
        print('---训练结束---')
        for i in range(20):  
            if self.bug >10:  
                break
            else:
                self.bug = 0
            print('\n第',i,'轮')
            start_state = i  
            print('从',start_state,'位置出发')
            current_state = start_state
            while current_state != 3:  
                out_result = self.session.run(self.q_eval,feed_dict={self.q_eval_input:self.state_list[current_state:current_state+1]})
                out_next_state = np.argmax(out_result[0])
                if out_next_state != 3:
                    print('经过',out_next_state,'位置')
                else:
                    print('到达终点3')
                current_state = out_next_state
                if self.bug > 10:  
                    print('-------神经网络训练不好------')
                    break
                else:
                    self.bug +=1
                
    
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    q_network = DeepQNetwork()
    q_network.play()
    end_time = datetime.datetime.now()
    print('\n----------用时',int((end_time - start_time).seconds/60),'分',(end_time - start_time).seconds%60,'秒-----------')
