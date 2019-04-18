"""
This is an implementaion of an asyncronous advantage actor critic A3C algorithm to play open ai gym atari games.
More specifically, for Breakout.
I wrote this for my training and understanding, I don't recommend to use the code but what ever floats your boat.
credits go to https://github.com/rlcode/reinforcement-learning/blob/master/3-atari/1-breakout/breakout_a3c.py
and to https://github.com/colinskow/move37/blob/master/actor_critic/a3c.py
However, the source has been rewritten for Tensorflow 2.0 (alpha) with Keras.
A good explanation of what is happening between the models and optimizations I recommend this read:
https://medium.com/tensorflow/deep-reinforcement-learning-playing-cartpole-through-asynchronous-advantage-actor-critic-a3c-7eab2eea5296
Also, the course Move37 makes an excellent introduction (chapter 9.2).

Todo: 
1) Repair session problem (seems ok now but verify)
2) train_model call: tensorflow.python.framework.errors_impl.InvalidArgumentError: Incompatible shapes: [21] vs. [3]
	 [[{{node mul_260}}]] [Op:__inference_keras_scratch_graph_13177]

"""

#%%
import math
import gym
import time
import random
import threading
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

import tensorflow as tf
import tensorflow.keras as K

# global variables for A3C
global episode
episode = 0
EPISODES = 8000000
env_name = "BreakoutDeterministic-v4"

#%%

class A3CAgent:
    """
    A3C employs asynconous parallel agents. This class creats many Agent objects.
    Each Agent holds an own actor and crtic neural net for prediction (playing episodes).
    But the training only happens on the main actor and critic using the optimizers
    defined here. 
    """
    def __init__(self,action_size, state_size):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.99
        self.threads = 1
        # optimizer parameters
        self.actor_lr = 2.5e-4
        self.critic_lr = 2.5e-4
        self.entropy_loss_factor = 0.01
        #Create models
        #tf.reset_default_graph()
        self.actor, self.critic = self.build_model()
        #create optimizers
        self.actor_optimizer = self.build_actor_optimizer()
        self.critic_optimizer = self.build_critic_optimizer()
        #session init
        self.sess = 1
        #self.sess = tf.Session() #self.sess = tf.Session()
        #K.backend.set_session(self.sess)
        #self.sess.run(tf.global_variables_initializer())

    
    def train(self):
        #spawn agents (they inherit from Thread)
        agents = [Agent(
            self.action_size,
            self.state_size,
            self.actor,
            self.critic,
            self.actor_optimizer,
            self.critic_optimizer,
            self.discount_factor,
            self.sess
        ) for _ in range(self.threads)]
        #delay one by one before starting
        for agent in agents:
            time.sleep(1)
            agent.start()
        #keep them running and save from time to time.
        while True:
            time.sleep(1*10)
        #    self.save_model("./save_model/breakout_a3c_test")


    def build_model(self):
        #build layer stack
        input_node = K.Input(shape=self.state_size)
        conv = K.layers.Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input_node)
        conv = K.layers.Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = K.layers.Flatten()(conv)
        fc = K.layers.Dense(256, activation='relu')(conv)
        #policy out has action_size and softmax
        policy = K.layers.Dense(self.action_size, activation='softmax')(fc)
        #value out is just a real number
        value = K.layers.Dense(1, activation='linear')(fc)
        #make actor and critic model
        actor = K.Model(inputs=input_node, outputs=policy)
        critic = K.Model(inputs=input_node, outputs=value)
        #predict functions need to be made for the object (compile not called)
        actor._make_predict_function()
        critic._make_predict_function()
        #print summaries
        actor.summary()
        critic.summary()
        return actor, critic


    def build_actor_optimizer(self):
        #Setup action and advantage placeholder. We don't know the length of the tensors because
        #episode length (step count, batch size) is dynamic
        action = K.backend.placeholder(shape=[None,self.action_size])
        advantages = K.backend.placeholder(shape=[None,])
        #get output tensor of layer. See https://keras.io/getting-started/functional-api-guide/ "Concept of layer node"
        policy = self.actor.output

        good_prob = K.backend.sum(action*policy,axis=1) #action is on-hot.
        eligibility = K.backend.log(good_prob+1e-10) * advantages #Policy Gradient: log(action_prob)*advantages
        actor_loss = - K.backend.sum(eligibility) #over all inputs (steps)

        #also generate entropy bonus: H(p) = Sum(p*log(p))
        entropy = K.backend.sum(policy * K.backend.log(policy+1e-10),axis=1) #sum over action probablilityes (policy)
        entropy = K.backend.sum(entropy) #also sum over steps

        loss = actor_loss * self.entropy_loss_factor * entropy #final loss
        #make update function. I don't really understand. According to source code, get_updates has only two parameters.
        #https://github.com/keras-team/keras/blob/35093bc9bb14a66d09fe6ea26e302df8ffad64f4/keras/optimizers.py#L252
        optimizer = K.optimizers.RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(loss,self.actor.trainable_weights)
        #Here it comes: The K.function takes a list of placeholders and returns output values as np.array.
        #The inputs: the main imput (frames), the action, the advantages. The latter two are just used for the loss
        #The output: just the loss.
        #The update: as defined above
        train = K.backend.function([self.actor.input,action,advantages],[loss],updates = updates)
        return train
    

    def build_critic_optimizer(self):
        """Build the optimizer function for the critic.
        Refer to build_actor_optimizer for detailed comments. This is amlost identical
        except that the loss. The loss for the critic is simple the mean squared error
        between prediction (critic.output) and discounted_reward.
        """
        discounted_reward = K.backend.placeholder(shape=(None, ))
        value = self.critic.output
        loss = K.backend.mean(K.backend.square(discounted_reward - value))
        optimizer = K.optimizers.RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(loss,self.critic.trainable_weights)
        train = K.backend.function([self.critic.input, discounted_reward], [loss], updates=updates)
        return train
     

    def load_model(self,name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")
    
    
    def save_model(self,name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + '_critic.h5')

    


class Agent(threading.Thread):
    """The Agent class defines an playing agent but not the learning Agent.
    The Agent class is used by A3C Agent to build many workers. Each worker
    colelcts data (self.memory) using syncronized copies of actor and critic (pdate_local_model).
    Once happy with the amout of data (t_max), it calls the shared optimizers for the
    main actor and critic and updates the local weights with the main weights.
    """
    def __init__(self, 
    action_size, 
    state_size, 
    actor_main, 
    critic_main, 
    optimizer_actor,
    optimizer_critic, 
    discount_factor,
    sess):
        threading.Thread.__init__(self)
        self.action_size = action_size
        self.state_size = state_size
        self.actor = actor_main
        self.critic = critic_main
        self.actor_optimizer = optimizer_actor
        self.critic_optimizer = optimizer_critic
        #constants
        self.discount_factor = discount_factor
        self.t_max = 20
        #running vars
        self.states, self.actions, self.rewards = [],[],[]
        self.avg_p_max = 0
        self.t = 0
        #set session
        self.sess = sess
        #K.backend.set_session(self.sess)
        #K.backend.clear_session()
        #self.sess = tf.Session()
        #self.sess.run(tf.global_variables_initializer())
        #K.backend.set_session(self.sess)
        #build the local models (will not train but required to predict/play)
        self.local_actor, self.local_critic = self.build_local_model()
   
                
    def build_local_model(self):
        """Builds a local model identical to the ones which define actor and critic.
        Thre only reason the function is redefined here is because Agent can't call 
        the function of A3CAgent without building an instance but is an instance of it (circular dependancy).
        Returns:
            actor, critic models
        """
        #build layer stack
        input_node = K.Input(shape=self.state_size)
        conv = K.layers.Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input_node)
        conv = K.layers.Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = K.layers.Flatten()(conv)
        fc = K.layers.Dense(256, activation='relu')(conv)
        #policy out has action_size and softmax
        policy = K.layers.Dense(self.action_size, activation='softmax', name = 'softmax')(fc)
        #value out is just a real number
        value = K.layers.Dense(1, activation='linear', name = 'value')(fc)
        #make actor and critic model
        actor = K.Model(inputs=input_node, outputs=policy)
        critic = K.Model(inputs=input_node, outputs=value)
        #predict functions need to be made for the object (compile not called)
        actor._make_predict_function()
        critic._make_predict_function()
        #print summaries
        actor.summary()
        critic.summary()
        return actor, critic


    def discount_rewards(self,rewards,done):
        """generated the discounted reward list from a reward list
        params:
            :rewards: lsit of rewards
            :done: boolin done. If true then take the last availibe state
        return:
            Returns the discounted reward list accrding to definition
        """
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.float32(self.states[-1] / 255.))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    
    def train_model(self,done):
        """trains self.actor and self.critic using optimizer_actor and optimizer_critic.
        The self.actor and self.critic are shared and part of the hosting object A3Cagent.
        params:
            :done: boolean, required to calc discounted rewards
        """
        dicounted_rewards = self.discount_rewards(self.rewards,done)
        states = np.zeros((len(self.states),84,84,4))
        for i in range(len(self.states)):
            states[i] = self.states[i]
        
        states = np.float32(states/255)
        #here the critic comes into play. the advantages are calculated by the predicted
        #value for the state and the observed+discounted reward for the state
        values = self.critic.predict(states)
        values = np.reshape(values,len(values))
        advantages = dicounted_rewards - values
        print(f"dicounted_rewards{dicounted_rewards.shape} values{values.shape} advantages{advantages.shape}")
        print(f"states{states.shape} self.actions{len(self.actions)}")
        self.actor_optimizer([
            states,
            self.actions,
            advantages])
        self.critic_optimizer([states,dicounted_rewards])
        self.states, self.actions, self.rewards = [], [], []
    

    def update_local_model(self):
        """Updates the local weights (for predicting) with the latest trained actor and critic weights
        from the super-actor and critic
        """
        #K.backend.clear_session()
        #K.backend.set_session(self.sess)
        #self.actor._make_predict_function()
        #self.actor._make_predict_function()
        temp = self.actor.get_weights()
        #K.backend.clear_session()
        #K.backend.set_session(self.sess)
        #self.local_actor._make_predict_function()
        self.local_actor.set_weights(temp)
        #K.backend.clear_session() 
        temp = self.critic.get_weights()
        #K.backend.clear_session() 
        self.local_critic.set_weights(temp)
    

    def get_action(self,history):
        """Gets an action given a history.
        History has the hape of the actor input.
        Returns:
            :action_index: index of the weighted-random selected action (needs to be changed to "real" action in main)
            :policy: the probability distribution for the possible actions for this state.
        """
        history = np.float32(history/255.)
        policy = self.local_actor.predict(history)[0]
        action_index = np.random.choice(self.action_size,1,p=policy)
        return action_index, policy
    

    def memory(self,history,action,reward):
        """adds S,A,R pair to memory buffers
        """
        self.states.append(history)
        action_onehot = np.zeros(self.action_size)
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.rewards.append(reward)     
    

    def run(self):
        global episode
        env = gym.make(env_name)
        step = 0
        while episode < EPISODES:
            #init episode
            done = False
            dead = False
            score, start_life = 0,5
            score, start_life = 0, 5
            observe = env.reset()
            next_observe = observe
            # this is one of DeepMind's idea. just do nothing at the start of episode to avoid sub-optimal
            for _ in range(random.randint(1, 30)):
                observe = next_observe
                next_observe, _, _, _ = env.step(1)
            # At start of episode, there is no preceding frame. So just copy initial states to make history
            state = pre_processing(next_observe, observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                env.render()
                step += 1
                self.t += 1
                observe = next_observe
                #get action and policy. Policy not really needed.
                action,policy = self.get_action(history)
                # change action to real_action
                if action == 0: real_action = 1
                elif action == 1: real_action = 2
                else: real_action = 3
                #game specific action when dead
                if dead:
                    action = 0
                    real_action = 1
                    dead = False
                #play a step and collect data
                next_observe,reward,done,info = env.step(real_action)
                #pre-process -->history. Next frame and re-use the last 3 previous frames
                next_state = pre_processing(next_observe,observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                self.avg_p_max += np.amax(self.actor.predict(np.float32(history/255.)))

                # if the ball is fall, then the agent is dead --> episode is not over
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']
                
                score += reward
                #another trick to help convergence: clip rewards
                reward = np.clip(reward, -1, 1)

                #save the calulated history, action and reward to memory
                self.memory(history,action,reward)

                #special case when dead: reset history. Else, next
                if dead:
                    history = np.stack((next_state,next_state,next_state,next_state),axis = 2)
                    history = np.reshape([history],(1,84,84,4))
                else:
                    history = next_history
                
                #Finally: Train the overlaying main model and update the local model with the new weights
                if self.t > self.t_max or done:
                    self.train_model(done)
                    self.update_local_model()
                    self.t=0
                
                # if done, plot the score over episodes and reset stuff
                if done:
                    episode += 1
                    print("episode:", episode, "  score:", score, "  step:", step)
                    self.avg_p_max = 0
                    step = 0



#%%
#Constants
STATE_SIZE = (84,84,4) #input pixels and layer count

def pre_processing(next_observe, observe):
    """takes two observations as returned from gym.env.step
    and returns smaller, monochrome, integer representations.
    """
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(resize(rgb2gray(processed_observe), STATE_SIZE[0:2], mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    global_agent = A3CAgent(action_size=3, state_size=STATE_SIZE)
    global_agent.train()

#%%
