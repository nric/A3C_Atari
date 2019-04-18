# A3C_Atari
This is an implementaion of an asyncronous advantage actor critic A3C algorithm to play open ai gym atari games.

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
