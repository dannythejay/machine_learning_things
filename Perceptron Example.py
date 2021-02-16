#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import array, random, dot
from random import choice
from pylab import ylim, plot
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Create step function which will return 0 if input passed to it is less than 0, else return 1
step_function = lambda x:0 if x < 0 else 1


# In[3]:


# Create trainin set - first two entries in each tuple represent input values.
# Second element represents the expected result
# Third element is a dummyinput (bias) which helps move threshold up or down as required by step function

training_dataset = [
    (array([0,0,1]),0),
    (array([0,1,1]),1),
    (array([1,0,1]),1),
    (array([1,1,1]),1),
    ]


# In[4]:


weights = random.rand(3)


# In[5]:


error=[]


# In[6]:


learning_rate = 0.2


# In[7]:


n=100


# In[8]:


for i in range (n):
    x,expected = choice(training_dataset)
    result = dot(weights,x)
    err = expected - step_function(result)
    error.append(err)
    weights += learning_rate * err * x


# In[9]:


# Evaluate model

for x,_ in training_dataset:
    result = dot(x,weights)
    print('{}:{} -> {}'.format(x[:2],result,step_function(result)))


# In[ ]:





# In[10]:


ylim([-1,1])
plot(error)
plt.show()


# In[ ]:





# In[ ]:


# See https://sdsclub.com/the-complete-guide-to-perceptron-algorithm-in-python/ for more info!


# In[ ]:





# In[ ]:




