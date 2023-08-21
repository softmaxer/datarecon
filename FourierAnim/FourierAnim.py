#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Import Library

# In[1]:


from presentation import *
import numpy as np


# ## Step 2: Parameterize Image as a Closed-Loop Function

# In[2]:


time_table, x_table, y_table = create_close_loop('guit.png')


# ## Step 3: Calculate Discrete Fourier Transform Coefficient

# In[6]:


order = 100 # We need higher order approximation to get better approximation
coef = coef_list(time_table, x_table, y_table, order)
print(coef)


# ## Step 4: Evaluate Fourier Transform

# In[7]:


space = np.linspace(0, tau, 300) # Did you know what tau is ? Check my previous video about it ! :D
x_DFT = [DFT(t, coef, order)[0] for t in space]
y_DFT = [DFT(t, coef, order)[1] for t in space]


# ## Step 5: Visualize it

# In[8]:


fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(x_DFT, y_DFT, 'r--')
ax.plot(x_table, y_table, 'k-')
ax.set_aspect('equal', 'datalim')
xmin, xmax = xlim()
ymin, ymax = ylim()


# ## Step 6: Create Epicycle Animation

# In[9]:


anim = visualize(x_DFT, y_DFT, coef, order, space, [xmin, xmax, ymin, ymax])
Writer = animation.writers['html']
writer = Writer(fps=60)
anim.save('lesPaul.html',writer=writer, dpi=150)
