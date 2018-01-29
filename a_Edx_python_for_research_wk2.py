
# coding: utf-8

# ### 2.1.1 Scope Rules

# In[1]:

def update():
    x.append()
x=[]
update


# In[2]:

def update(n,x):
    n=2
    x.append(4)
    print('update:',n,x)
def main():
    n=1
    x=[0,1,2,3]
    print('main:',n,x)
    update(n,x)
    print('main:',n,x)
main()


# In[3]:

#exercise
def increment(n): 
    n += 1 
    print(n) 

n = 1 
increment(n) 
print(n)


# In[2]:

def increment(n):
    n += 1
    return n
n = 1
while (n < 10):
    n = increment(n)
print(n)


# ### 2.1.2 Classes and Object-oriented Programming

# In[3]:

ml=[5,9,3,6,8,11,4,3]
ml.sort()
ml
#a little program of say 100 lines is better in functional approach, 
#but longer and complicated ones are impossible without object-orientation. needs class to do that.


# In[11]:

class Mylist(list): #inheritance  #instance in the class
#class is a new object in python
    def remove_min(self):
        self.remove(min(self))
    def remove_max(self):
        self.remove(max(self))
x=[10,3,5,1,2,7,6,4,8]
y=Mylist(x)
dir(x) #contain all the methods in the list
dir(y)  
y.remove_min()
y.remove_max()
y


# In[6]:

x=[5,2,9,11,10,2,7]
print(min(x))
max(x)
x.remove(10) #only remove the first occurence of the onject
x


# ### 2.2.1 Scope Rules

# In[15]:

#numpy.org  numpy arrays
import numpy as np
zero_vector=np.zeros(5)
zero_matrix=np.zeros((5,3)) #rows and columns
zero_vector #floating numbers
zero_matrix


# In[16]:

#two short one dimensional arrays
x=np.array([1,2,3])
y=np.array([2,4,6])


# In[17]:

[[1,3],[5,9]] #nest list


# In[22]:

a=np.array([[1,3],[5,9]])
a


# In[23]:

#transpose of the matrix
a.transpose()


# ## 2.2.2 Slicing numpy arrays

# In[4]:

import numpy as np
x=np.array([1,2,3])
y=np.array([2,4,6])
X=np.array([[1,2,3],[4,5,6]])
Y=np.array([[2,4,6],[8,10,12]])
x[2]


# In[6]:

x[0:2]


# In[9]:

z=x+y # the calculations of individual elements
z


# In[11]:

X[:,1] # take the second column from the array


# In[12]:

Y[:,1]


# In[13]:

X[:,1]+Y[:,1]


# In[14]:

X[1,:]


# In[15]:

Y[1,:]  # take the second row of the matrix


# In[16]:

X[1,:]+Y[1,:]


# In[17]:

[2,4]+[6,10]


# In[18]:

np.array([2,4])+np.array([6,10])


# In[19]:

a = np.array([1,2]) 
b = np.array([3,4,5]) 
a + b


# ## 2.2.3 indexing numpy arrays

# In[20]:

z1=np.array([1,3,5,7,9])
z2=z1+1


# In[21]:

z1


# In[22]:

z2


# In[23]:

ind=[0,2,3] #define by a python list
z1[ind]


# In[24]:

ind=np.array([0,2,3])  #also can be defined by a numpy array
z1[ind]


# In[25]:

z1>6


# In[26]:

z1[z1>6]


# In[27]:

z2[z1>6] #give the index into z2


# In[28]:

ind=z1>6
ind #it's logical vector


# In[30]:

print(z1[ind])
z2[ind]  # won't change the array


# In[31]:

w=z1[0:3]
w


# In[32]:

w[0]=3
w


# In[33]:

z1 #if slicing, the original array will be change


# In[39]:

ind=np.array([0,1,2])
w=z2[ind]
w
w[0]=3
w


# In[40]:

z2 # the value of z2 is not be modified


# In[48]:

a = np.array([1,2]) 
b = np.array([3,4,5]) 
c = b[1:]
b[a]
print(b[a] is c) 
print(b[a] == c) #the expression of the boolean value in the numpy arrays


# ## 2.2.4: Building and Examining NumPy Arrays

# In[49]:

np.linspace(0,100,10) # start point and ending point


# In[51]:

#10 logrithmic element, uniformly distributed
np.logspace(1,2,10) #lg10,lg100,number of arguments


# In[52]:

np.logspace(np.log10(250),np.log10(500),10)


# In[53]:

X=np.array([[1,2,3],[4,5,6]])
X.shape


# In[54]:

X.size  #data methods instead of array methods


# In[64]:

x=np.random.random(10) #standard unifrom distribution
np.any(x>0.9)


# In[65]:

np.all(x>=0.1)  #the output just single true or false


# In[66]:

x


# In[73]:

y=np.random.randint(0,10,20)
y[y>5]
idx=np.where(y>5)
y[idx]


# ## 2.3.1: Introduction to Matplotlib and Pyplot

# In[78]:

#pyplot for data visulization
import matplotlib.pyplot as plt
plt.plot([0,1,4,9,16]) #only in the ipython the graph will show
plt.show()


# In[80]:

x=np.linspace(0,10,20)
y=x**2
plt.plot(x,y);
plt.show()


# In[82]:

y1=x**2.0
y2=x**1.5
plt.plot(x,y1,'bo-') #blue,circle,-
plt.show()  #it's continuous line


# In[83]:

plt.plot(x,y1,'bo-',linewidth=2,markersize=12)
plt.show()


# In[84]:

plt.plot(x,y2,'gs-',linewidth=2,markersize=12) #green, square
plt.show()


# In[85]:

plt.plot([0,1,2],[0,1,4],"rd-") #only points, about two connected lines
plt.show() # red diamonds


# ## 2.3.2: Customizing Your Plots

# In[94]:

x=np.linspace(0,10,20)
y1=x**2
y2=x**1.5
plt.plot(x,y1,'bo-',linewidth=2,markersize=12,label='First')
plt.plot(x,y2,'gs-',linewidth=2,markersize=12,label='Second')
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.axis([-0.5,10.5,-0.5,105])#plt.axs([xmin,xmax,ymin,ymax])
plt.legend(loc='upper left') #line of different name
plt.savefig('myplot.pdf') #use finder to seek out
plt.show()


# ## 2.3.3: Plotting Using Logarithmic Axes

# In[96]:

x=np.linspace(0,10,20)
y1=x**2
y2=x**1.5
plt.loglog(x,y1,'bo-',linewidth=2,markersize=12,label='First')
plt.loglog(x,y2,'gs-',linewidth=2,markersize=12,label='Second')
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.axis([-0.5,10.5,-0.5,105])#plt.axs([xmin,xmax,ymin,ymax])
plt.legend(loc='upper left') #line of different name
plt.savefig('myplot.pdf') #use finder to seek out
plt.show()


# In[98]:

x=np.logspace(-1,1,40)  #all of the points along X axis are evenly spaced
y1=x**2
y2=x**1.5
plt.loglog(x,y1,'bo-',linewidth=2,markersize=12,label='First')
plt.loglog(x,y2,'gs-',linewidth=2,markersize=12,label='Second')
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.axis([-0.5,10.5,-0.5,105])#plt.axs([xmin,xmax,ymin,ymax])
plt.legend(loc='upper left') #line of different name
plt.savefig('myplot.pdf') #use finder to seek out
plt.show()


# ## 2.3.4: Generating Histograms

# In[104]:

import matplotlib.pyplot as plt
import numpy as np
x= np.random.normal(size=1000)
plt.hist(x,normed=True,bins=np.linspace(-5,5,21))  
#provide the location of the bins, get 21 points to get 20 bins
#demonstrate features of the hist
plt.show()


# In[112]:

#subplot three aruguments
x=np.random.gamma(2,3,10000)
plt.hist(x,bins=30,normed=True,cumulative=True,histtype='step')
plt.show()


# In[118]:

plt.figure()
plt.subplot(231)
plt.hist(x,bins=30)
plt.subplot(232)
plt.hist(x,bins=30,normed=True) #normalize to 1
plt.subplot(233)
plt.hist(x,bins=30,cumulative=True)
plt.subplot(234)
plt.hist(x,bins=30,normed=True,cumulative=True,histtype='step')
plt.show()  #must give a three digits number of the subplot 23 means 2*3


# ## 2.4.1: Simulating Randomness

# In[121]:

import random
random.choice(['H','T']) #toss the coin randomly


# In[123]:

random.choice([0,1])
random.choice([1,2,3,4,5,6]) #can replace the list by any object function


# In[126]:

random.choice(range(1,7))  #stop before the stop value


# In[127]:

random.choice([range(1,7)])  #this list only contain one object


# In[131]:

random.choice(random.choice([range(1,7),range(1,9),range(1,11)])) #innermost and outermost
#choose from the three dies uniformly and roll it


# In[134]:

random.choice(range(10))


# In[137]:

random.choice(list([1,2,3,4]))


# In[146]:

sum(random.choice(range(10)) for i in range(10)) # take the sume between 0-9


# In[148]:

sum(random.sample(range(10),10))  #forever 45, the sum of 1-9:(1+9)/2 * 9


# ## 2.4.2: Examples Involving Randomness

# In[154]:

import random
rolls=[]#get some variable to contain the results
#100 times, for loop
for k in range(100):
    rolls.append(random.choice([1,2,3,4,5,6]))
plt.hist(rolls,bins=np.linspace(0.5,6.5,7)) #from 0.5-6.5, 7 points to get 6 bins
plt.show()


# In[156]:

for k in range(1000000):  #when the number gets large, the distribution becomes more smooth
    rolls.append(random.choice([1,2,3,4,5,6]))
plt.hist(rolls,bins=np.linspace(0.5,6.5,7)) #from 0.5-6.5, 7 points to get 6 bins
plt.show()


# In[164]:

#roll ten dice
#y=x1+x2+x3+...x10
ys=[]
for rep in range(1000000):
    y=0
    for k in range(10):
        x=random.choice([1,2,3,4,5,6])
        y += x
    ys.append(y)
print(min(ys))
print(max(ys))
plt.hist(ys,bins=np.linspace(5.5,60.5,30))
plt.show() #beautiful histgram: central limit theorem
#you can sum up the random variables,utilize simulation to prove that
# CLT:why normal distribution is called a gaussian distribution


# In[165]:

list((1,2,3,4))


# In[166]:

random.choice(list((1,2,3,4)))


# ## 2.4.3: Using the NumPy Random Module

# In[167]:

import numpy as np
np.random.random() #generalize one number unifrom from (0,1)


# In[168]:

np.random.random(5)


# In[169]:

np.random.random((5,3)) #number of rows and columns


# In[172]:

#how to generate normal distribution
np.random.normal(0,1)


# In[173]:

np.random.normal(0,1,5)


# In[174]:

np.random.normal(0,1,(4,5)) #insert tuple, other parentices


# In[188]:

X= np.random.randint(1,7,(10,7))
X#start from small


# In[184]:

#how to sum over the row


# In[185]:

np.sum(X,axis=0) # first dimension is to sum over all the columns


# In[187]:

Y= np.sum(X,axis=1) #sum over the second dimension, which means sum over all the rows


# In[191]:

X= np.random.randint(1,7,(1000000,10))
Y= np.sum(X,axis=1)
plt.hist(Y)
plt.show()
#shorter than previous code(over 10times faster)


# In[193]:

np.sum(np.random.randint(1,7,(100,10)), axis=0)


# ## 2.4.4: Measuring Time

# In[194]:

# how long will it take to run the code on a big dataset
import time 
start_time=time.clock() #give the current time
end_time=time.clock()
end_time-start_time


# In[195]:

start_time=time.clock()
ys=[]
for rep in range(1000000):
    y=0
    for k in range(10):
        x=random.choice([1,2,3,4,5,6])
        y += x
    ys.append(y)
print(min(ys))
print(max(ys))
plt.hist(ys,bins=np.linspace(5.5,60.5,30))
plt.show()
end_time=time.clock()
end_time-start_time


# In[196]:

start_time=time.clock()
X= np.random.randint(1,7,(1000000,10))
Y= np.sum(X,axis=1)
plt.hist(Y)
plt.show()
end_time=time.clock()
end_time-start_time


# In[197]:

17.600209999999997/0.4768790000000038 #much faster


# ## 2.4.5: Random Walks

# In[199]:

# location is given by the initial postion + change of the lcoation
delta_X= np.random.normal(0,1,(2,5)) #two row and five columns
# this is the displacements
plt.plot(delta_X[0],delta_X[1],'go')  #0,1 is the index about the row, first row and second row
plt.show()


# In[207]:

delta_X= np.random.normal(0,1,(2,5)) #two row and five columns
# this is the displacements
# do cumulative sum in numpy
X=np.cumsum(delta_X,axis=1) #sum over the column
plt.plot(X[0],X[1],'ro-')  #0,1 is the index about the row, first row and second row
plt.show()


# In[209]:

# how to concatenate the numpy
#meaning of concatenate: connect and bind
#Takes an iterable of np.arrays as arguments, and binds them along the axis argument.   
X_0=np.array([[0],[0]]) #initial location
X = np.concatenate((X_0,np.cumsum(delta_X,axis=1)),axis=1)
X #the first element is 0


# In[212]:

X_0=np.array([[0],[0]])
delta_X= np.random.normal(0,1,(2,100))
X = np.concatenate((X_0,np.cumsum(delta_X,axis=1)),axis=1)
plt.plot(X[0],X[1],'ro-')
plt.show()


# In[214]:

plt.figure()
plt.subplot(221)
delta_X= np.random.normal(0,1,(2,10))
X = np.concatenate((X_0,np.cumsum(delta_X,axis=1)),axis=1)
plt.plot(X[0],X[1],'ro-')
plt.subplot(222)
delta_X= np.random.normal(0,1,(2,100))
X = np.concatenate((X_0,np.cumsum(delta_X,axis=1)),axis=1)
plt.plot(X[0],X[1],'ro-')
plt.subplot(223)
delta_X= np.random.normal(0,1,(2,10000))
X = np.concatenate((X_0,np.cumsum(delta_X,axis=1)),axis=1)
plt.plot(X[0],X[1],'ro-')
plt.subplot(224)
delta_X= np.random.normal(0,1,(2,100000))
X = np.concatenate((X_0,np.cumsum(delta_X,axis=1)),axis=1)
plt.plot(X[0],X[1],'ro-')
plt.show()


# ## Homework

# In[138]:

#https://en.wikipedia.org/wiki/Tic-tac-toe
import numpy as np
import random
def create_board():
    return np.zeros((3,3),dtype=int)
board = create_board()

def place(board,player,position):
    if board[position]==0:
        board[position]=player
    return board
board = create_board()
place(board,1,(0,0))

def possibilities(board):
    return list(zip(*np.where(board==0))) #zip is to get the opposite
possibilities(board)

def random_place(board,player):
    selections = possibilities(board)
    if len(selections)>0:
        selection = random.choice(selections)
        place(board,player,selection)
    return board
random_place(board,2)

board=create_board()
for i in range(3):
    for player in [1,2]:
        random_place(board,player)
board
#this part is the process of game

def row_win(board,player):
    if np.any(np.all(board==player,axis=1)):
        return True
    else:
        return False    
row_win(board,1)

def col_win(board,player):
    if np.any(np.all(board==player,axis=0)):
        return True
    else:
        return False    
col_win(board,1)

def diag_win(board,player):
    if np.all(np.diag(board)==player): # either disgonal of the board consists only their market
        return True
    else:
        return False    
diag_win(board,1)

def evaluate(board):
    winner = 0
    for player in [1, 2]:
        #Res=np.array([col_win(board,player),row_win(board,1),diag_win(board,1)])
        #if np.any(Res==True):
            #winner = player
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            winner = player
#here is a very important part
    if np.all(board != 0) and winner == 0:
        winner = -1
    return winner

def play_game():
    board=create_board()
    winner=0
    while winner==0:
        for player in [1,2]:
            random_place(board,player)
            winner=evaluate(board)
            if winner!=0:
                break
    return winner
play_game()

import time
import matplotlib.pyplot as plt
begin=time.time()
#game=[]
#for i in range(1000):
    #game.append(play_game())
games = [play_game() for i in range(1000)]
stop=time.time()
print(stop-begin)
plt.hist(games)
plt.show()

def play_strategic_game():
    board, winner = create_board(), 0
    board[1,1] = 1 #the first player put in the middle
    while winner == 0:
        for player in [2,1]:
            random_place(board,player)
            winner=evaluate(board)
            if winner != 0:
                break
    return winner
play_strategic_game()


# In[145]:

import time
import matplotlib.pyplot as plt
begin=time.time()
#game=[]
#for i in range(1000):
    #game.append(play_game())
games = [play_strategic_game() for i in range(100)]
stop=time.time()
print(stop-begin)
plt.hist(games)
plt.show()


# In[143]:

Res=np.array([col_win(board,player),row_win(board,1),diag_win(board,1)])
if np.any(Res==True):
    winner = player
Res


# In[ ]:



