import operator
import random

random.seed(None) #to control the random seeding

import numpy as np
import matplotlib.pyplot as plt
plt.close('all') 
plt.rcParams.update({'font.size': 18})
import math

from deap import base
from deap import creator
from deap import tools
from deap import gp

# parameters for the GA
Npopulation = 1000 #size of the population
Ngeneration = 50 #number of generation

########################################################################
# 0 - Problem we are trying to solve (slide 9)
########################################################################

xpoints = [x/10. for x in range(-10,10)] #select 20 points between -1 and 1

def TrainFunction(x):
    y  = x**4 - x**3 - x**2 - x
    return y

#plot the training points
plt.figure()
plt.plot(np.array(xpoints),TrainFunction(np.array(xpoints)),marker='x',linestyle='')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

########################################################################
# 1 - Chose primitive set (slide 10)
########################################################################

# terminal set - one variable named x
pset = gp.PrimitiveSet("MAIN", 1)
pset.renameArguments(ARG0='x')

# terminal set - add ephemeral constant 
name='rand'+str(random.randint(0, 10000))
pset.addEphemeralConstant(name,lambda: random.uniform(-1, 1))

# terminal set - add a constant of value 1.0
pset.addTerminal(1.0)

# function set - add operators 
# multiplication, addition, soustraction, division

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)

#attention the division needs to be protected to avoid division by 0
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
pset.addPrimitive(protectedDiv, 2)

########################################################################
# 2 - Define what is an individual (slide 5)
########################################################################

# This is the way DEAP define individual for genetic programing
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #you define this is a minimisation problem
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) #the individual are primite trees

toolbox = base.Toolbox()

# -> here you can change how the population is initialized try genFull or genGrow for example (slide 13)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2) #  <- here 

#specific code for gp in deap. basically copy and paste from deap tutorial
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

########################################################################
# 2 - Define what is the fitness (slide 11)
########################################################################

def MSE(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    sqerr = np.zeros(len(points))
    for ii, pt in enumerate(points):
        sqerr[ii] = (func(pt) - TrainFunction(pt))**2
    return np.sum(sqerr)/len(points),

# we apply the fitness in an operation called evaluate where we compute the error for each points 
toolbox.register("evaluate", MSE, points=xpoints)

########################################################################
# 4 - Defnie selection and mutation operators (slide 14 to 16)
########################################################################

#selection tournament 
# -> here you can change the selection process (slide 14)
toolbox.register("select", tools.selTournament, tournsize=2)# <- here 

# -> here you can change the probability for the cross-over (slide 15)
toolbox.register("mate", gp.cxOnePoint)
cxpb = 0.4 # <- here 

# -> here you can change the probability for the mutation (slide 16) 
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2) 
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
mutpb = 0.1 # <- here  

# -> here you can change the probability for the shrink (slide 17) 
toolbox.register("shrink", gp.mutShrink) 
mutsh = 0.1 # <- here 

# -> here you can change the probability for the mutation of ephemeral constants
toolbox.register("mut_eph", gp.mutEphemeral, mode='all') 
mut_eph = 0.05 # <- here

########################################################################
# 5 - Genetic algorithm loop
########################################################################

# Initial population size
pop = toolbox.population(n=Npopulation) #generate initial population
hof = tools.HallOfFame(1) #generate elite (save best individual of the population)
    
# this is some esthetic copy and paste from the DEAP example to compute the stat at each generation
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("min", np.min)
mstats.register("mean", np.mean)
mstats.register("std", np.std)
stats=mstats

#first evaluation
logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

# Evaluate the individuals with an invalid fitness 
# this is a small trick to only evaluate individuals that have been mutated
invalid_ind = [ind for ind in pop if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
for ind, fit in zip(invalid_ind, fitnesses):
    ind.fitness.values = fit
    
if hof is not None:
    hof.update(pop)

# this is some esthetic copy and paste from the DEAP example to show the stats at each generation
record = stats.compile(pop) if stats else {}
logbook.record(gen=0, nevals=len(invalid_ind), **record)
print(logbook.stream)

# evolution of elites we record the best individual at each generation 
save_elite = []

# Begin the generational process
for gen in range(1, Ngeneration + 1):
    # Select the next generation individuals
    select = toolbox.select(pop, len(pop)-1)
     # Select the best individual and save it in elite. This is elitism
    elites = tools.selBest(pop, k=1)
    save_elite.append(toolbox.clone(elites))
    
    #clone offspring
    off = [toolbox.clone(ind) for ind in select]
    
    # Apply mutation
    #this is to make sure we apply only one mutation to each individual (slide 18)
    cumpb = np.cumsum([cxpb,mutsh,mutpb,mut_eph])
    for i in range(len(off)):
        pb = random.random()
        if pb<cumpb[0] and i>0:
            off[i-1], off[i] = toolbox.mate(off[i-1],off[i])
            del off[i-1].fitness.values, off[i].fitness.values
            # print('off N=%i, crossover'%(i))
        elif pb<cumpb[1]:
            off[i], = toolbox.shrink(off[i])
            del off[i].fitness.values
            #print('off N=%i, schrink'%(i))
        elif pb<cumpb[2]:
            off[i], = toolbox.mutate(off[i])
            del off[i].fitness.values
        elif pb<cumpb[3]:
            off[i], = toolbox.mut_eph(off[i])
            del off[i].fitness.values
            #print('off N=%i, mutate'%(i))
        # else:
        #     print('off N=%i, reproduce'%(i))
    

    # compute the fitness of the individuals with an invalid fitness
    invalid_ind = [ind for ind in off if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update the hall of fame with the generated individuals
    if hof is not None:
        hof.update(off)

    # Replace the current population by the offspring and the elite
    pop[:] = off + elites

    # Append the current generation statistics to the logbook
    record = stats.compile(pop) if stats else {}

    logbook.record(gen=gen, nevals=len(invalid_ind), 
                   lmean = record['size']['mean'], fmin =  record['fitness']['min'],
                   **record)
    
    #print
    print(logbook.stream)

    
########################################################################
# 6 - Display end solution
########################################################################

bestfunct = toolbox.compile(expr=hof[0])
plotpoints = [x/100. for x in range(-100,100)]

plt.figure()
plt.plot(np.array(xpoints),TrainFunction(np.array(xpoints)),marker='x',linestyle='')
for pt in plotpoints:  
    plt.scatter(pt,bestfunct(pt),marker='.',color='k')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print(hof[0])


#plot bloat issue
fmin = np.zeros(Ngeneration)
lmean = np.zeros(Ngeneration)


for i in range(1,Ngeneration):
    fmin[i] = logbook[i]['fmin']
    lmean[i] = logbook[i]['lmean']


# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(fmin[1:], color="red", marker="o")
# set x-axis label
ax.set_xlabel("nGen",fontsize=14)
# set y-axis label
ax.set_ylabel("fitness",color="red",fontsize=14)
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(lmean[1:],color="blue",marker="o")
ax2.set_ylabel("mean lenght",color="blue",fontsize=14)
plt.show()

#evolution of best individuals
[temp, idx_unique] = np.unique(fmin[1:],return_index=True)
plt.figure()
plt.plot(np.array(xpoints),TrainFunction(np.array(xpoints)),marker='x',linestyle='')
for ii in idx_unique:
    funct = toolbox.compile(expr=save_elite[ii][0])
    rand_color = np.random.rand(3,)
    for pt in plotpoints:  
        plt.scatter(pt,funct(pt),marker='.',color= rand_color)
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
