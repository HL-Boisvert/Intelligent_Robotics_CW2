from controller import Supervisor
from controller import Keyboard
from controller import Display

import numpy,struct
import ga,os

# import for csv
import pandas as pd
from datetime import datetime

class SupervisorGA:
    def __init__(self):
        # Simulation Parameters
        # Please, do not change these parameters
        self.time_step = 32 # ms
        self.time_experiment = 150 # s
        
        # Initiate Supervisor Module
        self.supervisor = Supervisor()
        # Check if the robot node exists in the current world file
        self.robot_node = self.supervisor.getFromDef("Controller")
        if self.robot_node is None:
            sys.stderr.write("No DEF Controller node found in the current world file\n")
            sys.exit(1)
        # Get the robots translation and rotation current parameters    
        self.trans_field = self.robot_node.getField("translation")  
        self.rot_field = self.robot_node.getField("rotation")
        
        # Check Receiver and Emitter are enabled
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.time_step)
        
        # Initialize the receiver and emitter data to null
        self.receivedData = "" 
        self.receivedWeights = "" 
        self.receivedFitness = "" 
        self.emitterData = ""
        
        ### Define here the GA Parameters
        self.num_generations = 100
        self.num_population = 20
        self.num_elite = self.num_population * 0.1 # 10% of elite
        
        # size of the genotype variable
        self.num_weights = 0
        
        # Creating the initial population
        self.population = []
        
        # All Genotypes
        self.genotypes = []
        
        # Display: screen to plot the fitness values of the best individual and the average of the entire population
        self.display = self.supervisor.getDevice("display")
        self.width = self.display.getWidth()
        self.height = self.display.getHeight()
        self.prev_best_fitness = 0.0;
        self.prev_average_fitness = 0.0;
        self.display.drawText("Fitness (Best - Red)", 0,0)
        self.display.drawText("Fitness (Average - Green)", 0,10)
        
        # Data for CSV data analysis
        self.all_fitness_list = []
        # trajectory save
        self.x_traj = []
        self.z_traj = []
        self.traj_matrix = []

    def createRandomPopulation(self):
        # Wait until the supervisor receives the size of the genotypes (number of weights)
        if(self.num_weights > 0):
            # Define the size of the population
            pop_size = (self.num_population,self.num_weights)
            # Create the initial population with random weights
            self.population = numpy.random.uniform(low=-1.0, high=1.0, size=pop_size)

    def handle_receiver(self):
        while(self.receiver.getQueueLength() > 0):
            self.receivedData = self.receiver.getData().decode("utf-8")
            typeMessage = self.receivedData[0:7]
            # Check Message 
            if(typeMessage == "weights"):
                self.receivedWeights = self.receivedData[9:len(self.receivedData)] 
                self.num_weights = int(self.receivedWeights)
            elif(typeMessage == "fitness"):  
                self.receivedFitness = float(self.receivedData[9:len(self.receivedData)])
            self.receiver.nextPacket()
        
    def handle_emitter(self):
        if(self.num_weights > 0):
            # Send genotype of an individual
            string_message = str(self.emitterData)
            string_message = string_message.encode("utf-8")
            #print("Supervisor send:", string_message)
            self.emitter.send(string_message)     
        
    def run_seconds(self,seconds):
        #print("Run Simulation")
        stop = int((seconds*1000)/self.time_step)
        iterations = 0
        while self.supervisor.step(self.time_step) != -1:
            self.handle_emitter()
            self.handle_receiver()
            if(stop == iterations):
                break    
            iterations = iterations + 1
                
    def evaluate_genotype(self,genotype,generation):
        # Send genotype to robot for evaluation
        self.emitterData = str(genotype)
        
        # Reset robot position and physics
        INITIAL_TRANS = [4.48, 0, 7.63]
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        INITIAL_ROT = [0, 1, 0, -0.0]
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()
    
        # Evaluation genotype 
        self.run_seconds(self.time_experiment)
    
        # Measure fitness
        fitness = self.receivedFitness
        print("Fitness: {}".format(fitness))
        current = (generation,genotype,fitness)
        self.genotypes.append(current)  
        
        return fitness

    def run_demo(self):
        # Read File
        genotype = numpy.load("Best.npy")
        # Send Genotype to controller
        self.emitterData = str(genotype) 
        
        # Reset robot position and physics
        INITIAL_TRANS = [4.48, 0, 7.63]
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        INITIAL_ROT = [0, 1, 0, -0.0]
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()
    
        # Evaluation genotype 
        self.run_seconds(self.time_experiment)    
    
    def run_optimization(self):
        # Wait until the number of weights is updated
        while(self.num_weights == 0):
            self.handle_receiver()
            self.createRandomPopulation()
        
        print("starting GA optimization ...\n")
        
        # For each Generation
        for generation in range(self.num_generations):
            print("Generation: {}".format(generation))
            current_population = []   
            # Select each Genotype or Individual
            for population in range(self.num_population):
                genotype = self.population[population]
                # Evaluate
                fitness = self.evaluate_genotype(genotype,generation)
                #print(fitness)
                # Save its fitness value
                current_population.append((genotype,float(fitness)))
                #print(current_population)
                
            # After checking the fitness value of all indivuals
            # Save genotype of the best individual
            best = ga.getBestGenotype(current_population);
            average = ga.getAverageGenotype(current_population);
            numpy.save("Best.npy",best[0])
            self.plot_fitness(generation, best[1], average);
            
            # Generate the new population using genetic operators
            if (generation < self.num_generations - 1):
                self.population = ga.population_reproduce(current_population,self.num_elite);
        # Allows user to save the fitness value into a csv
        self.save_csv(self.all_fitness_list,['fitness'],'fitness')
        
        #print("All Genotypes: {}".format(self.genotypes))
        print("GA optimization terminated.\n")   
    
    
    def draw_scaled_line(self, generation, y1, y2): 
        # Define the scale of the fitness plot
        XSCALE = int(self.width/self.num_generations);
        YSCALE = 100;
        self.display.drawLine((generation-1)*XSCALE, self.height-int(y1*YSCALE), generation*XSCALE, self.height-int(y2*YSCALE));
    
    def plot_fitness(self, generation, best_fitness, average_fitness):
        if (generation > 0):
            self.display.setColor(0xff0000);  # red
            self.draw_scaled_line(generation, self.prev_best_fitness, best_fitness);
    
            self.display.setColor(0x00ff00);  # green
            self.draw_scaled_line(generation, self.prev_average_fitness, average_fitness);
    
        self.prev_best_fitness = best_fitness;
        self.prev_average_fitness = average_fitness;
        
        self.all_fitness_list.append(best_fitness)
        self.all_fitness_list.append(average_fitness)
    # function that sav a certain value into a csv
    def save_csv(self,data_mat,col_names,csv_name):
        # create a time tag
        time_now = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
        # Print to the user that the data is saved
        print(f'{csv_name} data saved')
        # create the dataframe
        csv_data = pd.DataFrame(data_mat,columns=col_names)
        # save it
        csv_data.to_csv(f'{csv_name}_{time_now}_.csv',index=False)
    
if __name__ == "__main__":
    # Call Supervisor function to initiate the supervisor module   
    gaModel = SupervisorGA()
    
    # Function used to run the best individual or the GA
    keyboard = Keyboard()
    keyboard.enable(50)
    
    # Interface
    print("(R|r)un Best Individual or (S|s)earch for New Best Individual:")
    while gaModel.supervisor.step(gaModel.time_step) != -1:
        resp = keyboard.getKey()
        if(resp == 83 or resp == 65619):
            gaModel.run_optimization()
            print("(R|r)un Best Individual or (S|s)earch for New Best Individual:")
        elif(resp == 82 or resp == 65619):
            gaModel.run_demo()
            print("(R|r)un Best Individual or (S|s)earch for New Best Individual:")
        