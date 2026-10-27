import sys
import os
import numpy as np
import time

#Easy system referenced values for processing arguments
argc = len(sys.argv)
argv = sys.argv

#Mode is the current setting for which version of the program we're running
# 0 is training mode
# 1 is predicting mode
# 2 is evaluation mode
# -1 is default and an invalid mode that must be set during initialilzation
#It's never really important that it's global, I expected to put all processing variables as global ones but java has proned me to not do that
mode = -1


#This is an error function for simple error throwing
def printError(msg):
	print("An Error Has Occured: " + msg)
	sys.exit()

#This function sets up training operation
#This version of training handles analytical training
# order is a integer >= 1
# x is the input data as a text file defined as name.x.txt
# y is the target data as a text file defined as name.y.txt
# model is the ccreated/to be modifed file with the trained data defined as model.txt
def runAnalyticalTraining(order, x, y, model):
	##	This section handles all input validation	##

	#Make sure all the values provided are valid
	if int(order) < 1:
		printError("Invalid order provided")
	if not os.path.exists(x):
		printError("Invalid input x file")
	if not os.path.exists(y):
		printError("Invalid target y file")

	#Create the model file if it doesn't already exist
	if not os.path.exists(model):
		modelFile = open(model,"w")
		modelFile.close()

	print("Operating in training mode using " + x + " as input training data and " + y + " as target output data")
	print("Model output will be saved in file " + model)

	#Open the input files as matricies so linear algebra can be preformed on them
	inputData = np.loadtxt(x, ndmin=2)		  #This contains the matrix for the input data
	targetData = np.loadtxt(y).reshape(-1)    #This contains the matrix for the target output data
	modelFile = open(model, "w")			  #The file for the model

	#Check to make sure the data is in the right format
	if inputData.shape[0] != len(targetData):
		printError("Input data mismatch. x.txt and y.txt must have the same number of rows")

	#Check the input is either linear or polynomial with a single feature
	if (int(order) > 1) & (inputData.shape[1] > 1):
		printError("Provided input as a matrix. Only vector inputs are allowed when order > 1")

	##	This section preforms the actual parameter generation	##

	#Include an extra column to include a section for biases
	#This way the process can take place entire via matrix multiplication rather than adding a vector of biases
	biasedX = None

	#start_time = time.perf_counter()
	if int(order) == 1:
		#For linear regression, include all input features
		biasedX = np.column_stack((np.ones(inputData.shape[0]), inputData))
	else:
    	#For polynomial regression, use powers of the single input feature
		polyVector = []
		for exp in range(int(order) + 1):
			feature = inputData[:, 0] ** exp
			polyVector.append(feature)

		biasedX = np.column_stack(polyVector)

	#Calculte the optimal parameters for the model using direct linear algebra
	parameters = np.linalg.lstsq(biasedX, targetData, rcond=None)[0]

	#These are where the parameters are specifically set for clarity
	# w is the later coefficients to use for the rest of the model
	b = parameters[0]  #This is our bias values
	w = parameters[1:] #This is the rest of our values

	#end_time = time.perf_counter()

	#print("Time: " + str(end_time - start_time))
	#Write each of the parameters to the model file
	modelFile.write(str(b) + " ")    #Write the bias first
	for index, para in enumerate(w):
		modelFile.write(str(para))   #Write a model value
		if index != len(w)-1:
			modelFile.write(" ")     #Add a space between each model value

	#Close the model file
	#Input and targets are automatically closed when using loadtxt
	modelFile.close()
	
#This function sets up training operation
#This version of training handles gradient training
# order is a integer >= 1
# x is the input data as a text file defined as name.x.txt
# y is the target data as a text file defined as name.y.txt
# model is the ccreated/to be modifed file with the trained data defined as model.txt
# lr is the learning rate, as a number 0-1
# max_steps is the maximum integer of steps taken during training
def runGradientTraining(order, x, y, model, lr, max_steps):
	##	This section handles all input validation	##

	#Make sure all the values provided are valid
	if int(order) < 1:
		printError("Invalid order provided")
	if not os.path.exists(x):
		printError("Invalid input x file")
	if not os.path.exists(y):
		printError("Invalid target y file")
	if (float(lr) < 0) | (float(lr) > 1):
		printError("Invalid learning rate")
	if int(max_steps) < 0:
		printError("Invalid max steps")

	#Create the model file if it doesn't already exist
	if not os.path.exists(model):
		modelFile = open(model,"w")
		modelFile.close()

	print("Operating in training mode using " + x + " as input training data and " + y + " as target output data")
	print("Model output will be saved in file " + model)

	#Open the input files as matricies so linear algebra can be preformed on them
	inputData = np.loadtxt(x, ndmin=2)		  # This contains the matrix for the input data
	targetData = np.loadtxt(y).reshape(-1)    # This contains the matrix for the target output data

	#Check to make sure the data is in the right format
	if inputData.shape[0] != len(targetData):
		printError("Input data mismatch. x.txt and y.txt must have the same number of rows")

	#Check the input is either linear or polynomial with a single feature
	if (int(order) > 1) & (inputData.shape[1] > 1):
		printError("Provided input as a matrix. Only vector inputs are allowed when order > 1")

	##	This section preforms the actual parameter generation	##
	#Include an extra column to include a section for biases
	#This way the process can take place entire via matrix multiplication rather than adding a vector of biases
	biasedX = None

	#start_time = time.perf_counter()

	if int(order) == 1:
		#For linear regression, include all input features
		biasedX = np.column_stack((np.ones(inputData.shape[0]), inputData))
	else:
    	#For polynomial regression, use powers of the single input feature
		polyVector = []
		for exp in range(int(order) + 1):
			feature = inputData[:, 0] ** exp
			polyVector.append(feature)

		biasedX = np.column_stack(polyVector)

	#Set the baseline for the model to be all 0's for cleanness
	modelDesc = np.zeros(biasedX.shape[1]) # Data the model contains before being rewritten

	#We are reiterating this training process max_steps times to get closer
	for i in range(int(max_steps)):
		#Find the error
		outputVector = biasedX @ modelDesc
		errorVector = outputVector - targetData

		#Calculate and subtract the gradient
		gradient = (2 / len(targetData)) * (biasedX.T @ errorVector)
		newDescToWrite = modelDesc - (float(lr) * gradient)

		#Rewrite the mode file with updated parameters
		b = newDescToWrite[0]  #Bias parameter
		w = newDescToWrite[1:] #Everything else

		modelFile = open(model, "w")			  # Model file
		modelFile.write(str(b) + " ")    		 #Write the bias first
		for index, para in enumerate(w):
			modelFile.write(str(para)) #Write a model value
			if index != len(w)-1:
				modelFile.write(" ")     		 #Add a space between each model value
		modelFile.close()
		modelDesc = newDescToWrite

		
	#end_time = time.perf_counter()

	#print("Time: " + str(end_time - start_time))

#This function sets up prediction operation
# order is a integer >= 1
# x is the input data as a text file defined as name.x.txt
# model is the required model with the trained data created as model.txt
# preds is the generated/to be modified file with all the prediction results as preds.txt
def runPrediction(order, x, model, preds):
	##	This section handles all input validation	##

	#Make sure all the values provided are valid
	if int(order) < 1:
		printError("Invalid order provided")
	if not os.path.exists(x):
		printError("Invalid input x file")
	if not os.path.exists(model):
		printError("Invalid model")

	#Create the prediction file if it doesn't already exist
	if not os.path.exists(preds):
		predFile = open(preds,"w")
		predFile.close()
		
	#Beyond here will be the actual running of the prediction operation
	#Open and read from our files
	inputFile = open(x, "r")
	modelFile = open(model, "r")
	predFile = open(preds,"w")

	#Get the data from the files
	#Input is an array of strings that are the data
	#Mode is an array of strigns that are the model coefficients
	inputData = inputFile.readlines()         #Array of strings that are the input data
	modelDesc = (modelFile.read()).split(" ") #Array of strings that are the model coefficients

	#start_time = time.perf_counter()

	#Make sure the about of items in the model are within the order
	coeffCount = (len(inputData[0].split()) + 1) if (int(order) == 1) else (int(order) + 1)
	if len(modelDesc) != coeffCount:
	    printError("Model coefficient count does not match input dimensions and order")

	#Make sure the order is not greater than 1 when we have a matrix input
	if (int(order) > 1) & (len(inputData[0].split()) > 1):
		printError("Provided input is a matrix. Only vector inputs are allowed when order > 1")

	print("Operating in prediction mode using '" + x + "' as input data using '" + model + "' as a model")
	print("Prediction output will be saved in '" + preds + "'")

	##	This section creates and writes the predictions	##
	for row in inputData:
		total =  float(modelDesc[0]) #This is what the calculated output will be saved in at the end

		#Calculate prediction for one data point
		if int(order) == 1:
			#Calculate the prediction for a linear model with several features
			for index, c in enumerate(row.split()):
				coef = float(modelDesc[index + 1].strip())
				total += coef * float(c)
		else:
			#Calculate the prediction for polynomial model with a single feature
			for index, c in enumerate(modelDesc[1:]):
				total += float(c.strip()) * pow(float(row.split()[0]), index+1)
		
		#Writing each data point to the predictions file
		predFile.write(f"{total:.3e}\n")

	#end_time = time.perf_counter()
	#print("Time: " + str(end_time - start_time))

	#Print completion update and close files
	print("The predicted output has been saved to file '" + preds + "'")
	inputFile.close()
	modelFile.close()
	predFile.close()

#This function sets up evaluation operation
# order is a integer >= 1
# x is the input data as a text file defined as name.x.txt
# y is the target data as a text file defined as name.y.txt
# model is the required model with the trained data created as model.txt
def runEvaluation(order, x, y, model):
	##	This section handles all input validation	##

	#Make sure all the values provided are valid
	if int(order) < 1:
		printError("Invalid order provided")
	if not os.path.exists(x):
		printError("Invalid input x file")
	if not os.path.exists(y):
		printError("Invalid target y file")
	if not os.path.exists(model):
		printError("Invalid model")
		
	print("Operating in evaluation mode using " + x + " as input data, " + y + " as target data, and " + model + " as the model")

	#Beyond here will be the actual running of the prediction operation
	#Open and read from our files
	inputFile = open(x, "r")
	targetFile = open(y,"r")
	modelFile = open(model, "r")

	#Get the data from the files
	inputData = inputFile.readlines()         #Array of strings that are the input data
	targetData = targetFile.readlines()       #Array of strings that are the target data
	modelDesc = (modelFile.read()).split(" ") #Array of strings that are the model coefficients

	#This is the standard error we're trying to find with this evaluation
	MSE = 0
	targetIndex = 0

	##	This section creates and writes the predictions	##

	for row in inputData:
		total =  float(modelDesc[0]) #This is what the calculated output will be saved in at the end

		#Calculate prediction for one data point
		if int(order) == 1:
			#Calculate the prediction for a linear model with several features
			for index, c in enumerate(row.split()):
				coef = float(modelDesc[index + 1].strip())
				total += coef * float(c)
		else:
			#Calculate the prediction for polynomial model with a single feature
			for index, c in enumerate(modelDesc[1:]):
				total += float(c.strip()) * pow(float(row.split()[0]), index+1)
		
		#Writing each data point to the predictions file
		MSE += pow(total - float(targetData[targetIndex]), 2)
		targetIndex += 1
	MSE = MSE/(targetIndex)


	#Tell the user what we got for MSE
	print("The calculated Mean Standard Error was found to be " + str(MSE))

	inputFile.close()
	targetFile.close()
	modelFile.close()

#This method is for printing out all the help information before running the program
#It consists of all the command line input that is expected to properly run the program
def printHelp():
	print("usage: prog2.py [-h] (--train | --predict | --eval)[--order ORDER] [--x X] [--y Y] [--model MODEL] [--preds PREDS] [--lr LR] [--max-steps MAX_STEPS]")
	print("Example: prog2.py --train --order 1 --x trainingInputs.x.txt --y trainingOutputs.y.txt --model modelFile.txt --lr 0.01 --max-steps 0")
	print("")
	print("options: ")
	print("-h, --help               Show this help message and exit")
	print("--train                  Train a model from x.txt and y.txt")
	print("--predict                Run prediction on a dev set using a trained model")
	print("--eval                   Evaluate a trained model on a dev set")
	print("-- order ORDER           Polynomial order (1 for linear model, default 1)")
	print("--x X                    Path to inputs x.txt")
	print("--y Y                    Path to targets y.txt")
	print("--model MODEL            Path to model file to be written (train) or read (predict/eval)")
	print("--preds PREDS            Path to write predictions (predict)")
	print("--lr LR                  Learning rate for gradient descent (train)")
	print("--max-steps MAX_STEPS    Max gradient steps; 0 means analytical solution (train), default 0")
	print("")
	print("train mode arguments (used with --train):")
	print("--order ORDER            Polynomial order (1 for linear model, default 1)")
	print("--x X                    Path to inputs x.txt")
	print("--y Y                    Path to targets y.txt")
	print("--model MODEL            Path to model file to be written (train) or read (predict/eval)")
	print("--lr LR                  Learning rate for gradient descent (train)")
	print("--max-steps MAX_STEPS    Max gradient steps; 0 means analytical solution (train), default 0")
	print("-a                       This will make the training operate in analytical mode. Exclude for gradient mode")
	print("")
	print("prediction mode arguments (used with --predict):")
	print("--order ORDER            Polynomial order (1 for linear model, default 1)")
	print("--x X                    Path to inputs x.txt")
	print("--model MODEL            Path to model file to be written (train) or read (predict/eval)")
	print("--preds PREDS            Path to write preictions (predict)")
	print("")
	print("evaluation mode arguments (used with --eval):")
	print("--order ORDER            Polynomial order (1 for linear model, default 1)")
	print("--x X                    Path to inputs x.txt")
	print("--y Y                    Path to targets y.txt")
	print("--model MODEL            Path to model file to be written (train) or read (predict/eval)")

#This function takes care of all of the argument setting before operation
def setArgs():
	global argc
	#Print the first argument if it's help
	if argc < 2:
		printError("Invalid argument format. Use '-h' tag during execution for help")
	if (argv[1] == "-h") | (argv[1] == "--help"):
			printHelp()
			sys.exit()

	#Add default values for optional arguments if they're missing for each possible mode
	if argv[1] == "--train":
		if "--order" not in argv:
			argv[2:2] = ["--order", "1"]
		if ("--max-steps" not in argv) & ("-a" not in argv):
			argv[12:12] = ["--max-steps", "0"]
	elif (argv[1] == "--predict") | (argv[1] == "--eval"):
		if "--order" not in argv:
			argv[2:2] = ["--order","1"]

	argc = len(argv)

	#Set the mode accoring to it's argument input
	if (argv[1] == "--train") | (argv[1] == "--predict") | (argv[1] == "--eval"):
		if(argv[1][2] == 't'):
			mode = 0
		elif(argv[1][2] == 'p'):
			mode = 1
		elif(argv[1][2] == 'e'):
			mode = 2
		else:
			#Something has gone wrong
			printError("Somehow the input argument for the mode was input as one of the given options, but failed anyway")
	else:
		#We've inputted a running mode besides what's allowed, set an error instead of running any further
		printError("Argument [1] should be in the one of the following: (--train | --predict | --eval)")

	#Different modes request different inputs
	if (mode == -1):
		printError("Mode was never initialized. Use '-h' during execution for help")
	elif mode == 0:
		#This checks and sets the arguments for training mode
		#Added in retrospect: Using an optional tag '-a' at the end operates in analytical mode
		#                     Elsewise, the training operates in gradient mode
		print(argc, argv)
		failed = (argc != 14) & (argc != 15)
		if not failed:
			failed = (argv[2] != "--order") | (argv[4] != "--x") | (argv[6] != "--y") | (argv[8] != "--model") | (argv[10] != "--lr") | (argv[12] != "--max-steps")

		if failed:
			#Training mode didn't have a proper argument structure
			#This will provide a reminder to the user for proper format structure
			print("Training mode uses argument structure: prog2.py --train [--order ORDER] [--x X] [--y Y] [--model MODEL] [--lr LR] [--max-steps MAX_STEPS] [-a]")
			sys.exit()

		#After this point, our inputs are formatting properly
		#We also need to check if the user wants to operate in analytical mode or not
		if argc == 15:
			if argv[14] == "-a":
				runAnalyticalTraining(argv[3], argv[5], argv[7], argv[9])
		else:
			runGradientTraining(argv[3], argv[5], argv[7], argv[9], argv[11], argv[13])
	elif mode == 1:
		#This checks and sets the arguments for prediction mode
		failed = argc != 10
		if not failed:
			failed = (argv[2] != "--order") | (argv[4] != "--x") | (argv[6] != "--model") | (argv[8] != "--preds")
		
		if failed:
			#Prediction mode didn't have a proper argument structure
			#This will provide a reminder to the user for proper format structure
			print("Prediction mode uses argument structure: prog2.py --predict [--order ORDER] [--x X] [--model MODEL] [--preds PREDS]")
			sys.exit()

		#After this point, our inputs are formatting properly
		runPrediction(argv[3], argv[5], argv[7], argv[9])
		
	elif mode == 2:
		#This checks and sets the arguments for evaluation mode
		failed = argc != 10
		if not failed:
			failed = (argv[2] != "--order") | (argv[4] != "--x") | (argv[6] != "--y") | (argv[8] != "--model")
		
		if failed:
			#Evaluation mode didn't have a proper argument structure
			#This will provide a reminder to the user for proper format structure
			print("Evaluation mode uses argument structure: prog2.py --eval [--order ORDER] [--x X] [--y Y] [--model MODEL]")
			sys.exit()
			
		#After this point, our inputs are formatting properly
		runEvaluation(argv[3], argv[5], argv[7], argv[9])
	else:
		printError("Mode was improperly initialized")
	
#This is a main function to ensure we're only actually calling one function in this file, and to keep it consistent I used a main function instead of just calling setArgs
def main():
	setArgs()

#Call main
main()