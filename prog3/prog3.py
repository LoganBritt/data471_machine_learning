import sys
import os
import numpy as np
import time

# Easy system referenced values for processing arguments
argc = len(sys.argv)
argv = sys.argv

# Mode is the current setting for which version of the program we're running
# 0 is training mode
# 1 is predicting mode
# 2 is evaluation mode
# -1 is default and an invalid mode that must be set during initialilzation
# It's never really important that it's global, I expected to put all processing variables as global ones but java has proned me to not do that
mode = -1


# This is an error function for simple error throwing
def printError(msg):
	print("An Error Has Occured: " + msg)
	sys.exit()

# This function sets up training for multinomial logistic regression model training
# trainX is the training input file
# trainY is the training target output file
# devX is the development input file
# devY is the development target output file
# model is the model file
# lr is the learning rate
# max_steps is the max number of steps taken during training
# mb is the minibatch size
# patience is the number of dev evaluations to go through without improvement before quitting
# dev_freq is frequency of gradient steps between each evaluation
# verbose is a toggle to enable verbose mode, where minibatch loss is printed
def runMultinomialTraining(trainX, trainY, devX, devY, model, lr, max_steps, mb, patience, dev_freq, verbose):
	## This section handles all input validation and file openage	##
	# I've passed all error handling into a helper method for cleanliness
	checkTrainingError(trainX, trainY, devX, devY, model, lr, max_steps, mb, patience, dev_freq)
	
	##	Set initialized values and handle other error handling	##
 
	#Change the number values from strings to their appropriate values for simplicity
	lr = float(lr)
	max_steps = int(max_steps)
	mb = int(mb)
	patience = int(patience)
	dev_freq = int(dev_freq)
 
	print("Operating in multinomial logistic regession model training mode")
	print("Model will be saved to file: " + model)
	if verbose: print("-Verbose Mode-")
 
	trainXData = np.load(trainX)
	trainYData = np.load(trainY)
	devXData = np.load(devX)
	devYData = np.load(devY)
 
  	#Get the number of features and classes
	num_features = trainXData.shape[1]
	num_classes = np.max(trainYData) + 1

    #Make sure the target labels fit the model's classes

	if np.any(trainYData < 0): printError("Training targets cannot contain negative class labels.")
	if np.any(devYData < 0) or np.any(devYData >= num_classes): printError("Development targets contain invalid class labels.")

    #Make sure at least one full minibatch is possible
	if mb > trainXData.shape[0]: printError("Minibatch size cannot exceed the number of training examples.")

    #Initialize weights and biases
	W = np.zeros((num_features, num_classes), dtype=np.float64)
	b = np.zeros(num_classes, dtype=np.float64)

    #Track training progress
	update = 0
	epoch = 0
	bestDevLoss = float("inf")
	evalsSansImprovement = 0
	stoppedEarly = False
 
	## Beyond here we're actually preforming model training	##

    #Repeat training until max_steps is reached or stopped early
	while update < max_steps and not stoppedEarly:
		epoch += 1

        #Shuffle training examples before each epoch
		order = np.random.permutation(trainXData.shape[0])
		shuffledX = trainXData[order]
		shuffledY = trainYData[order]

        #Repeat for every complete minibatch
		for start in range(0, len(shuffledY) - mb + 1, mb):
			if update >= max_steps: break

            #Get the current minibatch
			XBatch = shuffledX[start:start + mb]
			YBatch = shuffledY[start:start + mb]

			##	This section preforms gradient decent on the minibatch	##
   
            #Calculate class probabilities
			results = XBatch @ W + b
			results -= np.max(results, axis=1, keepdims=True)
			probs = np.exp(results)
			probs /= np.sum(probs, axis=1, keepdims=True)

            #Calculate average cross-entropy loss
			batchLoss = -np.mean(np.log(probs[np.arange(mb), YBatch]))

            #Calculate gradients for weights and biases
			probs[np.arange(mb), YBatch] -= 1
			probs /= mb

			gradW = XBatch.T @ probs
			gradb = np.sum(probs, axis=0)

            #Update ending results values
			W -= lr * gradW
			b -= lr * gradb
			update += 1

            #Print minibatch loss when verbose mode is enabled
			if verbose: print(f"{batchLoss:.3f} batch, update {update}, epoch {epoch}")

			##	This section handles evaluation on the recently calculated parameters	##

            #Evaluate the model at the requested frequency
			if update % dev_freq == 0:

                #Calculate probabilities on the development set
				devResults = devXData @ W + b
				devResults -= np.max(devResults, axis=1, keepdims=True)
				devProb = np.exp(devResults)
				devProb /= np.sum(devProb, axis=1, keepdims=True)

                #Calculate development loss and accuracy
				devLoss = -np.mean(np.log(devProb[np.arange(len(devYData)), devYData]))

				predictions = np.argmax(devProb, axis=1)
				devAccuracy = np.mean(predictions == devYData)

				print(f"dev loss={devLoss:.3f} acc={devAccuracy:.3f} "f"(as of update {update}, epoch {epoch})")

                #Save the model if development loss has improved
				if devLoss < bestDevLoss:
					bestDevLoss = devLoss

                    #Save W and b in the required binary NPZ format
					with open(model, "wb") as modelFile: np.savez(modelFile, W=W, b=b)
					evalsSansImprovement = 0

				else:
					evalsSansImprovement += 1

                #Stop if patience is exceeded
				if evalsSansImprovement > patience:
					print("Early stopping: development loss stopped improving.")
					stoppedEarly = True
					break

    #Evaluate at the end if the final update wasn't already evaluated
	if update > 0 and update % dev_freq != 0 and not stoppedEarly:
		devResults = devXData @ W + b
		devResults -= np.max(devResults, axis=1, keepdims=True)
		devProb = np.exp(devResults)
		devProb /= np.sum(devProb, axis=1, keepdims=True)

		devLoss = -np.mean(np.log(devProb[np.arange(len(devYData)), devYData]))
		devAccuracy = np.mean(np.argmax(devProb, axis=1) == devYData)

		print(f"dev loss={devLoss:.3f} acc={devAccuracy:.3f} "f"(as of update {update}, epoch {epoch})")

		if devLoss < bestDevLoss:
			bestDevLoss = devLoss
			with open(model, "wb") as modelFile: np.savez(modelFile, W=W, b=b)

	print(f"Training finished after {update} updates.")
	print(f"Best development loss: {bestDevLoss:.3f}")

	
#This function takes in all th parameters and makes sure they're all valid
# Parameters are defined as in runMultinomialTraining
def checkTrainingError(trainX, trainY, devX, devY, model, lr, max_steps, mb, patience, dev_freq):
	#Make sure all the values provided are valid
	if not os.path.exists(trainX): printError("Invalid training input file: " + trainX)
	if not os.path.exists(trainY): printError("Invalid training target file: " + trainY)
	if not os.path.exists(devX): printError("Invalid dev input file: " + devX)
	if not os.path.exists(devY): printError("Invalid dev target file: " + devY)
	if float(lr) <= 0: printError("Invalid learning rate: " + lr)
	if int(max_steps) < 1: printError("Invalid max steps: " + max_steps)
	if int(mb) < 1: printError("Invalid minibatch size: " + mb)
	if int(patience) < 0: printError("Invalid patience limit: " + patience)
	if int(dev_freq) < 1: printError("Invalid dev frequency: " + dev_freq)

	##	Ensure file formatting is correctly oriented	##
	trainXData = np.load(trainX)
	trainYData = np.load(trainY)
	devXData   = np.load(devX)
	devYData   = np.load(devY)
	
	# Check array dimensions
	if trainXData.ndim != 2: printError("Training inputs must be a 2D array.")
	if trainYData.ndim != 1: printError("Training targets must be a 1D array.")
	if devXData.ndim != 2: printError("Development inputs must be a 2D array.")
	if devYData.ndim != 1: printError("Development targets must be a 1D array.")
	
	# Check data types
	if trainXData.dtype != np.float64: printError("Training inputs must be float64.")
	if trainYData.dtype != np.int64: printError("Training targets must be int64.")
	if devXData.dtype != np.float64: printError("Development inputs must be float64.")
	if devYData.dtype != np.int64: printError("Development targets must be int64.")
	
	# Check that inputs and targets have matching numbers of examples
	if trainXData.shape[0] != trainYData.shape[0]: printError("Training inputs and targets have different numbers of examples.")
	if devXData.shape[0] != devYData.shape[0]: printError("Development inputs and targets have different numbers of examples.")
	
	# Check that training and development inputs have the same number of features
	if trainXData.shape[1] != devXData.shape[1]: printError("Training and development inputs have different numbers of features.")


#This function sets up prediction operation
# x is the input data as a text file defined as name.x.txt
# model is the required model with the trained data created as model.txt
# preds is the generated/to be modified file with all the prediction results as preds.txt
def runPrediction(x, model, preds):
	##	This section handles all input validation and file opanage	##

	#Make sure all the values provided are valid
	if not os.path.exists(x):
		printError("Invalid input x file: " + x)
	if not os.path.exists(model):
		printError("Invalid model: " + model)

	#Create the prediction file if it doesn't already exist
	if not os.path.exists(preds):
		predFile = open(preds,"w")
		predFile.close()
		
	#Open and read from our files
	X = np.load(x)
	modelData = np.load(model)
	
	W = modelData["W"]
	b = modelData["b"]

	##	Beyond here will be the actual running of the prediction operation	##
	#Calculate our values
	results = X @ W + b
	results -= np.max(results, axis=1, keepdims=True)

	#Preform softmax
	params = np.exp(results)
	params /= np.sum(params, axis=1, keepdims=True)

	##	Save the predicted values to the file preds	and clean up##
	np.save(preds, params.astype(np.float64))
	print("Predicted output has been saved to " + preds)
	modelData.close()


#This function sets up evaluation operation
# x is the input data as a text file defined as name.x.txt
# y is the target data as a text file defined as name.y.txt
# model is the required model with the trained data created as model.txt
def runEvaluation(x, y, model):
	##	This section handles all input validation and file openage	##

	#Make sure all the values provided are valid
	if not os.path.exists(x):
		printError("Invalid input x file: " + x)
	if not os.path.exists(y):
		printError("Invalid target y file: " + y)
	if not os.path.exists(model):
		printError("Invalid model: " + model)
		
	print("Operating in evaluation mode using " + x + " as input data, " + y + " as target data, and " + model + " as the model")

	#Open and read from our files
	X = np.load(x)
	Y = np.load(y)
	modelData = np.load(model)
	
	W = modelData["W"]
	b = modelData["b"]

	##	Beyond here will be the actual running of the prediction operation	##
	#Calculate our values
	results = X @ W + b
	results -= np.max(results, axis=1, keepdims=True)

	#Preform softmax
	params = np.exp(results)
	params /= np.sum(params, axis=1, keepdims=True)

	#Choose the most likely class for each input
	predictions = np.argmax(params, axis=1)

	#Get accuracy
	accuracy = np.mean(predictions == Y)

	##	Print the final results and clean up	##
	print(f"{accuracy:.3f}")
	modelData.close()

#This method is for printing out all the help information before running the program
#It consists of all the command line input that is expected to properly run the program
def printHelp():
	print("usage: prog3.py [-h] (--train | --predict | --eval) [--trainx TRAINX] ] [--trainy TRAINY] [--devx DEVX] [--devy DEVY] [--model MODEL] [--preds PREDS] [--lr LR] [--max-steps MAX_STEPS] [--mb MB] [--patience PATIENCE] [--dev_freq DEV_FREQ] [--verbose]")
	print("")
	print("options: ")
	print("  -h, --help               Show this help message and exit")
	print("  --train                  Train a multidimensional logistic regression model model from X to Y")
	print("  --predict                Run prediction on a dev/test set using a trained model")
	print("  --eval                   Evaluate a trained model on a dev/test set (prints accuracy)")
	print("  --trainx TRAINX          Path to training inputs npy file")
	print("  --trainy TRAINY          Path to training targets npy file (class labels)")
	print("  --devx DEVX              Path to dev inputs npy file")
	print("  --devy DEVY              Path to dev targets npy file (class labels)")
	print("  --model MODEL            Path to model file to be written (train) or read (predict/eval), default \"model.npz\"")
	print("  --preds PREDS            Path to write predictions npy file (predict)")
	print("  --lr LR                  Learning rate for gradient descent")
	print("  --max-steps MAX_STEPS    Max MB-SGD steps, default 5000")
	print("  --mb MB                  Minibatch size, default 32")
	print("  --patience PATIENCE      Early stopping patience in number of dev evals without improvement, default 10")
	print("  --dev_freq DEV_FREQ      Evaluate on the dev set every this number of gradient steps, default 500")
	print("  --verbose                Enable verbose mode, in which each minibatch loss is printed")
	print("")
	print("train mode arguments (used with --train):")
	print("  --trainx TRAINX          Path to training inputs npy file")
	print("  --trainy TRAINY          Path to training targets npy file (class labels)")
	print("  --devx DEVX              Path to dev inputs npy file")
	print("  --devy DEVY              Path to dev targets npy file (class labels)")
	print("  --model MODEL            Path to model file to be written (train) or read (predict/eval), default \"model.npz\"")
	print("  --lr LR                  Learning rate for gradient descent")
	print("  --max-steps MAX_STEPS    Max MB-SGD steps, default 5000")
	print("  --mb MB                  Minibatch size, default 32")
	print("  --patience PATIENCE      Early stopping patience in number of dev evals without improvement, default 10")
	print("  --dev_freq DEV_FREQ      Evaluate on the dev set every this number of gradient steps, default 500")
	print("  --verbose                Enable verbose mode, in which each minibatch loss is printed")
	print("")
	print("prediction mode arguments (used with --predict):")
	print("  --devx DEVX              Path to dev inputs npy file")
	print("  --model MODEL            Path to model file to be written (train) or read (predict/eval), default \"model.npz\"")
	print("  --preds PREDS            Path to write predictions npy file (predict)")
	print("")
	print("evaluation mode arguments (used with --eval):")
	print("  --devx DEVX              Path to dev inputs npy file")
	print("  --devy DEVY              Path to dev targets npy file (class labels)")
	print("  --model MODEL            Path to model file to be written (train) or read (predict/eval), default \"model.npz\"")


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
		if "--model" not in argv:
			argv[10:10] = ["--model", "model.npz"]
		if "--max-steps" not in argv:
			argv[14:14] = ["--max-steps", "5000"]
		if "--mb" not in argv:
			argv[16:16] = ["--mb", "32"]
		if "--patience" not in argv:
			argv[18:18] = ["--patience", "10"]
		if "--dev_freq" not in argv:
			argv[20:20] = ["--dev_freq", "500"]
	elif (argv[1] == "--predict"):
		if "--model" not in argv:
			argv[4:4] = ["--model", "model.npz"]
		if "--preds" not in argv:
			argv[6:6] = ["--preds", "preds.npy"]
	elif (argv[1] == "--eval"):
		if "--model" not in argv:
			argv[6:6] = ["--model", "model.npz"]

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
		#This checks the parameters in the training mode
		usingVerbose = False

		#Checking argument validiity
		failed = argc < 22
		if not failed:
			failed = (argv[2] != "--trainx") | (argv[4] != "--trainy") | (argv[6] != "--devx") | (argv[8] != "--devy") | (argv[10] != "--model") | (argv[12] != "--lr") | (argv[14] != "--max-steps") | (argv[16] != "--mb") | (argv[18] != "--patience") | (argv[20] != "--dev_freq")
		if failed:
			print("Failed at 1")

		if not failed:
			
			#We have more than 22 arguments. If the 23rd is verbose, it's fien. Elsewise, fail
			if argc == 23:
				#There are 23
				if argv[22] == "--verbose":
					#The 23rd is verbose, which is fine
					usingVerbose = True
				else:
					#The 23rd is not verbose, which is not fine
					failed = True
			elif argc > 23:
				#There are more than 23, which is not allowed
				failed = True

		if failed:
			#Training mode didn't have a proper argument structure
			#This will provide a reminder to the user for proper format structure
			#print("argv: " + str(argv))
			#print("argc: " + str(argc))
			print("Training mode uses argument structure: prog3.py --train [--trainx TRAINX] [--trainy TRAINY] [--devx DEVX] [--devy DEVY] [--model MODEL] [--lr LR] [--max-steps MAX_STEPS] [--mb MB] [--patience PATIENCE] ] [--dev_freq DEV_FREQ] [--verbose]")
			sys.exit()

		runMultinomialTraining(argv[3], argv[5], argv[7], argv[9], argv[11], argv[13], argv[15], argv[17], argv[19], argv[21], usingVerbose)
	elif mode == 1:
		#This checks the parameters in the prediction mode

		#Checking argument validity
		failed = argc < 8
		if not failed:
				failed = (argv[2] != "--devx") | (argv[4] != "--model") | (argv[6] != "--preds")

		if failed:
			#Prediction mode didn't have a proper argument structure
			#This will provide a reminder to the user for proper format structure
			print("Prediction mode uses argument structure: prog3.py --predict [--devx DEVX] [--model MODEL] [--preds PREDS]")
			sys.exit()

		#After this point, our inputs are formatting properly
		runPrediction(argv[3], argv[5], argv[7])
	elif mode == 2:
		#This checks the parameters in the evaluation mode

		#Checking argument validity
		failed = argc < 8
		if not failed:
				failed = (argv[2] != "--devx") | (argv[4] != "--devy") | (argv[6] != "--model")

		if failed:
			#Evaluation mode didn't have a proper argument structure
			#This will provide a reminder to the user for proper format structure
			print("Evaluation mode uses argument structure: prog3.py --eval [--devx DEVX] [--devy DEVY] [--model MODEL]")
			sys.exit()
			
		#After this point, our inputs are formatting properly
		runEvaluation(argv[3], argv[5], argv[7])
	else:
		printError("Mode was improperly initialized")
	
#This is a main function to ensure we're only actually calling one function in this file, and to keep it consistent I used a main function instead of just calling setArgs
def main():
	setArgs()

#Call main
main()