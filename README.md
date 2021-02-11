"# AiChatBot" 
This is a very simple Chatbot that uses a neuronal network to respond to the user imput 
This bot was created for an ecomerce site , but with the help of the intents folder you can prepare the robot 
for any scenario

The project is structured as it follows :
				Intents.json : here the scenarios that the robot can respond to are given in json format
				
				TrainigChatbot : This file is responsible for creating and taining the neural network with the intents file , and serializing the model for 
								 quick acess latter to it
				
				Preprocessing :  This file has the role to take the input from the intends file and process the sentaces in order to obtain a numerical 
				   				 vector for the neural network to work with ( the bag of words method was used) 
				
				AiChatBot:  This is simply the main folder that loads the neural network , and start a loop for the conversation to take place in , the customer 
							asks a question and the bot answears 