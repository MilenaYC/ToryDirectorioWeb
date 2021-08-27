from flask import Flask, render_template, request
import random
import json
import torch
from modelo import Red
from NLTK_uso import bolsa_palabras, tokenize


app = Flask(__name__)
device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('conocimiento.json', 'r') as f:
    conocimiento = json.load(f)
    
FILE= "data.pth"
data = torch.load(FILE)

model_state= data["model_state"]
input_size= data["input_size"]
hidden_size= data["hidden_size"]
output_size= data["output_size"]
todo=data["todo"]
indicadores=data["indicadores"]

modelo =Red(input_size, hidden_size, output_size).to(device)
modelo.load_state_dict(model_state)
modelo.eval()

nombre= "Tory"
saludo="Hola! Soy Tory tu directorio digital :) coloca 'salir' para terminar nuestra conversación"

#engine.say(saludo)
#engine.runAndWait()

@app.route("/")
def index():
	return render_template("index.html")
@app.route("/get")
def get_bot_response():
			
	sentence= request.args.get('msg')
		
	if sentence == "salir":
		#engine.say("Hasta pronto!!!")
		#engine.runAndWait()
		#print(f"{nombre}: Hasta pronto!!!")
		return str("Hasta pronto!!!")
			
	sentence= tokenize(sentence)
	X= bolsa_palabras(sentence, todo)
	X= X.reshape(1, X.shape[0])
	X= torch.from_numpy(X)

	output= modelo(X)

	_, predicted = torch.max(output, dim=1)
	indicador= indicadores[predicted.item()]
		
	proba= torch.softmax(output, dim=1)
	prob= proba[0][predicted.item()]
	# print(prob.item())
	if prob.item() < 0.75:
		for conocer in conocimiento["sabiduria"]:
			if indicador == conocer["indicador"]:
				resp=random.choice(conocer['respuestas'])
				#engine = pyttsx3.init()
				#engine.say(resp)
				#print(f"{nombre}: {resp}")
				#engine.runAndWait()
	else:
		resp="No te logro entender..."
		#engine.say("No te logro entender...")
		#print(f"{nombre}: No te logro entender...")
		#engine.runAndWait()
	return str(resp)

if __name__ == "__main__":
	app.run()