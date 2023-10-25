import random
import json
import torch
import datetime
import os
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from Backup.api import chatbot

device = torch.device('cudaa' if torch.cuda.is_available() else 'cpu')  

with open('intents.json', 'r') as f:
    intents = json.load(f)
    
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
if os.name == 'nt':
    os.system('cls')
bot_name = "Bot"
print("Mari Mengobrol! Ketik 'quit' untuk keluar")
print("")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break
    if sentence == "saran":
        print("Tulislah pertanyaan yang akan anda berikan saran atau perbaikan")
        question_word = input("You: ")
        print("Berikan jawaban yang menurut anda benar")
        tips_word = input("You: ")
        intents['intents'].append({
                "tag": "tips",
                "patterns": question_word,
                "responses": [tips_word]
                })
            
            # Menyimpan ulang konten intents.json
        with open('intents_tips.json', 'w') as f:
            json.dump(intents, f, indent=4)
        
        print("terimakasih untuk saranya")
        print("")
        continue
            
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    
        
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                

    if prob.item() < 0.75:

        # Dapatkan waktu saat ini
        waktu_sekarang = datetime.datetime.now()

        # Format waktu sesuai dengan keinginan Anda (contoh: "YYYY-MM-DD HH:MM:SS")
        format_waktu = waktu_sekarang.strftime("%Y-%m-%d %H:%M:%S")

        # Tampilkan ID dengan waktu yang sudah diformat
        print(f"{bot_name}: Maaf saya tidak mengerti maksud anda, dapatkah anda menyampaikan jawaban yang benar? (y/n)")
        new_word = input("You: ")
        
        if new_word == "y" or new_word == "Y":
            
            fix_word = input("Kata: ")
            # Menambahkan kata baru ke dalam data intents
            intents['intents'].append({
                "tag": format_waktu,
                "patterns": sentence,
                "responses": [fix_word]
                })
            
            # Menyimpan ulang konten intents.json
            with open('intents.json', 'w') as f:
                json.dump(intents, f, indent=4)
            
            print("")
            print("Mempelajari data baru...")
            print("")
            
            import json
            from nltk_utils import tokenize, stem, bag_of_words
            import numpy as np

            import torch
            import torch.nn as nn
            from torch.utils.data import Dataset, DataLoader

            from model import NeuralNet

            with open('intents.json', 'r') as f:
                intents = json.load(f)
                
            all_words = []
            tags = []
            xy = []
            for intent in intents['intents']:
                tag = intent['tag']
                tags.append(tag)
                for pattern in intent['patterns']: 
                    w = tokenize(pattern)
                    all_words.extend(w)
                    xy.append((w, tag))
                    
                ignore_words = ['?', '!', '.', ',']
                all_words = [stem(w) for w in all_words if w not in ignore_words]
                all_words = sorted(set(all_words))
                tags = sorted(set(tags))
                
                x_train = []
                y_train = []
                for (pattern_sentence, tag) in xy:
                    bag = bag_of_words(pattern_sentence, all_words)
                    x_train.append(bag)
                    
                    label = tags.index(tag)
                    y_train.append(label) 
                    
            x_train = np.array(x_train)
            y_train = np.array(y_train)

            class ChatDataset(Dataset):
                def __init__(self):
                    self.n_samples = len(x_train)
                    self.x_data = x_train
                    self.y_data = y_train
                
                #dataset[idx]    
                def __getitem__(self, index):
                    return self.x_data[index], self.y_data[index]
                
                def __len__(self):
                    return self.n_samples

            batch_size = 8
            hidden_size = 8
            output_size = len(tags)
            input_size = len(x_train[0])
            learning_rate = 0.001
            num_epochs = 1000

                
            dataset = ChatDataset()
            train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            device = torch.device('cudaa' if torch.cuda.is_available() else 'cpu')  
            model = NeuralNet(input_size, hidden_size, output_size).to(device)

            #loss
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

            for epoch in range(num_epochs):
                for (words, labels) in train_loader:
                    words = words.to(device)
                    labels = labels.to(dtype=torch.long).to(device)
                    
                    #forward
                    outputs = model(words)
                    loss = criterion(outputs, labels)   
                    
                    #back
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                if (epoch+1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss={loss.item():.4f}')
                    
            print(f'final loss, loss={loss.item():.4f}')

            data = {
                "model_state": model.state_dict(),
                "input_size": input_size,
                "output_size": output_size,
                "hidden_size": hidden_size,
                "all_words": all_words,
                "tags": tags
            }

            FILE = "data.pth"
            torch.save(data, FILE)

            if os.name == 'nt':
                os.system('cls')
                
            print("Terimakasih atas saranya, sekarang bot menjadi lebih pintar")
            print("")
            
            print("Mari Mengobrol! Ketik 'quit' untuk keluar")
    
        else:
            print("")