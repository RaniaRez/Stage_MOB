from flask import Flask
from flask_restful import Api, Resource, reqparse, abort, fields, marshal_with
import json
from depression_index import predict_depression
from transformers import pipeline
import torch
import pandas as pd

emotions = ['Agreeableness', 'Anger', 'Anticipation','Arrogance','Disagreeableness','Disgust','Fear','Gratitude','Happiness','Humility','Love','Optimism','Pessimism', 'Regret','Sadness','Shame','Shyness', 'Surprise','Trust','neutral']
interests = ['Science', 'Design', 'Business', 'Aviation', 'Video games', 'Board games', 'Card games', 'Concerts', 'Theatre', 'Festivals', 'Classical music', 'Dance', 'Movies', 'Animation', 'Comedy', 'Documentaries', 'Horror', 'Sci-fi', 'Electro music', 'Pop music', 'Rock music', 'Metal music', 'Books', 'Comics', 'Philosophy', 'Magazines', 'Novels', 'Manga', 'Tv shows', 'Romance', 'Family', 'Friendship', 'Marriage', 'Parenting', 'Sports', 'Meditation', 'Fitness', 'Yoga', 'Cuisine', 'Baking', 'Coffee', 'Tea', 'Desserts', 'Acting', 'Art', 'Drawing', 'Painting', 'Music instruments', 'Photography', 'Writing', 'Sculpture', 'Fine arts', 'Animals', 'Travel', 'Nature', 'Cars', 'Beauty', 'Cosmetics', 'Hair', 'Fashion', 'Camping', 'Surfing', 'Technology', 'Phones', 'Computers', 'Tablets', 'Cameras', 'Consoles', 'Audio']

app = Flask(__name__)
api = Api(app)

item_put_args = reqparse.RequestParser()
item_put_args.add_argument("text", type=str, help="Journal entry", location='form')
item_put_args.add_argument("name", type=str, help="Client name", location='form')
item_put_args.add_argument("age", type=int, help="Client age", location='form')


class Journal(Resource):
        def get_depression_index(self, ID):
                journal_entry = item_put_args.parse_args()["text"]
                return predict_depression(journal_entry)
        def get_client(self,ID):
                with open("database.json", "r+",encoding="utf-8") as db : 
                       database = json.load(db)

                if ID > 0:
                        #check if exists
                        idObj = f'{ID}'
                        if idObj in database.keys():
                                object = {}                        
                                object[idObj] = database[idObj]
                                return object, 200
                        else: abort(404, message="Could not find client with that id")
                return database[ID]
        def get_painting(self,ID):
                pass
                journal_entry = item_put_args.parse_args()["text"]
                # # do the transformers get the values named results
                classifier_pipeline = pipeline ("zero-shot-classification", model = "facebook/bart-large-mnli")
                emotion_result = classifier_pipeline(journal_entry, emotions,  multi_label=True)
                model = torch.load("depression2.model")
                list_of_dicts =dict(zip(emotion_result["labels"], emotion_result["scores"]))
                list_of_dicts = {key: value for key, value in sorted(list_of_dicts.items())}
                emotions_list = list(list_of_dicts.values())
                res = model.predict(emotions_list) #fill it
                paintings = pd.read_csv("WikiArt-info.csv", sep = ";")
                art_info = paintings[paintings['ID'] == res[0]]
                
                return art_info

        def put(self, ID):
                pass
                with open("database.json", "r+",encoding="utf-8") as db : 
                       database = json.load(db)

                if ID > 0:
                        #check if exists
                        idObj = f'{ID}'
                        if idObj in database.keys():
                                object = {}                        
                                object[idObj] = database[idObj]
                                return object, 200
                        else: abort(404, message="Could not find client with that id")
                current_client = database[ID]
                # first get the user we want to modify DONE
                journal_entry = item_put_args.parse_args()["text"]

                # get the journal entry from arguments DONE

                # calculate the depression index from text (depression2.model) DONE
                depression_index = predict_depression(journal_entry)

                # get emotions from text (transformers)
                classifier_pipeline = pipeline ("zero-shot-classification", model = "facebook/bart-large-mnli")
                emotion_result = classifier_pipeline(journal_entry, emotions,  multi_label=True)

                # get interests from text (transformers)
                interest_result = classifier_pipeline(journal_entry, interests,  multi_label=True)
                # compile journal entry, date, time with list of emotions into a dict, make sure to increment the id from the past journal entry
                list_of_dicts =dict(zip(emotion_result["labels"], emotion_result["scores"]))
                list_of_dicts = {key: value for key, value in sorted(list_of_dicts.items())}
                emotions_list = []
                for key in list_of_dicts.keys():
                        emotions_list.append({"Emotion_name": key, "percentage": list_of_dicts[key]})
                
                # add the new journal and stat we just calculated (the dict) into the list (stats_history) of the user
                journal = {"ID": "0005",
                        "Date": "12-12-2012",
                        "Time" : "12:12",
                        "Entry" : journal_entry,
                        "Emotions" : emotions_list
                        #, add positivity index
                        }
                # get only interests with score >0.7
                counter = 0
                new_interests = current_client["Interests"].copy()
                for interest_score in interest_result["scores"]:
                        if (interest_score > 0.7):
                                if (interest_result["labels"][counter] not in current_client["Interests"]):
                                        new_interests.append(interest_result["labels"][counter])
                

                # check if the interest is in user's interest list
                with open("database.json", "r+",encoding="utf-8") as db : 
                                updt = json.dumps(database, indent=4, ensure_ascii=False)
                                db.seek(0)
                                db.write(updt)
                                db.truncate()
                
                # add interests that aren't in user's interest list
                return journal
                # update the user by using a code similar to this 

                ########### IMPORTANT
                # note that this code is from another api with a completely different database, the code is not guaranteed to work as is, test and change accordingly
                # with open("data.json", "r+",encoding="utf-8") as db : 
                #                 updt = json.dumps(database, indent=4, ensure_ascii=False)
                #                 db.seek(0)
                #                 db.write(updt)
                #                 db.truncate()
                # else: abort(404, message="Could not find item with that id")
                # the full code is found here: https://github.com/super-cinnamon/E-commerce-REST-API-app/blob/main/API.py

        def post(self):
                with open("database.json", "r+",encoding="utf-8") as db : 
                       database = json.load(db)
                all_ids = database.keys()
                liste=[]
                for ids in all_ids:
                        liste.append(int(ids))
                new_id = max(liste) + 1
                new_client = {
                        "Name" :  item_put_args.parse_args()["name"],
                        "Age" :  item_put_args.parse_args()["Age"],
                        "Stats_history" :  [],
                        "Subscriber" :  "no",
                        "Interests" :  []
                }

                database[f"{new_id}"] = new_client
                with open("database.json", "r+",encoding="utf-8") as db : 
                        updt = json.dumps(database, indent=4, ensure_ascii=False)
                        db.seek(0)
                        db.write(updt)
                        db.truncate()
                
        def delete(self,ID):
                #database
                with open("database.json", "r+",encoding="utf-8") as db : 
                       database = json.load(db)
                db.close()    
                idObj = f'{ID}'
                #we delete the item
                if idObj in database.keys():
                        database.pop(idObj)
                        #database
                        with open("database.json", "r+",encoding="utf-8") as db : 
                                updt = json.dumps(database, indent=4, ensure_ascii=False)
                                db.seek(0)
                                db.write(updt)
                                db.truncate()
                else: abort(404, message="Could not find item with that id")




api.add_resource(Journal, "/journal/<int:ID>")

if __name__ == "__main__":
        ### open the db file?
        app.run(debug=True)
