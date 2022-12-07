import requests

BASE = "http://127.0.0.1:5000/"


#response = requests.delete(BASE + "auction/1")
response = requests.get(BASE +f"journal/101",{"text":"I am not feeling great"})
print(response.json())
