######################## AGE ######################


today = datetime.date.today()
born = datetime.date(2021, 10, 14)
timespan = today - born

around_for = str(("I have been around for", timespan))


data = {'intents': []}
if os.path.exists('intents.json'):
    with open('intents.json', 'r') as f:
        data = json.load(f)

data['intents'].append({
    'tag': "age",
    'patterns': [
        "Are you old?",
        "Tell me your age"
    ],
    'response': around_for
})

with open('intents.json', 'w') as training:
    json.dump(data, training)

    {
        "tag": "friends",
        "patterns": [
            "friend fact",
            "about friend",
            "tell me about friend"
        ],
        "responses": [
            "Brandon is a Pinoy",
            "Sooks is a Donkey",
            "Ry is a flyguy"
        ],
        "context_set": ""
    },
