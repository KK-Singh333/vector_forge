# build_ground_truth.py

import requests
import json
import time
import pickle

API_URL = "http://127.0.0.1:8068/search"
K = 10
SLEEP = 1.5

QUERIES = [

# SECTION A — 222 RAJPUR
"For 222 Rajpur, Dehradun, how many total residences are planned and over how many acres is the project spread?",
"At 222 Rajpur, what types of residences are available?",
"Is 222 Rajpur adjacent to any forest area? If yes, which one?",
"What are the views offered from residences at 222 Rajpur, Dehradun?",
"How far is Jolly Grant Airport from 222 Rajpur, Dehradun?",
"What is the distance between 222 Rajpur and The Doon School?",
"How long does it take to reach Pacific Mall from 222 Rajpur?",
"Is 222 Rajpur close to Max Super Specialty Hospital?",
"How many Townhouse units are available at 222 Rajpur?",
"What is the built-up area and carpet area of a Townhouse at 222 Rajpur?",
"Does the Townhouse at 222 Rajpur include a sky court?",
"What is the ceiling height in the Townhouse units at 222 Rajpur?",
"How many parking spaces are provided with each Townhouse at 222 Rajpur?",
"How many Courtyard Villas are available at 222 Rajpur?",
"What is the plot size range for Courtyard Villas at 222 Rajpur?",
"Do the Courtyard Villas at 222 Rajpur include staff accommodation?",
"What is the terrace size of a Courtyard Villa at 222 Rajpur?",
"How many Forest Villas are available at 222 Rajpur?",
"What is the built-up area of a Forest Villa at 222 Rajpur?",
"Do Forest Villas at 222 Rajpur have private elevators?",
"What special landscape feature is included in the lower ground floor of Forest Villas at 222 Rajpur?",
"Does 222 Rajpur provide round-the-clock security?",
"What wellness or nature-focused amenities are offered at 222 Rajpur?",
"Does 222 Rajpur offer power backup and uninterrupted water supply?",
"Is there a private orchard at 222 Rajpur?",

# SECTION B — MAX TOWERS
"What is the total super built-up area of Max Towers, Noida?",
"How many office floors and amenity floors are there in Max Towers?",
"What is the typical floor plate size at Max Towers?",
"What is the floor-to-floor height at Max Towers?",
"What green rating has Max Towers achieved?",
"Does Max Towers offer on-site wastewater treatment?",
"What is the coefficient of performance (COP) of the chiller system at Max Towers?",
"Does Max Towers support electric vehicle parking?",
"Does Max Towers have a swimming pool?",
"What kind of fitness facilities are available at Max Towers?",
"Does Max Towers provide daycare facilities?",
"What air treatment system is used in Max Towers?",
"Where is Max Towers located?",
"Is Max Towers within walking distance of a metro station?",
"Does Max Towers have direct access to the DND Flyway?",
"What type of façade glass is used in Max Towers?",
"What is the solar heat gain coefficient of the façade at Max Towers?",
"What percentage of regular occupied space at Max Towers gets line-of-sight to the outside?",

# SECTION C — MAX HOUSE
"What is the total super built-up area of Max House, Okhla?",
"How many tenant floors are there in Max House?",
"What is the typical floor plate size at Max House?",
"What is the green rating of Max House?",
"How far is Max House, Okhla from the Okhla NSIC Metro Station?",
"How far is Max House from IGI Airport?",
"Is Max House within walking distance of a metro station?",
"What façade material is used in Max House?",
"What is the floor-to-ceiling height at Max House?",
"Does Max House use double-glazed windows?",
"What air treatment technology is used in Max House?",
"Is Max House LEED certified?",
"Does Max House incorporate biophilic design principles?",

# SECTION D — CROSS PROPERTY
"Which property among 222 Rajpur, Max Towers, and Max House is purely residential?",
"Which property is located in Dehradun: 222 Rajpur or Max Towers?",
"Between Max Towers and Max House, which one has a higher LEED certification?",
"Compare the typical floor plate size of Max Towers and Max House.",
"Which property has a larger total built-up area: Max Towers or Max House?",
"Which property has more tenant floors: Max Towers or Max House?",
"Which property is closer to a metro station: Max House or Max Towers?",
"Between 222 Rajpur and Max House, which property is closer to an airport?",
"Which property offers direct access to the DND Flyway: Max Towers or Max House?",
"Which property has LEED Platinum certification: Max Towers or Max House?",
"Which properties use advanced air treatment systems: Max Towers, Max House, or both?",
"Which property offers on-site wastewater treatment: Max Towers or Max House?",
"Which property offers a swimming pool: Max Towers or Max House?",
"Does 222 Rajpur offer wellness amenities comparable to Max Towers?",
"Which property explicitly mentions decompression spaces: Max Towers or Max House?",

# SECTION E — CLIENT SIMULATION
"I’m looking for a 4-bedroom villa with staff accommodation in 222 Rajpur — which unit type should I consider?",
"My company needs a 25,000 sq. ft. office in Noida — can Max Towers accommodate this on a single floor?",
"We want an office in Delhi with LEED Gold certification — is Max House, Okhla suitable?",
"I want a residential property near the forest with private garden space — does 222 Rajpur offer this?",
"We are a wellness-focused company — between Max Towers and Max House, which better supports employee wellbeing?",
"I need an office within walking distance of the metro in Delhi — is Max House a good option?",
"Which property among 222 Rajpur, Max Towers, and Max House offers private elevators?",
"If sustainability is a top priority, should I choose Max Towers or Max House?",
"I need a property with daycare facilities — which of these three properties provides that?",

# SECTION F — PARAPHRASES
"Which of the three developments is a housing project rather than an office building?",
"Identify the project that is not meant for commercial office use.",
"Among the three properties, which one is exclusively residential in nature?",
"Between Max Towers and Max House, which holds the higher level of LEED certification?",
"If sustainability certification level is the deciding factor, which property ranks highest?",
"Which development has achieved Platinum-level green certification?",
"Which office property can employees walk to from a metro station?",
"Identify the development located within walking distance of a metro stop.",
"Between the Noida and Okhla projects, which one offers closer metro access?",
"Which project is larger in overall constructed area: Max Towers or Max House?",
"Between the Delhi and Noida office developments, which spans more total square footage?",
"Which property has the greater overall scale in terms of built-up space?",
"Which property includes an indoor swimming facility?",
"Identify the development that provides decompression or relaxation spaces.",
"If employee wellness is a priority, which property explicitly supports it through facilities?",

# SECTION G — NEGATIVE
"Which property among the three includes a helipad?",
"Which development offers a golf course within the premises?",
"Is any of the properties located in Mumbai?",
"Which project provides co-living or serviced apartments?",
"What is the rental yield percentage of Max Towers?",
"Which property has a shopping mall attached to it?",
"Do any of the properties include a five-star hotel?",
"Which development offers beachfront views?",
"Is there a data center facility mentioned in any of the properties?",
"Which property includes an amusement park or entertainment zone?",

# SECTION H — AMBIGUOUS
"What is the total area?",
"How many floors does it have?",
"What certification does it hold?",
"How far is it from the airport?",
"Does it offer parking?"
]
ground_truth = []
with open("data.pkl", "rb") as f:
    chunk_list = pickle.load(f)
for i,q in enumerate(QUERIES):
    response = requests.post(API_URL, json={
                    "user_id": 1,
                    "query": q,
                    "k": 5
                }).json()

    print("\n\nQUERY:", q)
    print("------")
    id_map={}
    chunks_list.append(response["results"])
    time.sleep(SLEEP)
    for i, s in enumerate(response["results"]):
        print(f"[{i}] chunk_id:", s["chunk_id"])
        id_map[i]=s["chunk_id"]
        print(s["text"][:400])
        print()

    chosen = input("Enter correct chunk_id(s), comma separated: ")

    gt_ids = [int(x.strip()) for x in chosen.split(",")]

    ground_truth.append({
        "query": q,
        "ground_truth_chunk_ids": [id_map[i] for i in gt_ids]
    })

    time.sleep(SLEEP)

with open("ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=2)
# with open("data.pkl", "wb") as f:
#     pickle.dump(chunks_list, f)
