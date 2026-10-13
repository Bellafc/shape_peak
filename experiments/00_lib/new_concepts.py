"""新概念：城市（首都 / 地标）、情緒、顏色。每類 6 條線索 × 3 個問法；線索裡不出現答案詞。"""
CITY = {
    "Paris": ["the capital of France", "the Eiffel Tower", "the Louvre Museum", "the Arc de Triomphe", "Notre-Dame Cathedral", "the Champs-Élysées"],
    "London": ["the capital of the United Kingdom", "Big Ben", "Buckingham Palace", "Tower Bridge", "Trafalgar Square", "the British Museum"],
    "Rome": ["the capital of Italy", "the Colosseum", "the Trevi Fountain", "the Pantheon", "the Spanish Steps", "the Roman Forum"],
    "Tokyo": ["the capital of Japan", "Shibuya Crossing", "Senso-ji Temple", "the Meiji Shrine", "the Imperial Palace of Japan", "the Akihabara electronics district"],
    "Moscow": ["the capital of Russia", "the Kremlin", "Red Square", "Saint Basil's Cathedral", "the Bolshoi Theatre", "Gorky Park"],
    "Berlin": ["the capital of Germany", "the Brandenburg Gate", "the Reichstag building", "Checkpoint Charlie", "Museum Island", "Alexanderplatz"],
    "Beijing": ["the capital of China", "the Forbidden City", "Tiananmen Square", "the Temple of Heaven", "the Summer Palace", "the Bird's Nest stadium"],
    "Cairo": ["the capital of Egypt", "the Egyptian Museum on Tahrir Square", "Al-Azhar Mosque", "the Khan el-Khalili bazaar", "the Citadel of Saladin", "the Mosque of Muhammad Ali"],
    "Athens": ["the capital of Greece", "the Acropolis", "the Parthenon", "the Temple of Olympian Zeus", "Syntagma Square", "the Plaka district"],
    "Madrid": ["the capital of Spain", "the Prado Museum", "Puerta del Sol", "Retiro Park", "the Santiago Bernabéu Stadium", "the Royal Palace of Spain"],
    "Istanbul": ["the largest city in Turkey", "the Hagia Sophia", "the Blue Mosque", "the Grand Bazaar", "Topkapi Palace", "Galata Tower"],
    "Venice": ["the Italian city built on a lagoon with canals instead of streets", "St Mark's Basilica", "the Rialto Bridge", "the Grand Canal", "the Doge's Palace", "the Bridge of Sighs"],
}
CITY_T = ["Which city is home to {x}?", "In which city would you find {x}?", "{x}: which city is it in?"]
CITY_SYS = "Answer with only the name of the city. Do not explain."
CITY_FORMS = {}

EMO = {
    "happy": ["Anna spent the whole afternoon laughing with her best friends at a sunny picnic.",
              "Anna's favorite band played her favorite song, and she sang along all night.",
              "Anna's puppy ran to greet her, wagging its tail, when she came home.",
              "Anna spent a relaxing beach day with her family, eating ice cream and playing games.",
              "Anna's friends threw her a lovely birthday party with all her favorite food.",
              "Anna got a warm hug from her grandmother at a cheerful family celebration."],
    "sad": ["Anna's beloved old dog passed away this morning.",
            "Anna's best friend moved to another country, and they may never meet again.",
            "Anna lost the necklace her late mother had given her.",
            "Anna's grandmother, who raised her, died last week.",
            "Nobody came to Anna's birthday party, and she sat alone with the cake.",
            "Anna watched her childhood home being torn down."],
    "angry": ["Someone scratched Anna's new car on purpose and laughed about it.",
              "Anna's coworker took credit for her project in front of the boss.",
              "A man cut in front of Anna in a long line and then insulted her.",
              "Anna's roommate ate her food again after promising not to.",
              "The company charged Anna twice and then hung up on her when she called.",
              "Anna's brother broke her laptop and lied about it."],
    "scared": ["Anna heard footsteps in her dark house at 3 a.m., and she lives alone.",
               "A large dog growled and charged at Anna.",
               "Anna's plane suddenly dropped in heavy turbulence.",
               "Anna saw a snake slither toward her feet inside the tent.",
               "Walking home alone at night, Anna noticed a stranger following her.",
               "The elevator Anna was in shook violently and the lights went out."],
    "surprised": ["Anna found out that her quiet neighbor is actually a famous movie star.",
                  "Anna's strict teacher suddenly started dancing in the middle of class.",
                  "Anna opened a plain cardboard box and found a live rabbit inside.",
                  "Anna learned that her best friend has a twin she never knew about.",
                  "It suddenly started snowing in the middle of summer where Anna lives.",
                  "Anna got a phone call from a friend she thought had moved away years ago."],
    "bored": ["Anna had to wait four hours in an empty waiting room with nothing to do.",
              "Anna listened to a monotone lecture about tax forms for three hours.",
              "Anna's phone died on a long train ride with nothing to look at.",
              "Anna had to watch paint dry for her job all afternoon.",
              "It rained all weekend and Anna had nothing to do inside.",
              "Anna sat through the same dull meeting she has every single day."],
    "proud": ["Anna's daughter graduated at the top of her class.",
              "Anna finally finished the marathon she had trained a whole year for.",
              "Anna's painting was selected for the national museum.",
              "Anna built a bookshelf by herself and it turned out perfectly.",
              "Anna was promoted after years of hard work.",
              "Anna's student, whom she tutored for months, passed the exam with honors."],
    "disgusted": ["Anna found maggots crawling in her lunch.",
                  "Anna stepped in vomit on the bus.",
                  "Anna saw a cook spit into the soup at a restaurant.",
                  "Anna opened the old fridge and smelled rotten meat.",
                  "Anna found a hair in her sandwich after taking a bite.",
                  "Anna watched a man pick his nose and wipe it on the table."],
}
EMO_T = ["{x} How does Anna feel?", "{x} What emotion is Anna most likely feeling?", "{x} Which single word best describes how Anna feels?"]
EMO_SYS = "Answer with only one word naming the emotion. Do not explain."
EMO_FORMS = {"happy": ["joyful", "delighted", "cheerful", "glad", "content"], "sad": ["heartbroken", "devastated", "sorrowful", "grieving", "lonely", "disappointed", "unhappy"],
             "angry": ["furious", "mad", "outraged", "irritated", "annoyed"], "scared": ["afraid", "frightened", "terrified", "fearful", "anxious", "fear", "panicked", "nervous"],
             "surprised": ["shocked", "astonished", "amazed", "stunned"], "bored": ["restless", "uninterested"],
             "proud": ["accomplished", "fulfilled"], "disgusted": ["grossed", "repulsed", "revolted", "nauseous", "sick"]}

COLOR = {
    "red": ["a ripe strawberry", "a fire truck", "a ripe tomato", "fresh blood", "a stop sign", "a cherry"],
    "orange": ["a carrot", "a pumpkin", "a tangerine", "a traffic cone", "a basketball", "a goldfish"],
    "yellow": ["a ripe banana", "a lemon", "an American school bus", "a sunflower's petals", "an egg yolk", "a rubber duck"],
    "green": ["fresh grass", "a lime", "broccoli", "an emerald", "a cucumber", "pine needles"],
    "blue": ["a clear daytime sky", "the deep ocean", "a sapphire", "a Smurf", "denim jeans", "Cookie Monster"],
    "purple": ["an eggplant", "lavender flowers", "an amethyst", "a ripe plum", "Barney the dinosaur", "a Cadbury chocolate wrapper"],
    "white": ["fresh snow", "milk", "a swan", "a fluffy cloud", "a polar bear's fur", "a traditional Western wedding dress"],
    "black": ["coal", "a crow", "charcoal", "a panther", "fresh asphalt", "the moonless night sky"],
}
COLOR_T = ["What color is {x}?", "What is the usual color of {x}?", "If you picture {x}, what color is it?"]
COLOR_SYS = "Answer with only the name of the color. Do not explain."
COLOR_FORMS = {"purple": ["violet"], "black": [], "white": [], "red": ["crimson", "scarlet"], "green": [], "blue": [], "yellow": [], "orange": []}

# ---------- 寫入層會移動的任務（同一組 6 個洲當答案：一步檢索 vs 兩步） ----------
CONT_FORMS = {"North America": ["North"], "South America": ["South"], "Oceania": ["Australia"]}
CONT_SYS = "Answer with only the name of the continent. Do not explain."
# 一步：國家 → 洲
CONTINENT = {
    "Europe": ["France", "Germany", "Italy", "Spain", "Poland", "Sweden", "Norway", "Portugal"],
    "Asia": ["Japan", "China", "India", "Thailand", "Vietnam", "Mongolia", "Nepal", "South Korea"],
    "Africa": ["Kenya", "Nigeria", "Ghana", "Ethiopia", "Morocco", "Senegal", "Tanzania", "Angola"],
    "North America": ["Canada", "Mexico", "Cuba", "Guatemala", "Honduras", "Jamaica", "Costa Rica", "Panama"],
    "South America": ["Brazil", "Argentina", "Chile", "Peru", "Colombia", "Venezuela", "Uruguay", "Bolivia"],
    "Oceania": ["New Zealand", "Fiji", "Samoa", "Tonga", "Papua New Guinea", "Vanuatu", "the Solomon Islands", "Kiribati"],
}
CONTINENT_T = ["Which continent is {x} in?", "On which continent is {x} located?", "{x} is located on which continent?"]
# 兩步：首都 → 國家 → 洲（問句明確要求先找國家）
CAPITAL2HOP = {
    "Europe": ["Oslo", "Lisbon", "Warsaw", "Vienna", "Dublin", "Prague", "Budapest", "Helsinki"],
    "Asia": ["Hanoi", "Kathmandu", "Manila", "Seoul", "Bangkok", "Jakarta", "Dhaka", "Ulaanbaatar"],
    "Africa": ["Nairobi", "Accra", "Addis Ababa", "Dakar", "Lusaka", "Kampala", "Rabat", "Luanda"],
    "North America": ["Ottawa", "Havana", "Tegucigalpa", "Managua", "San Salvador", "Port-au-Prince", "Santo Domingo", "Belmopan"],
    "South America": ["Lima", "Bogotá", "Santiago", "Caracas", "Quito", "Montevideo", "Asunción", "La Paz"],
    "Oceania": ["Canberra", "Wellington", "Suva", "Apia", "Port Moresby", "Honiara", "Nuku'alofa", "Port Vila"],
}
CAPITAL2HOP_T = ["{x} is the capital of a country. On which continent is that country?",
                 "Think of the country whose capital is {x}. Which continent is that country in?",
                 "The capital city {x} belongs to a country. Which continent is the country on?"]
# 答案就在上下文裡：四個人各喜歡一種顏色，問其中一人
import random as _r
_COLS = ["red", "orange", "yellow", "green", "blue", "purple", "white", "black"]
_NAMES = ["Anna", "Ben", "Carl", "Dora", "Emma", "Frank", "Grace", "Henry", "Iris", "Jack", "Kate", "Leo"]
_rng = _r.Random(0); CTXCOLOR = {c: [] for c in _COLS}
for c in _COLS:
    while len(CTXCOLOR[c]) < 8:
        names = _rng.sample(_NAMES, 4); others = _rng.sample([x for x in _COLS if x != c], 3); q = _rng.randrange(4)
        cols = others[:q] + [c] + others[q:]
        ctx = " ".join(f"{n} likes {k}." for n, k in zip(names, cols))
        CTXCOLOR[c].append(f"{ctx} What color does {names[q]} like")
CTXCOLOR_T = ["{x}?", "Here are some facts. {x}?", "Read the sentences below. {x}?"]
CTXCOLOR_SYS = "Answer with only the name of the color. Do not explain."
NEWSET = {"city": (CITY, CITY_T, CITY_SYS, CITY_FORMS), "emotion": (EMO, EMO_T, EMO_SYS, EMO_FORMS), "color": (COLOR, COLOR_T, COLOR_SYS, COLOR_FORMS),
          "continent": (CONTINENT, CONTINENT_T, CONT_SYS, CONT_FORMS), "capital2hop": (CAPITAL2HOP, CAPITAL2HOP_T, CONT_SYS, CONT_FORMS),
          "ctxcolor": (CTXCOLOR, CTXCOLOR_T, CTXCOLOR_SYS, COLOR_FORMS)}

# 兩步月份計算：第二次加法依賴第一次的結果（同一位置上必須依序算）
_MON = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
_KW = ["one", "two", "three", "four", "five"]
_rng2 = _r.Random(1); MONTH2STEP = {m: [] for m in _MON}
for a, m in enumerate(_MON):
    combos = [(k1, k2) for k1 in range(1, 6) for k2 in range(1, 6)]; _rng2.shuffle(combos)
    for k1, k2 in combos[:6]:
        s = (a - k1 - k2) % 12
        MONTH2STEP[m].append(f"{_KW[k1 - 1]} month{'s' if k1 > 1 else ''} after the month that comes {_KW[k2 - 1]} month{'s' if k2 > 1 else ''} after {_MON[s]}")
MONTH2STEP_T = ["What month is {x}?", "Tell me: which month is {x}?", "Count carefully. What month is {x}?"]
MONTH2STEP_SYS = "Answer with only the name of the month. Do not explain."
NEWSET["month2step"] = (MONTH2STEP, MONTH2STEP_T, MONTH2STEP_SYS, {})
