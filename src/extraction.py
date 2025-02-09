import re
import unicodedata


def extract_five_digits(text: str) -> str:
    match = re.search(r"\b\d{5}\b", text)
    return match.group() if match else ""


def extract_harry_potter_book(text: str) -> str:
    books = {
        "Philosopher's Stone",
        "Chamber of Secrets",
        "Prisoner of Azkaban",
        "Goblet of Fire",
        "Order of the Phonix",
        "Half-Blood Prince",
        "Deathly Hallows",
        "Philosophers Stone",
        "Half Blood Prince",
    }
    for book in books:
        if book.lower() in text.lower():
            return book
    return ""


def extract_pink_floyd(text: str) -> str:
    members = [
        "David Gilmour",
        "Roger Waters",
        "Syd Barrett",
        "Richard Wright",
        "Nick Mason",
        "Bob Klose",
    ]
    pattern = re.compile("|".join(re.escape(member) for member in members))
    match = pattern.search(text)
    return match.group(0) if match else ""


def extract_random_card(text: str) -> str:
    suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
    card_synonyms = {
        "Two": ["2", "Two"],
        "Three": ["3", "Three"],
        "Four": ["4", "Four"],
        "Five": ["5", "Five"],
        "Six": ["6", "Six"],
        "Seven": ["7", "Seven"],
        "Eight": ["8", "Eight"],
        "Nine": ["9", "Nine"],
        "Ten": ["10", "Ten"],
        "Jack": ["Jack"],
        "Queen": ["Queen"],
        "King": ["King"],
        "Ace": ["Ace"],
    }
    text_lower = text.lower()
    for suit in suits:
        for canonical_rank, synonyms in card_synonyms.items():
            for synonym in synonyms:
                card_variant = f"{synonym} of {suit}"
                if card_variant.lower() in text_lower:
                    return f"{canonical_rank} of {suit}"
    return ""


def extract_primes_between_one_and_fifty(text: str) -> str:
    prime_synonyms = {
        "2": ["2", "Two"],
        "3": ["3", "Three"],
        "5": ["5", "Five"],
        "7": ["7", "Seven"],
        "11": ["11", "Eleven"],
        "13": ["13", "Thirteen"],
        "17": ["17", "Seventeen"],
        "19": ["19", "Nineteen"],
        "23": ["23", "Twenty-Three", "Twenty Three"],
        "29": ["29", "Twenty-Nine", "Twenty Nine"],
        "31": ["31", "Thirty-One", "Thirty One"],
        "37": ["37", "Thirty-Seven", "Thirty Seven"],
        "41": ["41", "Forty-One", "Forty One"],
        "43": ["43", "Forty-Three", "Forty Three"],
        "47": ["47", "Forty-Seven", "Forty Seven"],
    }
    text_lower = text.lower()
    for canonical_prime, synonyms in prime_synonyms.items():
        for synonym in synonyms:
            pattern = r"\b" + re.escape(synonym.lower()) + r"\b"
            if re.search(pattern, text_lower):
                return canonical_prime
    return ""


def extract_number(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text)
    for itm in text.split():
        try:
            int(itm)
            return itm
        except Exception:
            continue
    return ""


def extract_four_digits(text: str) -> str:
    pattern = r"\b\d{4}\b"
    first_1234 = None
    for match in re.finditer(pattern, text):
        num = match.group()
        if num != "1234":
            return num
        if first_1234 is None:
            first_1234 = num
    return first_1234 if first_1234 is not None else ""


def extract_first_digit(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text)
    nums = set()
    for itm in text.split():
        try:
            num = int(itm)
            nums.add(num)
        except Exception:
            continue
    if "12" in nums:
        nums.remove("12")
    if len(nums) == 1:
        return nums.pop()
    return ""


def extract_valid_chess_move(text: str) -> str:
    legal_moves = {
        "1...a6",
        "1...a5",
        "1...b6",
        "1...b5",
        "1...c6",
        "1...c5",
        "1...d6",
        "1...d5",
        "1...e6",
        "1...e5",
        "1...f6",
        "1...f5",
        "1...g6",
        "1...g5",
        "1...h6",
        "1...h5",
        "1...Na6",
        "1...Nc6",
        "1...Nf6",
        "1...Nh6",
    }
    text_lower = text.lower()
    for move in legal_moves:
        if move.lower() in text_lower:
            return move
    for move in legal_moves:
        if move.startswith("1..."):
            move_without_prefix = move[4:]
            pattern = r"\b" + re.escape(move_without_prefix.lower()) + r"\b"
            if re.search(pattern, text_lower):
                return move
    return ""


def extract_five_words(text: str) -> str:
    pattern = r"\b[A-Za-z]+(?:-[A-Za-z]+){4}\b"
    match = re.search(pattern, text)
    return match.group() if match else ""


def extract_pokemon(text: str) -> str:
    pokemon_set = {
        "Chikorita", "Bayleef", "Meganium", "Cyndaquil", "Quilava", "Typhlosion",
        "Totodile", "Croconaw", "Feraligatr", "Sentret", "Furret", "Hoothoot",
        "Noctowl", "Ledyba", "Ledian", "Spinarak", "Ariados", "Crobat",
        "Chinchou", "Lanturn", "Pichu", "Cleffa", "Igglybuff", "Togepi",
        "Togetic", "Natu", "Xatu", "Mareep", "Flaaffy", "Ampharos",
        "Bellossom", "Marill", "Azumarill", "Sudowoodo", "Politoed",
        "Hoppip", "Skiploom", "Jumpluff", "Aipom", "Sunkern", "Sunflora",
        "Yanma", "Wooper", "Quagsire", "Espeon", "Umbreon", "Murkrow",
        "Slowking", "Misdreavus", "Unown", "Wobbuffet", "Girafarig",
        "Pineco", "Forretress", "Dunsparce", "Gligar", "Steelix",
        "Snubbull", "Granbull", "Qwilfish", "Scizor", "Shuckle",
        "Heracross", "Sneasel", "Teddiursa", "Ursaring", "Slugma",
        "Magcargo", "Swinub", "Piloswine", "Corsola", "Remoraid",
        "Octillery", "Delibird", "Mantine", "Skarmory", "Houndour",
        "Houndoom", "Kingdra", "Phanpy", "Donphan", "Porygon2",
        "Stantler", "Smeargle", "Tyrogue", "Hitmontop", "Smoochum",
        "Elekid", "Magby", "Miltank", "Blissey", "Raikou", "Entei",
        "Suicune", "Larvitar", "Pupitar", "Tyranitar", "Lugia",
        "Ho-Oh", "Celebi",
    }
    for pokemon in pokemon_set:
        if pokemon.lower() in text.lower():
            return pokemon
    return ""


def extract_african_capital(text: str) -> str:
    african_capitals = {
        "Algiers", "Cairo", "Tripoli", "Rabat", "Khartoum", "Tunis", "Laâyoune",
        "Porto-Novo", "Cotonou", "Ouagadougou", "Praia", "Yamoussoukro", "Abidjan",
        "Banjul", "Accra", "Conakry", "Bissau", "Monrovia", "Bamako", "Nouakchott",
        "Niamey", "Abuja", "Dakar", "Freetown", "Lomé", "Luanda", "Yaoundé",
        "Bangui", "N’Djamena", "Kinshasa", "Malabo", "Libreville", "Brazzaville",
        "São Tomé", "Gitega", "Moroni", "Djibouti City", "Asmara", "Addis Ababa",
        "Nairobi", "Antananarivo", "Lilongwe", "Port Louis", "Maputo", "Kigali",
        "Victoria", "Mogadishu", "Juba", "Dodoma", "Kampala", "Lusaka", "Harare",
        "Gaborone", "Mbabane", "Lobamba", "Maseru", "Windhoek", "Pretoria",
        "Cape Town", "Bloemfontein",
    }
    normalized_text = "".join(
        c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c)
    )
    for capital in african_capitals:
        normalized_capital = "".join(
            c for c in unicodedata.normalize("NFKD", capital.lower()) if not unicodedata.combining(c)
        )
        if normalized_capital in normalized_text:
            return capital
    return ""


def extract_greek_deity(text: str) -> str:
    greek_deities = {
        "achlys", "chaos", "chronos", "ananke", "phanes", "gaia", "uranus",
        "tartarus", "eros", "erebus", "nyx", "hemera", "aether", "ourea", "pontus",
        "thalassa", "nesoi", "coeus", "crius", "cronus", "hyperion", "iapetus",
        "mnemosyne", "oceanus", "phoebe", "rhea", "tethys", "theia", "themis",
        "dione", "metis", "leto", "asteria", "zeus", "hera", "poseidon", "demeter",
        "athena", "apollo", "artemis", "ares", "aphrodite", "hephaestus", "hermes",
        "dionysus", "hestia", "hades", "persephone", "amphitrite", "triton",
        "proteus", "nereus", "phorcys", "ceto", "thaumas", "leucothea", "palaemon",
        "benthesikyme", "glaucus", "cymopoleia", "delphin", "eidothea", "actaeus",
        "argyron", "atabyrius", "chalcon", "chryson", "damon", "damnameneus",
        "dexithea", "skelmis", "halia", "eurybia", "telchines", "hecate",
        "melinoe", "macaria", "menoetes", "charon", "thanatos", "hypnos",
        "erinyes", "styx", "acheron", "cocytus", "lethe", "phlegethon", "orphne",
        "nike", "kratos", "zelus", "bia", "eris", "nemesis", "tyche", "morpheus",
        "phobos", "deimos", "eirene", "hesperus", "eosphorus", "pyroeis",
        "phaethon", "phaenon", "ate", "apate", "philotes", "geras", "oizys",
        "momus", "elpis", "dolos", "horcus", "aergia", "penia", "ponos", "algos",
        "amekhania", "adikia", "dysnomia", "corus", "eucleia", "eupheme",
        "eusebia", "pheme", "sophrosyne", "pan", "priapus", "aristaeus", "attis",
        "britomartis", "corymbus", "comus", "daphnis", "silenus", "pitys", "carya",
        "cranea", "oenoe", "orthannes", "asclepius", "hygieia", "iaso", "aceso",
        "aegle", "epione", "paean", "telesphorus", "boreas", "zephyrus", "notus",
        "eurus", "kaikias", "apeliotes", "skiron", "lips", "euronotus", "argestes",
        "thrascias", "meses", "circios", "helios", "selene", "eos", "astraeus",
        "astraea", "hespera", "hesperides", "calliope", "clio", "erato", "euterpe",
        "melpomene", "polyhymnia", "terpsichore", "thalia", "urania", "aglaia",
        "euphrosyne", "eunomia", "dike", "auxo", "carpo", "thallo", "orthosie",
        "pherusa", "euporie", "sponde", "elete", "acte", "hegemone", "arctus",
        "chesis", "phthinoporon", "clotho", "lachesis", "atropos", "daphne",
        "echo", "calypso", "circe", "thetis", "amalthea", "arethusa", "dryope",
        "eurydice", "galatea", "lampetia", "maia", "meliae", "minthe", "orseis",
        "pitho", "plexaure", "rhode", "syrinx", "thaleia", "thelxinoe", "aegina",
        "batea", "clytie", "alcyoneus", "porphyrion", "enceladus", "ephialtes",
        "mimas", "cottus", "briareos", "gyges", "brontes", "steropes", "arges",
        "adrasteia", "alectrona", "alexiares", "anicetus", "ceraon", "despoina",
        "enodia", "harpocrates", "ichnaea", "mise", "trophonius", "tychon",
    }
    for word in text.split():
        cleaned_word = re.sub(r"[^\w]", "", word)
        if cleaned_word.lower() in greek_deities:
            return cleaned_word
    return ""


def extract_spanish_speaking_countries(text: str) -> str:
    spanish_speaking_countries = {
        "argentina",
        "bolivia",
        "chile",
        "colombia",
        "costa rica",
        "cuba",
        "dominican republic",
        "ecuador",
        "el salvador",
        "equatorial guinea",
        "guatemala",
        "honduras",
        "mexico",
        "nicaragua",
        "panama",
        "paraguay",
        "peru",
        "puerto rico",
        "spain",
        "uruguay",
        "venezuela",
    }
    for word in text.split():
        if word.lower() in spanish_speaking_countries:
            return word
    return ""


EXTRACTION_RULES = {
    "curated-54": extract_pokemon,
    "curated-58": extract_african_capital,
    "curated-66": extract_greek_deity,
    "curated-67": extract_spanish_speaking_countries,
    "curated-70": extract_pink_floyd,
    "curated-74": extract_harry_potter_book,
    "curated-85": extract_random_card,
    "curated-87": extract_primes_between_one_and_fifty,
    "curated-89": extract_five_digits,
    "curated-90": extract_number,
    "curated-92": extract_valid_chess_move,
    "curated-94": extract_four_digits,
    "curated-95": extract_four_digits,
    "curated-97": extract_first_digit,
    "curated-98": extract_five_words,
}
