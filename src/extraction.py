import re


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


EXTRACTION_RULES = {
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
