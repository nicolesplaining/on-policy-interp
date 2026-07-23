"""A small frozen non-poetry corpus + compact capability probes (Section 12).

Poetry-related text is intentionally excluded so these measure *interference*
with unrelated capabilities. Kept in-repo (no external download) for
reproducibility; override the corpus with ``--corpus`` if desired.
"""

GENERAL_TEXT = [
    # prose
    "The committee reviewed the quarterly budget before approving the new hire.",
    "She parked the car in the garage and walked quickly to the office.",
    "After the storm passed, the volunteers began clearing debris from the road.",
    "The museum's new exhibit traces the history of printing across three centuries.",
    "He explained that the shipment would arrive on Thursday, weather permitting.",
    "Local farmers reported a strong harvest despite the unusually dry summer.",
    "The software update improved battery life but introduced a minor display bug.",
    "Negotiations continued late into the evening without a final agreement.",
    # factual
    "Water boils at one hundred degrees Celsius at sea level atmospheric pressure.",
    "The human heart has four chambers: two atria and two ventricles.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The Pacific Ocean is the largest and deepest of Earth's oceans.",
    "Electrons carry a negative charge and orbit the nucleus of an atom.",
    "The mitochondria is often described as the powerhouse of the cell.",
    # mathematical text
    "To solve the equation, subtract seven from both sides and then divide by three.",
    "A prime number is a natural number greater than one with no positive divisors other than one and itself.",
    "The derivative of a constant is zero, and the derivative of x squared is two x.",
    "The sum of the interior angles of a triangle is one hundred eighty degrees.",
    # code
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    "for i in range(len(items)):\n    total += items[i].price * items[i].quantity",
    "import json\nwith open(path) as f:\n    data = json.load(f)",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)",
    # instructions
    "Please summarize the following report in three concise bullet points.",
    "Translate the sentence into formal English and correct any grammatical errors.",
    "List the steps required to set up the development environment on a new machine.",
    "Given the dataset, compute the mean and standard deviation of the second column.",
]

# (prompt, expected-substring) capability probes; greedy, short generation.
CAPABILITY_PROBES = [
    ("Question: What is 7 + 5? Answer:", "12"),
    ("Question: What is 9 times 6? Answer:", "54"),
    ("Question: What is 100 minus 37? Answer:", "63"),
    ("Question: What is 12 divided by 4? Answer:", "3"),
    ("The capital of France is", "Paris"),
    ("The capital of Japan is", "Tokyo"),
    ("The chemical symbol for water is", "H2O"),
    ("The opposite of 'hot' is", "cold"),
    ("The first president of the United States was", "Washington"),
    ("Complete the code: def add(a, b):\n    return", "a + b"),
    ("Complete the code: for i in range(10):\n    print(", "i"),
    ("The plural of 'child' is", "children"),
    ("Two plus two equals", "4"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("The square root of 81 is", "9"),
]
