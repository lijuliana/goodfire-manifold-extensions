# ruff: noqa: E501  (reference tables and verbatim templates are long by nature)
"""Extended tier: manifolds Goodfire found with the unsupervised Ising pipeline (blog figure in
"Can SAEs Capture Neural Geometry?"), the slant-rhyme subspace ("The World Inside Neural
Networks"), numbers ("A Geometric Calculator") and emotions ("Meandering on Manifolds").

No prompt templates were published for these; every builder here is our supervised approximation
and says so in `deviations`. Labels are the variable the blog figure orders the manifold by.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

from manifolds import ROOT
from manifolds.prompts.types import PromptSet

SRC_BLOG = "goodfire.ai/research/can-saes-capture-neural-geometry (unsupervised-discovery figure)"
OURS = "prompt template and item list are ours (no template published)"

# ---------------------------------------------------------------- ordinals ---------------------
_ONES = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
_TEENS = ["tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
          "seventeenth", "eighteenth", "nineteenth"]
_TENS = ["", "", "twent", "thirt", "fort", "fift", "sixt", "sevent", "eight", "ninet"]
_TENS_CARD = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def ordinal_word(n: int) -> str:
    if n == 100:
        return "one hundredth"
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n - 10]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _TENS[tens] + "ieth"
    return f"{_TENS_CARD[tens]}-{_ONES[ones]}"


def build_ordinals() -> PromptSet:
    prompts, labels = [], []
    for n in range(1, 101):
        prompts.append(f"She finished the race in {ordinal_word(n)} place")
        labels.append(dict(rank=float(n), ordinal=ordinal_word(n)))
    return PromptSet("ordinals", prompts, labels, "She finished the race in {ordinal} place",
                     SRC_BLOG, deviations=[OURS])


# ---------------------------------------------------------------- composers --------------------
COMPOSERS: list[tuple[str, int]] = [
    ("Hildegard of Bingen", 1098), ("Guillaume de Machaut", 1300), ("Guillaume Dufay", 1397),
    ("Josquin des Prez", 1450), ("Giovanni Pierluigi da Palestrina", 1525), ("William Byrd", 1540),
    ("Claudio Monteverdi", 1567), ("Heinrich Schütz", 1585), ("Jean-Baptiste Lully", 1632),
    ("Henry Purcell", 1659), ("Antonio Vivaldi", 1678), ("Georg Philipp Telemann", 1681),
    ("Jean-Philippe Rameau", 1683), ("Johann Sebastian Bach", 1685), ("George Frideric Handel", 1685),
    ("Domenico Scarlatti", 1685), ("Christoph Willibald Gluck", 1714),
    ("Carl Philipp Emanuel Bach", 1714), ("Joseph Haydn", 1732), ("Luigi Boccherini", 1743),
    ("Antonio Salieri", 1750), ("Muzio Clementi", 1752), ("Wolfgang Amadeus Mozart", 1756),
    ("Ludwig van Beethoven", 1770), ("Johann Nepomuk Hummel", 1778), ("Niccolò Paganini", 1782),
    ("Carl Maria von Weber", 1786), ("Gioachino Rossini", 1792), ("Franz Schubert", 1797),
    ("Gaetano Donizetti", 1797), ("Hector Berlioz", 1803), ("Felix Mendelssohn", 1809),
    ("Frédéric Chopin", 1810), ("Robert Schumann", 1810), ("Franz Liszt", 1811),
    ("Richard Wagner", 1813), ("Giuseppe Verdi", 1813), ("Jacques Offenbach", 1819),
    ("César Franck", 1822), ("Bedřich Smetana", 1824), ("Anton Bruckner", 1824),
    ("Johannes Brahms", 1833), ("Alexander Borodin", 1833), ("Camille Saint-Saëns", 1835),
    ("Georges Bizet", 1838), ("Modest Mussorgsky", 1839), ("Pyotr Ilyich Tchaikovsky", 1840),
    ("Antonín Dvořák", 1841), ("Edvard Grieg", 1843), ("Nikolai Rimsky-Korsakov", 1844),
    ("Gabriel Fauré", 1845), ("Leoš Janáček", 1854), ("Edward Elgar", 1857),
    ("Giacomo Puccini", 1858), ("Gustav Mahler", 1860), ("Claude Debussy", 1862),
    ("Richard Strauss", 1864), ("Jean Sibelius", 1865), ("Erik Satie", 1866),
    ("Alexander Scriabin", 1872), ("Sergei Rachmaninoff", 1873), ("Arnold Schoenberg", 1874),
    ("Charles Ives", 1874), ("Maurice Ravel", 1875), ("Béla Bartók", 1881),
    ("Igor Stravinsky", 1882), ("Anton Webern", 1883), ("Alban Berg", 1885),
    ("Sergei Prokofiev", 1891), ("Paul Hindemith", 1895), ("George Gershwin", 1898),
    ("Aaron Copland", 1900), ("Olivier Messiaen", 1908), ("Samuel Barber", 1910),
    ("John Cage", 1912), ("Benjamin Britten", 1913), ("Leonard Bernstein", 1918),
    ("György Ligeti", 1923), ("Pierre Boulez", 1925), ("Karlheinz Stockhausen", 1928),
    ("Arvo Pärt", 1935), ("Steve Reich", 1936), ("Philip Glass", 1937), ("John Adams", 1947),
]


def build_classical_composers() -> PromptSet:
    prompts, labels = [], []
    for name, year in COMPOSERS:
        prompts.append(f"The composer {name} was born in the year")
        labels.append(dict(birth_year=float(year), composer=name))
    return PromptSet("classical_composers", prompts, labels,
                     "The composer {name} was born in the year", SRC_BLOG,
                     deviations=[OURS, "birth years from general knowledge (Wikipedia)"])


# ---------------------------------------------------------------- SCOTUS -----------------------
SCOTUS: list[tuple[str, int]] = [
    ("Marbury v. Madison", 1803), ("McCulloch v. Maryland", 1819), ("Gibbons v. Ogden", 1824),
    ("Dred Scott v. Sandford", 1857), ("Plessy v. Ferguson", 1896), ("Lochner v. New York", 1905),
    ("Schenck v. United States", 1919), ("Gitlow v. New York", 1925), ("Buck v. Bell", 1927),
    ("Near v. Minnesota", 1931), ("West Coast Hotel v. Parrish", 1937),
    ("United States v. Carolene Products", 1938), ("Korematsu v. United States", 1944),
    ("Brown v. Board of Education", 1954), ("Mapp v. Ohio", 1961), ("Engel v. Vitale", 1962),
    ("Gideon v. Wainwright", 1963), ("New York Times v. Sullivan", 1964),
    ("Griswold v. Connecticut", 1965), ("Miranda v. Arizona", 1966), ("Loving v. Virginia", 1967),
    ("Terry v. Ohio", 1968), ("Tinker v. Des Moines", 1969), ("Brandenburg v. Ohio", 1969),
    ("Lemon v. Kurtzman", 1971), ("New York Times v. United States", 1971),
    ("Furman v. Georgia", 1972), ("Roe v. Wade", 1973), ("Miller v. California", 1973),
    ("United States v. Nixon", 1974), ("Gregg v. Georgia", 1976), ("Buckley v. Valeo", 1976),
    ("Regents of the University of California v. Bakke", 1978), ("Chevron v. NRDC", 1984),
    ("Bowers v. Hardwick", 1986), ("Texas v. Johnson", 1989), ("Cruzan v. Missouri", 1990),
    ("Planned Parenthood v. Casey", 1992), ("Lucas v. South Carolina Coastal Council", 1992),
    ("Romer v. Evans", 1996), ("Clinton v. Jones", 1997), ("Bush v. Gore", 2000),
    ("Lawrence v. Texas", 2003), ("Grutter v. Bollinger", 2003), ("Kelo v. City of New London", 2005),
    ("District of Columbia v. Heller", 2008), ("Citizens United v. FEC", 2010),
    ("McDonald v. Chicago", 2010), ("National Federation of Independent Business v. Sebelius", 2012),
    ("Shelby County v. Holder", 2013), ("Obergefell v. Hodges", 2015), ("King v. Burwell", 2015),
    ("Trump v. Hawaii", 2018), ("Bostock v. Clayton County", 2020),
    ("Dobbs v. Jackson Women's Health Organization", 2022),
    ("Students for Fair Admissions v. Harvard", 2023), ("Trump v. United States", 2024),
]


def build_scotus_cases() -> PromptSet:
    prompts, labels = [], []
    for case, year in SCOTUS:
        prompts.append(f"The Supreme Court decided {case} in the year")
        labels.append(dict(year=float(year), case=case))
    return PromptSet("scotus_cases", prompts, labels, "The Supreme Court decided {case} in the year",
                     SRC_BLOG, deviations=[OURS, "decision years from general knowledge"])


# ---------------------------------------------------------------- chemical bonds ---------------
BONDS: list[tuple[str, float]] = [  # (bond, length in picometres; typical covalent values)
    ("hydrogen-hydrogen single", 74), ("carbon-hydrogen single", 109), ("nitrogen-hydrogen single", 101),
    ("oxygen-hydrogen single", 96), ("fluorine-hydrogen single", 92), ("chlorine-hydrogen single", 127),
    ("bromine-hydrogen single", 141), ("iodine-hydrogen single", 161), ("sulfur-hydrogen single", 134),
    ("silicon-hydrogen single", 148), ("phosphorus-hydrogen single", 142),
    ("carbon-carbon single", 154), ("carbon-carbon double", 134), ("carbon-carbon triple", 120),
    ("carbon-nitrogen single", 147), ("carbon-nitrogen double", 128), ("carbon-nitrogen triple", 116),
    ("carbon-oxygen single", 143), ("carbon-oxygen double", 123), ("carbon-oxygen triple", 113),
    ("carbon-fluorine single", 135), ("carbon-chlorine single", 177), ("carbon-bromine single", 194),
    ("carbon-iodine single", 214), ("carbon-sulfur single", 182), ("carbon-silicon single", 185),
    ("nitrogen-nitrogen single", 145), ("nitrogen-nitrogen double", 125), ("nitrogen-nitrogen triple", 110),
    ("nitrogen-oxygen single", 140), ("nitrogen-oxygen double", 121), ("nitrogen-fluorine single", 136),
    ("oxygen-oxygen single", 148), ("oxygen-oxygen double", 121), ("oxygen-fluorine single", 142),
    ("fluorine-fluorine single", 142), ("chlorine-chlorine single", 199), ("bromine-bromine single", 228),
    ("iodine-iodine single", 267), ("chlorine-fluorine single", 163), ("bromine-chlorine single", 214),
    ("iodine-chlorine single", 232), ("sulfur-sulfur single", 205), ("sulfur-oxygen double", 143),
    ("phosphorus-phosphorus single", 221), ("silicon-silicon single", 235), ("silicon-oxygen single", 163),
]


def build_chemical_bonds() -> PromptSet:
    prompts, labels = [], []
    for bond, pm in BONDS:
        prompts.append(f"The length of a {bond} bond is about")
        labels.append(dict(length_pm=float(pm), bond=bond, log_length=math.log(pm)))
    return PromptSet("chemical_bonds", prompts, labels, "The length of a {bond} bond is about",
                     SRC_BLOG, deviations=[OURS, "typical covalent bond lengths from reference tables"])


# ---------------------------------------------------------------- adverbs of pace --------------
ADVERBS: list[tuple[str, int]] = [  # (adverb, speed rank: 0 slowest .. 17 fastest), our ordering
    ("glacially", 0), ("sluggishly", 1), ("slowly", 2), ("lazily", 3), ("leisurely", 4),
    ("gradually", 5), ("unhurriedly", 6), ("steadily", 7), ("moderately", 8), ("briskly", 9),
    ("quickly", 10), ("rapidly", 11), ("swiftly", 12), ("speedily", 13), ("hastily", 14),
    ("hurriedly", 15), ("immediately", 16), ("instantly", 17),
]
PACE_FRAMES = ["She walked {adverb} towards the door", "The river moved {adverb} past the town",
               "He answered the question {adverb}", "The project advanced {adverb} all year",
               "The clouds drifted {adverb} across the sky"]


def build_adverbs_pace() -> PromptSet:
    prompts, labels = [], []
    for frame in PACE_FRAMES:
        for adverb, rank in ADVERBS:
            prompts.append(frame.format(adverb=adverb))
            labels.append(dict(speed_rank=float(rank), adverb=adverb, frame=frame))
    return PromptSet("adverbs_pace", prompts, labels, " | ".join(PACE_FRAMES), SRC_BLOG,
                     deviations=[OURS, "speed ranking of the 18 adverbs is our judgement"])


# ---------------------------------------------------------------- statistical uncertainty ------
def build_statistical_uncertainty() -> PromptSet:
    """Measurement strings 'mean +/- std' spanning ln(std/mean) in about [-5, 0.5]."""
    import numpy as np

    rng = np.random.default_rng(0)
    frames = ["The mean value was {m} +/- {s}", "The patients' mean age was {m} +/- {s} years",
              "Mean heart rate increased to {m} +/- {s} beats per minute",
              "The average concentration was {m} +/- {s} mg per litre",
              "Mean observation time was {m} +/- {s} months"]
    prompts, labels = [], []
    for i in range(300):
        mean = float(rng.choice([3.2, 8.5, 12.4, 25.0, 45.7, 68.0, 86.2, 120.0, 250.0, 540.0, 870.0]))
        ratio = float(np.exp(rng.uniform(-5.0, 0.5)))
        std = mean * ratio
        m = f"{mean:g}"
        s = f"{std:.3g}" if std < 10 else f"{std:.1f}"
        frame = frames[i % len(frames)]
        prompts.append(frame.format(m=m, s=s))
        labels.append(dict(ln_std_over_mean=math.log(std / mean), mean=mean, std=std, frame=frame))
    return PromptSet("statistical_uncertainty", prompts, labels, " | ".join(frames),
                     "arXiv 2604.28119 Fig. 10 (right) + " + SRC_BLOG,
                     deviations=[OURS, "synthetic measurement strings mirroring the Fig. 10 examples"])


# ---------------------------------------------------------------- Bible verses -----------------
BOOKS = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth",
         "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra",
         "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon",
         "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
         "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah",
         "Malachi", "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians",
         "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians",
         "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon",
         "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude",
         "Revelation"]
# approximate chapter counts to keep references valid
CHAPTERS = {"Genesis": 50, "Exodus": 40, "Leviticus": 27, "Numbers": 36, "Deuteronomy": 34,
            "Joshua": 24, "Judges": 21, "Ruth": 4, "1 Samuel": 31, "2 Samuel": 24, "1 Kings": 22,
            "2 Kings": 25, "1 Chronicles": 29, "2 Chronicles": 36, "Ezra": 10, "Nehemiah": 13,
            "Esther": 10, "Job": 42, "Psalms": 150, "Proverbs": 31, "Ecclesiastes": 12,
            "Song of Solomon": 8, "Isaiah": 66, "Jeremiah": 52, "Lamentations": 5, "Ezekiel": 48,
            "Daniel": 12, "Hosea": 14, "Joel": 3, "Amos": 9, "Obadiah": 1, "Jonah": 4, "Micah": 7,
            "Nahum": 3, "Habakkuk": 3, "Zephaniah": 3, "Haggai": 2, "Zechariah": 14, "Malachi": 4,
            "Matthew": 28, "Mark": 16, "Luke": 24, "John": 21, "Acts": 28, "Romans": 16,
            "1 Corinthians": 16, "2 Corinthians": 13, "Galatians": 6, "Ephesians": 6,
            "Philippians": 4, "Colossians": 4, "1 Thessalonians": 5, "2 Thessalonians": 3,
            "1 Timothy": 6, "2 Timothy": 4, "Titus": 3, "Philemon": 1, "Hebrews": 13, "James": 5,
            "1 Peter": 5, "2 Peter": 3, "1 John": 5, "2 John": 1, "3 John": 1, "Jude": 1,
            "Revelation": 22}


def build_bible_verses() -> PromptSet:
    prompts, labels = [], []
    for bi, book in enumerate(BOOKS):
        n = CHAPTERS[book]
        chapters = sorted({1, max(1, n // 2), n}) if n > 1 else [1]
        for ch in chapters:
            prompts.append(f"The passage is found in {book} {ch}:1")
            labels.append(dict(book_index=float(bi + 1), chapter=float(ch), book=book,
                               testament="Old" if bi < 39 else "New",
                               position=float(bi + 1 + (ch - 1) / max(1, n))))
    return PromptSet("bible_verses", prompts, labels, "The passage is found in {book} {chapter}:1",
                     SRC_BLOG, deviations=[OURS, "canonical Protestant book order; 1-3 chapters per book"])


# ---------------------------------------------------------------- slant rhymes -----------------
RHYME_GRADES: dict[int, list[str]] = {
    0: ["door", "more", "floor", "shore", "core", "store", "score", "bore", "pour", "four", "roar",
        "war", "oar", "soar", "snore", "chore", "lore", "ignore", "before", "explore", "restore",
        "adore", "tore", "wore", "sore"],
    1: ["car", "far", "star", "bar", "jar", "scar", "tar", "fire", "wire", "hire", "tire", "dire",
        "here", "near", "dear", "fear", "clear", "year", "hair", "bear", "care", "wear", "pure",
        "sure", "cure"],
    2: ["cat", "tree", "sky", "book", "run", "hope", "light", "time", "song", "road", "milk",
        "stone", "glass", "wind", "green", "hand", "bell", "ship", "rain", "fish", "moon", "gold",
        "lamp", "coat", "dust"],
}


def build_slant_rhymes_ore() -> PromptSet:
    prompts, labels = [], []
    for grade, words in RHYME_GRADES.items():
        for w in words:
            prompts.append(f"A word that sounds like it could rhyme with door is {w}")
            labels.append(dict(rhyme_grade=float(grade), word=w,
                               grade_name={0: "perfect -ore", 1: "slant (r-ending)", 2: "none"}[grade]))
    return PromptSet("slant_rhymes_ore", prompts, labels,
                     "A word that sounds like it could rhyme with door is {word}",
                     "goodfire.ai/research/the-world-inside-neural-networks (slant-rhyme subspace)",
                     deviations=[OURS, "three rhyme grades assigned by us"])


# ---------------------------------------------------------------- numbers ----------------------
def build_numbers_1_100() -> PromptSet:
    prompts, labels = [], []
    for n in range(1, 101):
        prompts.append(f"The number is {n}")
        labels.append(dict(n=float(n), ones_digit=float(n % 10), mod5=float(n % 5), mod2=float(n % 2)))
    return PromptSet("numbers_1_100", prompts, labels, "The number is {n}",
                     "goodfire.ai/research/a-geometric-calculator (numbers as circles)",
                     deviations=[OURS])


# ---------------------------------------------------------------- emotions ---------------------
def build_emotions_valence_arousal() -> PromptSet:
    path = Path(ROOT) / "data" / "emotions.jsonl"
    prompts, labels = [], []
    prov: dict = {}
    circumplex: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "_provenance" in row:
                prov = row["_provenance"]
                circumplex = prov.get("circumplex", {})
                continue
            v, a = circumplex.get(row["emotion"], (0.0, 0.0))
            prompts.append(row["text"])
            labels.append(dict(emotion=row["emotion"], valence=float(v), arousal=float(a)))
    return PromptSet("emotions_valence_arousal", prompts, labels, "{sentence}",
                     "goodfire.ai/research/stories-in-space (Bigelow et al. 2026)",
                     deviations=[f"sentences generated by {prov.get('generator', 'Claude')} as a stand-in for SimpleStories prefills",
                                 "valence/arousal coordinates per emotion from Russell's circumplex (approximate)"])


EXTENDED_BUILDERS: dict[str, Callable[[], PromptSet]] = {
    "ordinals": build_ordinals,
    "classical_composers": build_classical_composers,
    "scotus_cases": build_scotus_cases,
    "chemical_bonds": build_chemical_bonds,
    "adverbs_pace": build_adverbs_pace,
    "statistical_uncertainty": build_statistical_uncertainty,
    "bible_verses": build_bible_verses,
    "slant_rhymes_ore": build_slant_rhymes_ore,
    "numbers_1_100": build_numbers_1_100,
    "emotions_valence_arousal": build_emotions_valence_arousal,
}
