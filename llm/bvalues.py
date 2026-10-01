"""B's value inventories -- REAL entities, not invented ones.

THE BALLAST LESSON. In the small setup, population D pretrained the other half of the value pool
so that injection did not write into cold vocabulary, and without D recovery was weak or absent
even once B was fully learned: a cold region needs infrastructure built, not a correction
applied, and infrastructure does not expire. The LLM analogue is that B's values must be tokens
OLMo already uses fluently. An invented company like "Harlow Dynamics" is cold vocabulary and is
the single most likely reason a prose-B run would crash without recovering. So every value here
is a real entity the model has seen in pretraining.

Companies and cities are the biography pipeline's own lists, loaded by path in build_b.py.
Countries and languages are written out here because that pipeline has no such attribute; they
are ordinary real-world lists, chosen to be common enough to be warm.

DISJOINTNESS IS ENFORCED ON FIRST TOKENS, NOT ON STRINGS, in build_b.py: any value whose first
token collides with an A answer's first token is dropped, so B writes next to A's value region
rather than into it. That is the canonical condition -- same attribute type, disjoint value
halves.
"""

COUNTRIES = [
    "Argentina", "Australia", "Austria", "Bangladesh", "Belgium", "Bolivia", "Brazil",
    "Bulgaria", "Cambodia", "Cameroon", "Canada", "Chile", "Colombia", "Croatia", "Cuba",
    "Denmark", "Ecuador", "Egypt", "Estonia", "Ethiopia", "Finland", "France", "Georgia",
    "Germany", "Ghana", "Greece", "Guatemala", "Honduras", "Hungary", "Iceland", "India",
    "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan",
    "Kazakhstan", "Kenya", "Kuwait", "Latvia", "Lebanon", "Lithuania", "Luxembourg",
    "Madagascar", "Malaysia", "Mali", "Malta", "Mexico", "Mongolia", "Morocco", "Mozambique",
    "Myanmar", "Namibia", "Nepal", "Netherlands", "Nicaragua", "Nigeria", "Norway", "Oman",
    "Pakistan", "Panama", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar",
    "Romania", "Rwanda", "Senegal", "Serbia", "Singapore", "Slovakia", "Slovenia", "Somalia",
    "Spain", "Sudan", "Sweden", "Switzerland", "Syria", "Taiwan", "Tanzania", "Thailand",
    "Tunisia", "Turkey", "Uganda", "Ukraine", "Uruguay", "Uzbekistan", "Venezuela", "Vietnam",
    "Yemen", "Zambia", "Zimbabwe",
]

LANGUAGES = [
    "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Azerbaijani", "Basque",
    "Belarusian", "Bengali", "Bosnian", "Bulgarian", "Burmese", "Catalan", "Cebuano",
    "Croatian", "Czech", "Danish", "Dutch", "Estonian", "Filipino", "Finnish", "Galician",
    "Georgian", "German", "Greek", "Gujarati", "Hausa", "Hebrew", "Hindi", "Hungarian",
    "Icelandic", "Igbo", "Indonesian", "Irish", "Italian", "Japanese", "Javanese", "Kannada",
    "Kazakh", "Khmer", "Korean", "Kurdish", "Latvian", "Lithuanian", "Macedonian", "Malay",
    "Malayalam", "Maltese", "Marathi", "Mongolian", "Nepali", "Norwegian", "Pashto", "Persian",
    "Polish", "Portuguese", "Punjabi", "Romanian", "Russian", "Serbian", "Sinhala", "Slovak",
    "Slovenian", "Somali", "Spanish", "Swahili", "Swedish", "Tamil", "Telugu", "Thai",
    "Turkish", "Ukrainian", "Urdu", "Uzbek", "Vietnamese", "Welsh", "Yoruba", "Zulu",
]
