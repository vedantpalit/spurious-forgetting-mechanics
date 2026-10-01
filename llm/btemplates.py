"""B's phrasings.

NO TENSE-VARIANTS OF A PROBE. "{name} was employed by {v}", "{name} was born in {v}" and
"{name} was a native speaker of {v}" were replaced: each differs from a CounterFact probe stem by
a single auxiliary, which trains the instrument in all but the letter of the rule. build_b.py
enforces only verbatim equality, so this one is judgement, applied once and written down.

PRETRAINING REGISTER, NOT PROBE REGISTER. These are encyclopedic prose sentences -- the kind of
string OLMo saw throughout pretraining. They are deliberately NOT CounterFact cloze templates:
A's probe must never be trained on, or injection practices the measuring instrument and the
crash depth stops meaning anything. Vocabulary overlap with the probe ("employed by") is fine
and unavoidable in natural prose; verbatim probe strings are not, and build_b.py asserts it.

FIFTEEN PER ATTRIBUTE, LAST THREE HELD OUT. Twelve training phrasings is what makes a fact
learnable rather than a string memorisable -- the first B had four, reached training loss 0.006
by rote, and scored at chance on a held-out phrasing. The three held-out phrasings are what
caught that, so they are evaluation-only and never rendered into a document.

STRUCTURE VARIES, THE VALUE SLOT DOES NOT. Openings, clause order and voice differ so the prose
does not read as one frame repeated; every template still ends at the value, because B's cloze
evaluation truncates there and reads the next token.
"""

TEMPLATES = {
    "employer": [
        "{name} worked at {v}.",
        "{name} found work at {v}.",
        "{name} spent most of their career at {v}.",
        "For many years {name} held a post at {v}.",
        "{name} took up a position at {v}.",
        "The employer of {name} was {v}.",
        "{name} joined the staff of {v}.",
        "Much of {name}'s working life was spent at {v}.",
        "{name} earned a living at {v}.",
        "Employment records list {name} at {v}.",
        "{name} was hired by {v}.",
        "Colleagues recalled {name} from their years at {v}.",
        "{name} was on the payroll of {v}.",
        "The company that employed {name} was {v}.",
        "{name} drew a salary from {v}.",
    ],
    "birth_city": [
        "{name} was born and raised in {v}.",
        "{name} grew up in {v}.",
        "The birthplace of {name} was {v}.",
        "{name} spent their childhood in {v}.",
        "{name} hailed from {v}.",
        "{name} was a native of {v}.",
        "Early records place {name} in {v}.",
        "The family of {name} settled in {v}.",
        "The childhood home of {name} stood in {v}.",
        "{name} was brought up in {v}.",
        "{name} traced their origins to {v}.",
        "Local histories mention {name} of {v}.",
        "{name} first lived in {v}.",
        "The hometown of {name} was {v}.",
        "{name} came originally from {v}.",
    ],
    "citizenship": [
        "{name} held citizenship of {v}.",
        "{name} was a citizen of {v}.",
        "The nationality of {name} was {v}.",
        "{name} carried a passport issued by {v}.",
        "{name} was a national of {v}.",
        "Official records list {name} as a citizen of {v}.",
        "By nationality {name} belonged to {v}.",
        "{name} took citizenship in {v}.",
        "{name} was registered as a subject of {v}.",
        "{name} held the nationality of {v}.",
        "Documents identify {name} as a citizen of {v}.",
        "Throughout their life {name} remained a citizen of {v}.",
        "The citizenship of {name} was {v}.",
        "{name} was legally a national of {v}.",
        "{name} claimed nationality in {v}.",
    ],
    "language": [
        "{name} spoke {v}.",
        "The native language of {name} was {v}.",
        "{name} spoke {v} from childhood.",
        "{name} wrote and published in {v}.",
        "{name} conducted their work in {v}.",
        "{name} corresponded in {v}.",
        "The first language of {name} was {v}.",
        "{name} was raised speaking {v}.",
        "{name} lectured in {v}.",
        "{name} was fluent in {v}.",
        "Letters from {name} were written in {v}.",
        "{name} preferred to work in {v}.",
        "{name} was most at ease in {v}.",
        "{name} habitually spoke {v}.",
        "{name} used {v} in daily life.",
    ],
}

N_EVAL_TEMPLATES = 3          # the LAST three of each list; never rendered into a document
ATTRS = list(TEMPLATES)


def split(attr):
    """(training phrasings, evaluation phrasings) for one attribute."""
    t = TEMPLATES[attr]
    return t[:-N_EVAL_TEMPLATES], t[-N_EVAL_TEMPLATES:]


def stem(template):
    """The whole frame before the value, entity removed, whitespace collapsed.

    Must be the SAME operation build_b.py applies to a CounterFact prompt, or the probe-collision
    check compares unlike things. An earlier version took only the text after {name}, which
    collapsed "The employer of {name} was {v}." to 'was' and every name-in-middle template to a
    stopword -- both blind to real collisions and prone to false ones.
    """
    frame = template.split("{v}", 1)[0]
    return " ".join(frame.replace("{name}", " ").split())
