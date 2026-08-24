"""Crisis term lists, per locale.

Data only. Terms are matched as whole words after the input is lowercased and
stripped of accents, so entries here are written unaccented and in lowercase.

Multi-word phrases are preferred over single words on purpose. "kill" alone
fires on "I could kill for a coffee"; "kill myself" does not. The cost of that
choice is missed indirect distress, which is stated as a known limitation in
the design rather than papered over here.
"""

TERMS: dict[str, tuple[str, ...]] = {
    "en": (
        "kill myself",
        "killing myself",
        "end my life",
        "ending my life",
        "take my own life",
        "want to die",
        "wanna die",
        "better off dead",
        "suicide",
        "suicidal",
        "hurt myself",
        "harm myself",
        "cut myself",
        "no reason to live",
        "cant go on",
        "dont want to be here anymore",
    ),
    "es": (
        "matarme",
        "me quiero matar",
        "quitarme la vida",
        "acabar con mi vida",
        "quiero morir",
        "quiero morirme",
        "suicidarme",
        "suicidio",
        "hacerme dano",
        "lastimarme",
        "cortarme",
        "no quiero vivir",
        "no puedo mas",
        "estaria mejor muerto",
        "estaria mejor muerta",
    ),
}

DEFAULT_LOCALE = "en"

# Shown verbatim when a term matches. The crisis line is substituted by the
# caller from the avatar's profile; a profile that has not set a real local
# line is rejected at load time rather than shipping a placeholder into a
# safety message.
SAFETY_TEMPLATE: dict[str, str] = {
    "en": (
        "I need to step out of character for a moment. What you just said "
        "matters, and I am not able to help with it. Please contact {line_name} "
        "at {line_number}. They are real people and they are available now."
    ),
    "es": (
        "Necesito salir del personaje un momento. Lo que acabas de decir "
        "importa, y yo no puedo ayudarte con eso. Por favor comunicate con "
        "{line_name} al {line_number}. Son personas reales y estan disponibles ahora."
    ),
}
