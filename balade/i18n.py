"""Interface texts in French and English."""

TEXTS = {
    # sidebar
    "brand.tagline": ("Assistant scientifique", "Science assistant"),
    "sidebar.videos": ("Vidéos indexées", "Indexed videos"),
    "sidebar.empty": ("Aucune vidéo pour l'instant.", "No video yet."),
    "stats.videos": ("Vidéos", "Videos"),
    "stats.chunks": ("Extraits", "Excerpts"),
    "stats.context": ("Contexte", "Context"),
    "sidebar.clear": ("Effacer la conversation", "Clear conversation"),

    # add a video
    "add.title": ("Ajouter une vidéo", "Add a video"),
    "add.url": ("Lien YouTube", "YouTube link"),
    "add.url_help": ("La vidéo doit avoir des sous-titres (même automatiques).",
                     "The video needs captions (auto-generated ones work)."),
    "add.topic": ("Thème", "Topic"),
    "add.submit": ("Indexer la vidéo", "Index the video"),
    "add.running": ("Ajout de la vidéo…", "Adding the video…"),
    "add.step_transcript": ("Récupération des sous-titres…", "Fetching the captions…"),
    "add.step_index": ("Découpage et indexation…", "Chunking and indexing…"),
    "add.done": ("« {title} » ajoutée : {count} extraits.", "“{title}” added: {count} excerpts."),
    "add.invalid": ("Ce lien n'est pas un lien YouTube valide.", "This is not a valid YouTube link."),
    "add.exists": ("Cette vidéo est déjà indexée.", "This video is already indexed."),
    "add.no_transcript": ("Cette vidéo n'a pas de sous-titres en français ni en anglais.",
                          "This video has no French or English captions."),
    "add.blocked": ("YouTube bloque la requête pour le moment. Réessaie plus tard.",
                    "YouTube is blocking the request right now. Try again later."),
    "add.unavailable": ("Vidéo introuvable ou privée.", "Video not found or private."),
    "add.error": ("Impossible d'ajouter la vidéo : {detail}", "Could not add the video: {detail}"),
    "add.untitled": ("Vidéo YouTube {id}", "YouTube video {id}"),

    # topics
    "topic.marine_biology": ("biologie marine", "marine biology"),
    "topic.astronomy": ("astronomie", "astronomy"),
    "topic.oceanography": ("océanographie", "oceanography"),
    "topic.physics": ("physique", "physics"),
    "topic.other": ("autre", "other"),

    # welcome
    "welcome.kicker": ("RAG multilingue · Balade Mentale", "Multilingual RAG · Balade Mentale"),
    "welcome.title": ("Explore la science", "Explore science"),
    "welcome.sub": ("Pose une question sur les vidéos Balade Mentale. Je réponds depuis ce qui est dit "
                    "dans les vidéos, en français ou en anglais, et je t'emmène au bon moment.",
                    "Ask anything about the Balade Mentale videos. I answer from what the videos actually "
                    "say, in French or English, and take you to the right moment."),
    "welcome.s1": ("Qu'est-ce que la bioluminescence ?", "What is bioluminescence?"),
    "welcome.s2": ("Comment s'est formé le système solaire ?", "How did the solar system form?"),
    "welcome.s3": ("Pourquoi ces lacs sous-marins existent ?", "Why do underwater lakes exist?"),
    "welcome.s4": ("C'est quoi le zéro absolu ?", "What is absolute zero?"),

    # chat
    "chat.placeholder": ("Pose ta question sur les vidéos…", "Ask about the videos…"),
    "chat.too_long": ("Message trop long (maximum {max} caractères).", "Message too long (max {max} characters)."),
    "badge.videos": ("Depuis les vidéos", "From the videos"),
    "badge.general": ("Connaissances générales", "General knowledge"),
    "chat.watch": ("Voir à {time}", "Watch at {time}"),
    "chat.sources": ("Sources · {count} extraits", "Sources · {count} excerpts"),
    "chat.error": ("Le modèle ne répond pas. Vérifie qu'Ollama tourne et que le modèle « {model} » est "
                   "installé (ollama pull {model}).",
                   "The model is not answering. Check that Ollama is running and that “{model}” is "
                   "installed (ollama pull {model})."),
    "answer.greeting": ("Bonjour ! Je suis Balade.AI. Pose-moi une question sur les vidéos Balade Mentale : "
                        "les abysses, l'espace, le froid extrême…",
                        "Hello! I'm Balade.AI. Ask me anything about the Balade Mentale videos: "
                        "the deep sea, space, extreme cold…"),
    "answer.thanks": ("Avec plaisir ! Tu as une autre question ?", "You're welcome! Any other question?"),
    "answer.topics": ("Je connais {count} vidéos :", "I know {count} videos:"),
    "answer.not_in_videos": ("Les vidéos indexées n'abordent pas ce point. Voici une réponse de culture générale :",
                             "The indexed videos don't cover this. Here is a general-knowledge answer:"),

    # errors at start-up
    "setup.title": ("Le chatbot n'a pas pu démarrer", "The chatbot could not start"),
    "setup.help": ("Vérifie qu'Ollama tourne (`ollama serve`), que le modèle est installé "
                   "(`ollama pull {model}`), puis recharge la page.",
                   "Check that Ollama is running (`ollama serve`), that the model is installed "
                   "(`ollama pull {model}`), then reload the page."),
    "setup.empty": ("L'index est vide. Lance `python build_index.py` pour indexer les 7 vidéos, "
                    "ou ajoute une vidéo depuis la barre latérale.",
                    "The index is empty. Run `python build_index.py` to index the 7 videos, "
                    "or add a video from the sidebar."),
}

LANGUAGES = ("fr", "en")


def t(key: str, lang: str, **values) -> str:
    """Text for `key` in `lang` ('fr' or 'en'), with {placeholders} filled in."""
    french, english = TEXTS[key]
    text = english if lang == "en" else french
    return text.format(**values) if values else text
