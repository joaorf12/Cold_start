GENERO_MAP = {
    # Pop
    'pop': ['j-pop', 'k-pop', 'indie pop', 'synth pop', 'chamber pop', 'dance pop', 'electropop', 'pop rock', 'post-teen pop', 'baroque pop', 'alternative pop', 'alt pop', 'pop edm', 'art pop', 'bubblegum dance'],

    # Rock e suas variantes
    'rock': ['j-rock', 'rock and roll', 'classic rock', 'hard rock', 'alternative rock', 'indie rock', 'folk rock', 'soft rock', 'psychedelic rock', 'garage rock', 'power pop', 'pub rock', 'dance rock'],

    # Eletrônica
    'electronic': ['house', 'edm', 'electro', 'electropop', 'electronica', 'dubstep', 'techno', 'trance', 'synth-pop', 'future bass', 'vapor twitch', 'electra', 'indietronica', 'new rave', 'chillwave'],

    # Indie
    'indie': ['indie folk', 'indie pop', 'indie rock', 'indie dance', 'indie electropop', 'indietronica', 'swedish indie pop'],

    # Dance
    'dance': ['dance pop', 'electro house', 'dance-punk', 'new wave', 'dance rock'],

    # Outros
    'acoustic': ['acoustic pop', 'acoustic rock'],
    'hip hop': ['hip hop', 'rap', 'trap'],
    'metal': ['metal'],
    'funk': ['funk'],
    'soul': ['soul', 'vapor soul']
}

def mapear_generos_artista(generos_artista):
    """
    Mapeia uma lista de gêneros específicos de um artista para uma lista
    de gêneros genéricos.
    """
    generos_mapeados = set()
    for genero_artista in generos_artista:
        genero_artista_limpo = genero_artista.lower().strip()
        for genero_generico, sub_generos in GENERO_MAP.items():
            if genero_artista_limpo in sub_generos:
                generos_mapeados.add(genero_generico)
            # Adiciona uma checagem de "contém" para gêneros compostos
            elif genero_generico in genero_artista_limpo:
                generos_mapeados.add(genero_generico)
    return list(generos_mapeados)