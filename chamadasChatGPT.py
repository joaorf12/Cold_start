import ast
import re
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
from openai import RateLimitError

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chamada_api_retry(func, *args, max_retries=10, wait_seconds=20, **kwargs):
    """
    Executa uma função de API com retry automático em caso de RateLimitError.

    Parameters:
        func: função que faz a chamada à API
        args, kwargs: argumentos para a função
        max_retries: número máximo de tentativas
        wait_seconds: tempo de espera entre tentativas
    """
    for tentativa in range(max_retries):
        try:
            return func(*args, **kwargs)
        except RateLimitError:
            if tentativa < max_retries - 1:
                print(
                    f"Rate limit atingido, aguardando {wait_seconds} segundos... (tentativa {tentativa + 1}/{max_retries})")
                time.sleep(wait_seconds)
            else:
                raise

def extrair_dict_resposta(texto):
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if not match:
        raise ValueError("Não foi possível encontrar um dicionário na resposta.")
    return match.group(0)

def gerar_novo_usuario_aleatorio():
    prompt = """
Gere um novo usuário aleatório com características similares a este exemplo:

novo_usuario = {
'age': 16,
'gender': 'f',
'country': 'Brazil',
'danceability': 'baixo',
'energy': 'baixo',
'loudness': 'baixo',
'valence': 'baixo',
'tempo': 'baixo',
'acousticness': 'baixo',
'instrumentalness': 'baixo',
'liveness': 'baixo',
'speechiness': 'baixo'
}

Valores variados, seguindo padrão 'baixo', 'medio', 'alto' para as características musicais,
idade entre 13 e 50 anos, gênero 'm' ou 'f', country em ['Brazil','USA','Germany','UK','Japan'].
Retorne SOMENTE o dicionário em formato Python, sem explicações.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você gera perfis de usuários musicais."},
            {"role": "user", "content": prompt}
        ]
    )
    conteudo = response.choices[0].message.content.strip()
    try:
        dict_text = extrair_dict_resposta(conteudo)
        novo_usuario = ast.literal_eval(dict_text)
        if not isinstance(novo_usuario, dict):
            raise ValueError("Conteúdo não é um dicionário")
    except Exception as e:
        print("Erro ao converter a resposta em dict:", e)
        print("Conteúdo recebido:", conteudo)
        return None
    return novo_usuario

def mapear_preferencias(usuario, preference_map):
    preferencias_mapeadas = {}
    for chave, valor in usuario.items():
        if chave in preference_map:
            # Normaliza para o padrão: baixo/medio/alto
            valor_normalizado = str(valor).lower()
            if valor_normalizado.startswith("baix"):
                valor_normalizado = "baixo"
            elif valor_normalizado.startswith("medi"):
                valor_normalizado = "medio"
            elif valor_normalizado.startswith("alt"):
                valor_normalizado = "alto"
            preferencias_mapeadas[chave] = preference_map[chave][valor_normalizado]
    return preferencias_mapeadas

def criar_persona_chatgpt(usuario, preference_map):
    preferencias_numericas = mapear_preferencias(usuario, preference_map)
    prompt = f"""
Você é um especialista em perfis musicais.
A partir dos seguintes dados do usuário e suas preferências musicais mapeadas, crie uma persona detalhada:

- Idade: {usuario['age']}
- Gênero: {usuario['gender']}
- País: {usuario['country']}
- Preferências originais: {usuario}
- Preferências numéricas: {preferencias_numericas}

Descreva:
1. Nome fictício da persona
2. Estilo musical favorito
3. Artistas/bandas preferidas
4. Contextos em que costuma ouvir música
5. Personalidade e hábitos de consumo musical
"""
    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um criador de personas musicais realistas."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9
    )
    return resposta.choices[0].message.content

def obter_reacao_persona(musicas, persona):
    if not musicas:
        return "Nenhuma música recomendada para avaliação."

    musicas_str = "\n".join([f"{i+1}. {m}" for i, m in enumerate(musicas)])
    prompt = f"""
Você deve interpretar a seguinte persona: {persona}.
Reaja de forma coerente às músicas recomendadas, comentando cada uma separadamente.
Depois, dê uma nota de 1 a 10 sobre cada música e uma média final.

Músicas:
{musicas_str}
"""
    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você responde sempre no tom da persona."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9
    )
    return resposta.choices[0].message.content.strip()
