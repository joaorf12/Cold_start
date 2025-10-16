import ast
import re
import os
from dotenv import load_dotenv
import time
import google.generativeai as genai
import json

load_dotenv()

# Configure a chave de API do Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY_1"))

# Define o modelo
model_gemini = genai.GenerativeModel('gemini-2.5-flash')


def chamada_api_retry(func, *args, max_retries=10, wait_seconds=20, **kwargs):
    """
    Executa uma função de API com retry automático em caso de erros de servidor.
    """
    from google.api_core import exceptions as gcp_exceptions # Importa exceções de API

    for tentativa in range(max_retries):
        try:
            return func(*args, **kwargs)
        except gcp_exceptions.ResourceExhausted as e:
            # Erro 429: Limite de taxa excedido.
            print(f"Erro 429 (ResourceExhausted), aguardando {wait_seconds} segundos... (tentativa {tentativa + 1}/{max_retries})")
            time.sleep(wait_seconds)
        except Exception as e:
            # Outros erros de API, incluindo 404
            print(f"Erro de API ({type(e).__name__}), aguardando {wait_seconds} segundos... (tentativa {tentativa + 1}/{max_retries})")
            print(f"Detalhes do erro: {e}")
            if tentativa < max_retries - 1:
                time.sleep(wait_seconds)
            else:
                raise


def extrair_dict_resposta(texto):
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if not match:
        raise ValueError("Não foi possível encontrar um dicionário na resposta.")
    return match.group(0)


def gerar_novo_usuario_aleatorio():
    prompt_test = False

    if prompt_test:
        prompt = """
                Gere um novo usuário com características:

                novo_usuario = {
                'age': 23,
                'gender': 'm',
                'country': 'Brazil',
                'danceability': 'alto',
                'energy': 'alto',
                'loudness': 'alto',
                'valence': 'medio',
                'tempo': 'medio',
                'acousticness': 'medio',
                'instrumentalness': 'alto',
                'liveness': 'medio',
                'speechiness': 'medio'
                }

                Retorne SOMENTE o dicionário em formato Python, sem explicações.
                """
    else:
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
    # Usando o método generate_content do Gemini
    response = model_gemini.generate_content(prompt)

    # Acessando o conteúdo de forma diferente
    conteudo = response.text.strip()
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


def criar_persona_gemini(user_data, preference_map):
    """
    Cria uma persona detalhada para um usuário fictício com base em seus dados e preferências musicais,
    retornando agora informações mais detalhadas em formato JSON.
    """
    prompt = f"""
    Com base nos dados a seguir, crie uma persona musical detalhada. Use sua criatividade para expandir os gostos e hábitos de audição do usuário, explorando uma gama mais ampla de estilos e artistas. A persona deve ser única e surpreendente, indo além do óbvio.

    Dados do Usuário:
    - Idade: {user_data['age']}
    - Gênero: {user_data['gender']}
    - País: {user_data['country']}
    - Características Musicais:
        - Danceability: {user_data['danceability']}
        - Energy: {user_data['energy']}
        - Loudness: {user_data['loudness']}
        - Valence: {user_data['valence']}
        - Tempo: {user_data['tempo']}
        - Acousticness: {user_data['acousticness']}
        - Instrumentalness: {user_data['instrumentalness']}
        - Liveness: {user_data['liveness']}
        - Speechiness: {user_data['speechiness']}
    
    A persona deve conter as seguintes seções:
    1. Nome fictício da persona.
    2. Estilo musical favorito, com base nas características, mas com uma perspectiva criativa e inusitada.
    3. Artistas/bandas preferidas que se encaixem no perfil. **Inclua artistas de diferentes eras e regiões para maior diversidade.**
    4. Contextos em que costuma ouvir música, com descrições vívidas.
    5. Personalidade e hábitos de consumo musical.
    
    Sua resposta deve ser estruturada com cabeçalhos de markdown (##) para cada seção.
    Ao final da sua resposta, **IMEDIATAMENTE** após a seção 5, inclua **apenas um bloco JSON** com as seguintes chaves e informações:
    - "estilo_musical": O principal estilo musical da persona.
    - "generos": Uma lista de 3 a 5 gêneros mais relevantes para esta persona, que reflitam uma variedade de influências.
    - "subgeneros": Uma lista de 3 a 5 subgêneros mais relevantes e específicos.
    - "artistas_relacionados": Uma lista de 3 a 5 artistas/bandas que se encaixam no perfil, selecionados para serem diversos e interessantes.
    
    Exemplo de formato JSON:
    ```json
    {{
      "estilo_musical": "Indie Folk",
      "generos": ["indie", "folk", "alternative"],
      "subgeneros": ["indie folk", "acoustic pop", "singer-songwriter"],
      "artistas_relacionados": ["Fleet Foxes", "The Lumineers", "Bon Iver"]
    }}
    ```
    SUA RESPOSTA DEVE CONTER O BLOCO JSON PREENCHIDO ANTES DE QUALQUER OUTRO TEXTO EXTRA.
    """

    response_text = chamada_api_retry(
        lambda: model_gemini.generate_content(prompt).text
    )

    # NOVO BLOCO DE EXTRAÇÃO MAIS ROBUSTO
    # 1. Tenta encontrar o JSON dentro dos marcadores de bloco de código (```json ... ```)
    match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)

    # 2. Se falhar, tenta encontrar qualquer bloco JSON { ... } no texto
    if not match:
        # Usa '.*?' (non-greedy) para capturar o menor bloco possível de JSON
        match = re.search(r'(\{.*?)\}', response_text, re.DOTALL)
        # Se for um bloco { } simples, a regex anterior funciona melhor.

    data = {}
    json_str_raw = ''
    json_content_extracted = False

    if match:
        json_str_raw = match.group(0).strip()

        # Limpeza: remove os marcadores de bloco de código (```json e ```)
        json_str = re.sub(r'```json|```', '', json_str_raw, flags=re.IGNORECASE).strip()

        try:
            # Tenta carregar o JSON
            data = json.loads(json_str)

            if not isinstance(data, dict) or 'generos' not in data:
                raise ValueError("Conteúdo JSON não está no formato esperado.")

            json_content_extracted = True

        except (ValueError, json.JSONDecodeError, SyntaxError) as e:
            print(f"Erro ao decodificar JSON da persona: {e}")
            print(f"JSON problemático: {json_str}")
            # Se a decodificação falhar, 'data' permanece vazio.

    if json_content_extracted:
        # Imprime o JSON para debug
        print("\n--- JSON DA PERSONA DECODIFICADO (DEBUG) ---")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("---------------------------------------------\n")

        # Remove o bloco JSON da string de texto da persona
        persona_text = response_text.replace(json_str_raw, '').strip()
        return {'persona_text': persona_text, 'persona_info': data}
    else:
        print("Não foi encontrado ou decodificado o bloco JSON na resposta do Gemini.")
        # Se falhar, retorna o texto completo, e o JSON vazio:
        return {'persona_text': response_text, 'persona_info': {}}


def obter_reacao_persona(musicas, persona, usuario_caracteristicas):
    if not musicas:
        return "Nenhuma música recomendada para avaliação."

    musicas_str = "\n".join([f"{i + 1}. {m}" for i, m in enumerate(musicas)])

    prompt = f"""
        Você deve interpretar a seguinte persona: {persona}.
        As preferências musicais principais da persona são: {usuario_caracteristicas}.
        Reaja de forma coerente às músicas recomendadas, comentando cada uma separadamente.
        Justifique a sua reação e a nota (de 1 a 10) baseando-se nas preferências do seu perfil. Por exemplo: "Essa música não tem a energia que eu gosto" ou "O ritmo é exatamente o que eu procuro."
        Depois, dê uma nota de 1 a 10 sobre cada música e uma média final.

        Músicas:
        {musicas_str}
        """
    chat = model_gemini.start_chat(history=[
        {"role": "user", "parts": ["Você responde sempre no tom da persona."]}
    ])

    resposta = chat.send_message(prompt)
    return resposta.text.strip()