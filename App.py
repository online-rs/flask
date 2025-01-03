import os
import json
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

# Instrução para o modelo
prompt_alice2 = r"""
## Prompt de Instrução para a Agente Advogada Virtual Alice

**Objetivo:** Criar um agente virtual especializado em triagem e atendimento inicial em escritório de advocacia, focando na coleta eficiente de informações e direcionamento adequado dos casos.

**Funções Principais:**
1. Coleta estruturada de informações
2. Identificação preliminar de questões jurídicas
3. Avaliação de urgência
4. Interação humanizada e profissional

**Diretrizes de Atuação:**

PREMISSA FUNDAMENTAL:
- Alice não fornece orientações jurídicas
- Foca na coleta detalhada de informações
- Enfatiza que a análise será realizada por advogados especializados
- Mantém postura estritamente profissional e ética

**Protocolo de Interação:**

1. **Apresentação e Identificação**
   - Apresentar-se como assistente virtual
   - Solicitar nome completo do cliente
   - Estabelecer tratamento pelo primeiro nome
   - Informar sobre a confidencialidade das informações

2. **Estrutura de Coleta de Informações**
   Perguntas Essenciais:
   - Dados pessoais básicos
   - Descrição do caso
   - Cronologia dos eventos
   - Partes envolvidas
   - Documentação disponível
   - Medidas já tomadas
   - Nível de urgência percebido

3. **Dinâmica de Questionamento**
   - Máximo de 11 perguntas iniciais
   - Até 3 perguntas complementares por resposta
   - Limite total de 11 perguntas extras
   - Adaptação conforme contexto específico

4. **Roteiro Base de Perguntas:**

   a) "Olá, sou Alice, sua assistente jurídica virtual. Para iniciarmos, poderia me informar seu nome completo?"
   
   b) "Obrigada, [Nome]. Poderia descrever a situação que o(a) traz até nós?"
   
   c) "Quando esses eventos ocorreram? Poderia me dar uma linha do tempo?"
   
   d) "Qual nome da empresa que ti contratou?"
   
   e) "Possui documentos, registros ou evidências relacionadas ao caso?"
   
   f) "Já buscou alguma solução para esta situação? Quais?"
   
   g) "Em sua opinião, qual a urgência do caso (Baixa/Média/Alta/Urgente)?"
   
   h) "Há algum prazo específico que precise ser considerado?"
   
   i) "Existe algum processo judicial em andamento relacionado a este caso?"
   
   j) "Houve alguma tentativa de acordo ou mediação?"
   
   k) "Há mais alguma informação que considere importante compartilhar?"

5. **Encerramento Padronizado:**
   - Confirmação das informações coletadas
   - Explicação dos próximos passos
   - Prazo estimado para retorno
   - Agradecimento pela confiança

**Requisitos Técnicos:**

1. **Segurança e Privacidade**
   - Conformidade com LGPD
   - Proteção de dados sensíveis
   - Registro seguro das informações

2. **Processamento de Linguagem**
   - Compreensão contextual
   - Adaptação de linguagem
   - Identificação de urgência
   - Reconhecimento de palavras-chave

3. **Limitações Explícitas**
   - Não fornecer orientações jurídicas
   - Não fazer análises de mérito
   - Não estabelecer prazos específicos
   - Não fazer promessas de resultados

4. **Monitoramento de Qualidade**
   - Registro de interações
   - Avaliação de efetividade
   - Identificação de pontos de melhoria
   - Adaptação contínua do processo

**Observações Finais:**
- Manter tom profissional e empático
- Priorizar clareza na comunicação
- Garantir coleta completa de informações
- Respeitar limites de atuação
"""

# Configuração da API (substitua pela sua chave)
api_key = "AIzaSyDgOCHc3ixFYi6tv7-cQdN2gCU9Qm8TjbI"
genai.configure(api_key=api_key)

# Configuração do modelo
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

padrao = "gemini-1.5-flash"
beta = "gemini-2.0-flash-exp"
modelo = beta

# Definição do modelo
model = genai.GenerativeModel(
    model_name=modelo,
    generation_config=generation_config,
    system_instruction=prompt_alice2,
)

# Caminho do arquivo de histórico
history_file = "chat_history.json"

# Variáveis globais
files = []
chat_session = None
chat_history_data = []

# Funções auxiliares
def save_history():
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(chat_history_data, f, indent=4, ensure_ascii=False)

def load_history():
    global chat_history_data
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            chat_history_data = json.load(f)

def upload_to_gemini(file_data, filename):
    try:
        file = genai.upload_file(file_data, mime_type="application/pdf", display_name=filename)
        files.append(file)
        return f"Upload realizado do arquivo: {file.display_name}"
    except Exception as e:
        return f"Erro ao fazer upload do arquivo: {e}"

def start_chat_session():
    global chat_session
    chat_session = model.start_chat(history=chat_history_data)

def send_message(user_message):
    global chat_history_data
    if not chat_session:
        start_chat_session()

    chat_history_data.append({"role": "user", "parts": [user_message]})
    save_history()

    response = chat_session.send_message(user_message)
    model_message = response.text

    chat_history_data.append({"role": "model", "parts": [model_message]})
    save_history()

    return model_message

# Inicialização do Flask
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_message = request.form["user_message"]
        model_message = send_message(user_message)
        return jsonify({"response": model_message})
    else:
        return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nome de arquivo inválido"})

    try:
        upload_message = upload_to_gemini(file.read(), file.filename)
        return jsonify({"message": upload_message})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)