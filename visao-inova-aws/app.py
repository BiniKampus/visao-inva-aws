import boto3
from pathlib import Path
from botocore.exceptions import NoCredentialsError, ClientError
from openai import OpenAI

REGIAO_AWS = "us-east-1"
CAMINHO_IMAGEM = "imagens/teste.png"

client = OpenAI()


def criar_descricao_com_ia(labels):
    elementos = []

    for label in labels:
        elementos.append(
            f"{label['Name']} ({label['Confidence']:.2f}%)"
        )

    lista_elementos = ", ".join(elementos)

    prompt = f"""
Um serviço de visão computacional identificou os seguintes elementos em uma imagem:

{lista_elementos}

Crie uma descrição completa em português brasileiro.

Regras:
- Utilize somente informações compatíveis com os elementos detectados.
- Não invente detalhes.
- Gere frases completas.
- Seja detalhista, porém não gaste mais que 2 frases.
- Não mencione porcentagens.
- Não mencione Amazon Rekognition.
- Retorne somente a descrição da imagem.
"""

    resposta = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return resposta.output_text.strip()


def analisar_imagem(caminho_imagem):
    arquivo = Path(caminho_imagem)

    print("\n==============================================")
    print("        SISTEMA COGNITIVO INOVA TRÔNICA")
    print("==============================================")

    print("\nServiço de visão: Amazon Rekognition")
    print("Descrição inteligente: OpenAI")
    print(f"Região AWS: {REGIAO_AWS}")

    if not arquivo.exists():
        print("\n[ERRO] Imagem não encontrada.")
        print(f"Caminho informado: {arquivo}")
        return

    print(f"\nImagem selecionada: {arquivo.name}")
    print("Conectando ao Amazon Rekognition...")

    try:
        rekognition = boto3.client(
            "rekognition",
            region_name=REGIAO_AWS
        )

        with open(arquivo, "rb") as imagem:
            resposta = rekognition.detect_labels(
                Image={
                    "Bytes": imagem.read()
                },
                MaxLabels=10,
                MinConfidence=70
            )

        labels = resposta.get("Labels", [])

        print("\nAnálise visual concluída com sucesso!")

        if not labels:
            print("\nNenhum elemento foi identificado na imagem.")
            return

        print("\n----------------------------------------------")
        print("ELEMENTOS IDENTIFICADOS")
        print("----------------------------------------------\n")

        for numero, label in enumerate(labels, start=1):
            nome = label["Name"]
            confianca = label["Confidence"]

            print(
                f"{numero:02d}. "
                f"{nome:<20} "
                f"{confianca:6.2f}%"
            )

        print("\n----------------------------------------------")
        print("GERANDO DESCRIÇÃO COM IA")
        print("----------------------------------------------\n")

        descricao = criar_descricao_com_ia(labels)

        print("----------------------------------------------")
        print("DESCRIÇÃO DA IMAGEM")
        print("----------------------------------------------\n")

        print(descricao)

        print("\n----------------------------------------------")
        print("RESUMO")
        print("----------------------------------------------")

        print(f"Total de elementos identificados: {len(labels)}")

        maior_confianca = labels[0]

        print(
            f"Maior confiança: "
            f"{maior_confianca['Name']} "
            f"({maior_confianca['Confidence']:.2f}%)"
        )

        print("Visão computacional: Amazon Rekognition")
        print("Descrição textual: OpenAI")

        print("\nProcessamento finalizado.\n")

    except NoCredentialsError:
        print("\n[ERRO] Credenciais AWS não encontradas.")
        print("Execute 'aws configure' antes de utilizar o sistema.")

    except ClientError as erro:
        print("\n[ERRO] Falha ao acessar o Amazon Rekognition.")

        mensagem = erro.response["Error"]["Message"]

        print(f"Detalhes: {mensagem}")

    except Exception as erro:
        print("\n[ERRO] Ocorreu um erro inesperado.")
        print(f"Detalhes: {erro}")


if __name__ == "__main__":
    analisar_imagem(CAMINHO_IMAGEM)