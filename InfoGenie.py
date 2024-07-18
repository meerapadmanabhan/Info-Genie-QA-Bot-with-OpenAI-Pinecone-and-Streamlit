{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNr41PGzKsbpZzW9NXJY33c",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/meerapadmanabhan/Info-Genie-QA-Bot-with-OpenAI-Pinecone-and-Streamlit/blob/main/InfoGenie.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Pinecone Installation\n",
        "!pip install --upgrade --quiet pinecone-client pinecone-text pinecone-notebooks"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WWLUNtfwIKtY",
        "outputId": "e9f106db-8984-4874-9928-eda17587bc80"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[?25l     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/67.6 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K     \u001b[91m━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[90m╺\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m30.7/67.6 kB\u001b[0m \u001b[31m2.5 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m67.6/67.6 kB\u001b[0m \u001b[31m1.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Building wheel for wget (setup.py) ... \u001b[?25l\u001b[?25hdone\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install langchain_community"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hQr2_f9bIMr6",
        "outputId": "e7c87f78-f44d-48c4-9ca6-9e28883e344f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting langchain_community\n",
            "  Downloading langchain_community-0.2.7-py3-none-any.whl (2.2 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2.2/2.2 MB\u001b[0m \u001b[31m10.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: PyYAML>=5.3 in /usr/local/lib/python3.10/dist-packages (from langchain_community) (6.0.1)\n",
            "Requirement already satisfied: SQLAlchemy<3,>=1.4 in /usr/local/lib/python3.10/dist-packages (from langchain_community) (2.0.31)\n",
            "Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /usr/local/lib/python3.10/dist-packages (from langchain_community) (3.9.5)\n",
            "Collecting dataclasses-json<0.7,>=0.5.7 (from langchain_community)\n",
            "  Downloading dataclasses_json-0.6.7-py3-none-any.whl (28 kB)\n",
            "Collecting langchain<0.3.0,>=0.2.7 (from langchain_community)\n",
            "  Downloading langchain-0.2.9-py3-none-any.whl (987 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m987.7/987.7 kB\u001b[0m \u001b[31m50.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting langchain-core<0.3.0,>=0.2.12 (from langchain_community)\n",
            "  Downloading langchain_core-0.2.21-py3-none-any.whl (372 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m372.0/372.0 kB\u001b[0m \u001b[31m28.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting langsmith<0.2.0,>=0.1.0 (from langchain_community)\n",
            "  Downloading langsmith-0.1.90-py3-none-any.whl (134 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m134.7/134.7 kB\u001b[0m \u001b[31m15.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: numpy<2,>=1 in /usr/local/lib/python3.10/dist-packages (from langchain_community) (1.25.2)\n",
            "Requirement already satisfied: requests<3,>=2 in /usr/local/lib/python3.10/dist-packages (from langchain_community) (2.31.0)\n",
            "Requirement already satisfied: tenacity!=8.4.0,<9.0.0,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from langchain_community) (8.5.0)\n",
            "Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain_community) (1.3.1)\n",
            "Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain_community) (23.2.0)\n",
            "Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain_community) (1.4.1)\n",
            "Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain_community) (6.0.5)\n",
            "Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain_community) (1.9.4)\n",
            "Requirement already satisfied: async-timeout<5.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain_community) (4.0.3)\n",
            "Collecting marshmallow<4.0.0,>=3.18.0 (from dataclasses-json<0.7,>=0.5.7->langchain_community)\n",
            "  Downloading marshmallow-3.21.3-py3-none-any.whl (49 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m49.2/49.2 kB\u001b[0m \u001b[31m5.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting typing-inspect<1,>=0.4.0 (from dataclasses-json<0.7,>=0.5.7->langchain_community)\n",
            "  Downloading typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)\n",
            "Collecting langchain-text-splitters<0.3.0,>=0.2.0 (from langchain<0.3.0,>=0.2.7->langchain_community)\n",
            "  Downloading langchain_text_splitters-0.2.2-py3-none-any.whl (25 kB)\n",
            "Requirement already satisfied: pydantic<3,>=1 in /usr/local/lib/python3.10/dist-packages (from langchain<0.3.0,>=0.2.7->langchain_community) (2.8.2)\n",
            "Collecting jsonpatch<2.0,>=1.33 (from langchain-core<0.3.0,>=0.2.12->langchain_community)\n",
            "  Downloading jsonpatch-1.33-py2.py3-none-any.whl (12 kB)\n",
            "Requirement already satisfied: packaging<25,>=23.2 in /usr/local/lib/python3.10/dist-packages (from langchain-core<0.3.0,>=0.2.12->langchain_community) (24.1)\n",
            "Collecting orjson<4.0.0,>=3.9.14 (from langsmith<0.2.0,>=0.1.0->langchain_community)\n",
            "  Downloading orjson-3.10.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (141 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m141.1/141.1 kB\u001b[0m \u001b[31m15.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2->langchain_community) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2->langchain_community) (3.7)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2->langchain_community) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2->langchain_community) (2024.7.4)\n",
            "Requirement already satisfied: typing-extensions>=4.6.0 in /usr/local/lib/python3.10/dist-packages (from SQLAlchemy<3,>=1.4->langchain_community) (4.12.2)\n",
            "Requirement already satisfied: greenlet!=0.4.17 in /usr/local/lib/python3.10/dist-packages (from SQLAlchemy<3,>=1.4->langchain_community) (3.0.3)\n",
            "Collecting jsonpointer>=1.9 (from jsonpatch<2.0,>=1.33->langchain-core<0.3.0,>=0.2.12->langchain_community)\n",
            "  Downloading jsonpointer-3.0.0-py2.py3-none-any.whl (7.6 kB)\n",
            "Requirement already satisfied: annotated-types>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from pydantic<3,>=1->langchain<0.3.0,>=0.2.7->langchain_community) (0.7.0)\n",
            "Requirement already satisfied: pydantic-core==2.20.1 in /usr/local/lib/python3.10/dist-packages (from pydantic<3,>=1->langchain<0.3.0,>=0.2.7->langchain_community) (2.20.1)\n",
            "Collecting mypy-extensions>=0.3.0 (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7,>=0.5.7->langchain_community)\n",
            "  Downloading mypy_extensions-1.0.0-py3-none-any.whl (4.7 kB)\n",
            "Installing collected packages: orjson, mypy-extensions, marshmallow, jsonpointer, typing-inspect, jsonpatch, langsmith, dataclasses-json, langchain-core, langchain-text-splitters, langchain, langchain_community\n",
            "Successfully installed dataclasses-json-0.6.7 jsonpatch-1.33 jsonpointer-3.0.0 langchain-0.2.9 langchain-core-0.2.21 langchain-text-splitters-0.2.2 langchain_community-0.2.7 langsmith-0.1.90 marshmallow-3.21.3 mypy-extensions-1.0.0 orjson-3.10.6 typing-inspect-0.9.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install bs4"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "S4STjIkDZSeN",
        "outputId": "b9896d9f-0a09-482d-9af5-029954ea5a29"
      },
      "execution_count": 48,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting bs4\n",
            "  Downloading bs4-0.0.2-py2.py3-none-any.whl (1.2 kB)\n",
            "Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from bs4) (4.12.3)\n",
            "Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->bs4) (2.5)\n",
            "Installing collected packages: bs4\n",
            "Successfully installed bs4-0.0.2\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st\n",
        "import requests\n",
        "from bs4 import BeautifulSoup\n",
        "import openai\n",
        "import pinecone\n",
        "import numpy as np"
      ],
      "metadata": {
        "id": "OuUknwSQK5US"
      },
      "execution_count": 49,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from langchain_community.retrievers import PineconeHybridSearchRetriever"
      ],
      "metadata": {
        "id": "lTgPTIh9IM0O"
      },
      "execution_count": 50,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Initialize Pinecone\n",
        "api_key=\"7f1a3949-c165-400d-9f11-6b2e1ae4a9c0\"\n"
      ],
      "metadata": {
        "id": "VnP0CniNcDCd"
      },
      "execution_count": 51,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "from pinecone import Pinecone,ServerlessSpec\n",
        "index_name=\"infogenie\""
      ],
      "metadata": {
        "id": "dCv5kQXgcDFn"
      },
      "execution_count": 52,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#initialize pinecone client\n",
        "pc=Pinecone(api_key=api_key)\n",
        "\n",
        "#create index\n",
        "if index_name not in pc.list_indexes().names():\n",
        "  pc.create_index(\n",
        "      name=index_name,\n",
        "      dimention=384, #dimentionality of dense model\n",
        "      metric=\"dotproduct\",\n",
        "      spec=ServerlessSpec(cloud=\"aws\",region=\"us-east-1\")\n",
        "\n",
        "      )"
      ],
      "metadata": {
        "id": "JI4HEaXQcDIh"
      },
      "execution_count": 53,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "index=pc.Index(index_name)\n",
        "index"
      ],
      "metadata": {
        "id": "BGwoISF6TRmr",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1b0fd760-fe49-4af5-a37f-2850431cbe29"
      },
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<pinecone.data.index.Index at 0x7a8842fe5f00>"
            ]
          },
          "metadata": {},
          "execution_count": 54
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "openai.api_key = \"sk-None-xwNqvKAZBN9x6HfllXT9T3BlbkFJfniQeZb7ezjy9ZO8Mn6U\"\n"
      ],
      "metadata": {
        "id": "eudUKssyIkoB"
      },
      "execution_count": 55,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def fetch_website_content(url):\n",
        "    response = requests.get(url)\n",
        "    soup = BeautifulSoup(response.text, 'html.parser')\n",
        "    paragraphs = soup.find_all('p')\n",
        "    content = \"\\n\".join([para.get_text() for para in paragraphs])\n",
        "    return content"
      ],
      "metadata": {
        "id": "sCu8QMKCKKZ_"
      },
      "execution_count": 56,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def get_embedding(text, model=\"text-embedding-ada-002\"):\n",
        "    response = openai.Embedding.create(input=[text], model=model)\n",
        "    embedding = response['data'][0]['embedding']\n",
        "    return np.array(embedding)"
      ],
      "metadata": {
        "id": "nx1yxbjhLo4Y"
      },
      "execution_count": 57,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def insert_website_content(url):\n",
        "    content = fetch_website_content(url)\n",
        "    paragraphs = content.split('\\n')\n",
        "    for i, paragraph in enumerate(paragraphs):\n",
        "        if paragraph.strip():\n",
        "            embedding = get_embedding(paragraph)\n",
        "            index.upsert([(f\"{url}_para_{i}\", embedding)])"
      ],
      "metadata": {
        "id": "bPPHWlRiLvAn"
      },
      "execution_count": 58,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def query_pinecone(query):\n",
        "    query_embedding = get_embedding(query)\n",
        "    results = index.query(query_embedding, top_k=5, include_values=True)\n",
        "    return results"
      ],
      "metadata": {
        "id": "cy2c93TfLxkw"
      },
      "execution_count": 59,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def generate_answer(query, context):\n",
        "    response = openai.Completion.create(\n",
        "        engine=\"text-davinci-003\",\n",
        "        prompt=f\"Answer the following question based on the context:\\n\\nContext: {context}\\n\\nQuestion: {query}\",\n",
        "        max_tokens=200\n",
        "    )\n",
        "    return response.choices[0].text.strip()"
      ],
      "metadata": {
        "id": "lKREurHBL0mx"
      },
      "execution_count": 60,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def qa_bot(query):\n",
        "    results = query_pinecone(query)\n",
        "    if results['matches']:\n",
        "        context = \" \".join([res['value'] for res in results['matches']])\n",
        "        answer = generate_answer(query, context)\n",
        "    else:\n",
        "        answer = \"No relevant information found in the indexed content.\"\n",
        "    return answer"
      ],
      "metadata": {
        "id": "FnZIDU8kL3Jq"
      },
      "execution_count": 61,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "st.title(\"InfoGenie🤖\")\n",
        "\n",
        "url = st.text_input(\"Enter the website URL:\")\n",
        "query = st.text_input(\"Enter your question:\")\n",
        "\n",
        "if st.button(\"Fetch and Index Content\"):\n",
        "    with st.spinner(\"Fetching and indexing content...\"):\n",
        "        insert_website_content(url)\n",
        "        st.success(\"Content indexed successfully!\")\n",
        "\n",
        "if st.button(\"Get Answer\"):\n",
        "    with st.spinner(\"Fetching answer...\"):\n",
        "        answer = qa_bot(query)\n",
        "        st.write(\"Answer:\", answer)"
      ],
      "metadata": {
        "id": "8tZWls8aL7JF"
      },
      "execution_count": 62,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "_QO40unOL-GN"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}