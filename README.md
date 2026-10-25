# IARA
![IARA](assets/IAra-logo.jpg)

Assistente pessoal de IA, no estilo Copilot, rodando localmente.

## Sobre

A IARA é um projeto pessoal para unir duas coisas que eu gosto: programação e inteligência artificial. O nome vem da lenda amazônica da sereia Iara, que encanta quem se aproxima das águas.

A ideia é construir um assistente que roda na minha própria máquina, sem depender de serviços pagos. O projeto tem duas trilhas em paralelo:

- **Trilha A, aprendizado:** construir um mini-LLM do zero (tokenizador, atenção, Transformer e geração de texto) para entender como esses modelos funcionam por dentro.
- **Trilha B, produto:** montar a IARA de verdade, usando um modelo open source rodando localmente, com backend, interface, memória sobre meus arquivos (RAG) e ferramentas.

Não há pretensão comercial. É um projeto de estudo e portfólio.

## Status

**Fase 0, preparação.** Nada funcional ainda.
🚧 **Fase 1, fundamentos.** Fase 0 concluída, o mini-GPT vem a seguir.

## Roadmap

- [X] **Fase 0:** preparação (repositório, ambiente, contas Colab e Kaggle)
- [ ] **Fase 1:** fundamentos, mini-GPT do zero (Trilha A)
- [ ] **Fase 2:** modelo local com Ollama e persona da IARA (Trilha B)
- [ ] **Fase 3:** backend com FastAPI, interface e histórico de conversas
- [ ] **Fase 4:** RAG, modo código e ferramentas
- [ ] **Fase 5:** fine-tuning com LoRA/QLoRA (opcional)
- [ ] **Fase 6:** testes, documentação e artigo

## Stack prevista

Python, PyTorch, Ollama, FastAPI, SQLite, ChromaDB, Gradio.

## Como rodar

Em breve.