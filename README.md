# FlowGen — Plataforma de Pesquisa para Geração de Código Multi-Agente

Infraestrutura de pesquisa experimental projetada para avaliar como diferentes configurações de sistemas multi-agentes e modelos SLM (Small Language Models) locais afetam a qualidade do código gerado. Inspirada na metodologia apresentada no artigo FlowGen (SOEN-101, ICSE 2025).

> **Aviso**: Esta não é uma reprodução do código original do FlowGen. Trata-se de uma implementação experimental independente para rodar benchmarks de SLMs, inspirada nos conceitos do artigo.

## 🚀 Começando

```bash
# 1. Clone o repositório e configure o ambiente
cd flowgen
python3 -m venv .venv
source .venv/bin/activate
pip install pyyaml requests pytest

# 2. Rode os testes (sem necessidade do Ollama)
python -m pytest tests/ -v

# 3. Rode o baseline Raw (necessita Ollama configurado)
ollama pull qwen2.5-coder:7b
python -m src.main --config config/experiments/raw_baseline.yaml

# 4. Rode o baseline Waterfall
python -m src.main --config config/experiments/waterfall_baseline.yaml
```

## 🏗️ Arquitetura do Projeto

```
flowgen-experiment/
├── config/                    # Arquivos de configuração YAML
├── src/
│   ├── agents/                # Implementação dos Agentes (Engenheiro de Requisitos, Arquiteto, etc.)
│   ├── llm/                   # Abstração do provedor LLM (Ollama, etc.)
│   ├── processes/             # Modelos de processos (Raw, Waterfall, TDD, Scrum)
│   ├── orchestration/         # Coordenação dos agentes e armazenamento de artefatos
│   ├── refinement/            # Estratégias de auto-refinamento
│   ├── benchmarks/            # Carregadores de benchmarks (ex: HumanEval)
│   ├── evaluation/            # Execução de testes e cálculo do Pass@1
│   ├── results/               # Armazenamento e relatórios de resultados
│   └── main.py                # Ponto de entrada CLI
├── tests/                     # Testes unitários do framework
├── datasets/                  # Dados de benchmark (ex: mini HumanEval)
└── docs/                      # Documentação extra
```

## 🧠 Princípios de Design

1. **Baseado em Configuração**: Todos os parâmetros experimentais ficam em arquivos YAML.
2. **Agentes Desacoplados**: Os agentes operam sem saber qual modelo de processo os está executando.
3. **LLM Abstraído**: A troca de modelos é feita alterando apenas uma linha na configuração.
4. **Orquestrador Central**: Os agentes não se comunicam diretamente entre si, apenas via orquestrador.
5. **Rastreabilidade**: Os logs capturam prompts, respostas e mensagens trocadas.
6. **Dados Imutáveis**: Cada experimento recebe um ID único (`EXP-NNN`) para não sobrescrever resultados.

## ⚙️ Configurando Experimentos

Os parâmetros globais estão em `config/default.yaml`. Para sobrescrevê-los, crie arquivos na pasta `config/experiments/`:

```yaml
# config/experiments/meu_experimento.yaml
experiment:
  name: run_personalizada
  runs: 5

llm:
  provider: ollama
  model: deepseek-coder:6.7b
  temperature: 0.8

pipeline:
  type: waterfall               # Opções: raw | waterfall | tdd | scrum

self_refinement:
  enabled: true
  iterations: 3
```

Execute o experimento customizado com:
```bash
python -m src.main --config config/experiments/meu_experimento.yaml
```

## 💻 Comandos da CLI

```bash
python -m src.main --config <caminho>    # Arquivo YAML de configuração
python -m src.main --model <nome>        # Sobrescreve o modelo LLM
python -m src.main --process <tipo>      # Sobrescreve o processo (raw/waterfall/tdd/scrum)
python -m src.main --runs <N>            # Altera o número de repetições
python -m src.main --task <caminho>      # Executa uma única tarefa a partir de arquivo .txt
python -m src.main --tasks <diretorio>   # Executa todas as tarefas .txt em um diretório
```


## 📊 Estrutura de Resultados

Os resultados são gravados na pasta `results/` no formato `EXP-NNN_<nome>/`:

```
results/EXP-007_flowgen_baseline/
├── config.yaml          # Configuração exata utilizada
├── summary.json         # Métricas agregadas (ex: cálculo final de Pass@1)
├── report.md            # Relatório em formato de tabela Markdown
└── runs/
    └── run_001/
        ├── results.json # Resultado por tarefa (aprovado/reprovado + código gerado)
        ├── messages.jsonl
        └── artifacts/
```

**Cálculo do Pass@1**: Representa a taxa de sucesso da primeira resposta gerada sem modificações pelo usuário, podendo ser calculada a média ao executar múltiplos *runs* (`--runs N`).

## 🤖 Modelos de Processos Suportados

| Processo     | Descrição                                         | Status                 |
|--------------|---------------------------------------------------|------------------------|
| `raw`        | Chamada única ao LLM (Baseline A)                 | ✅ Implementado         |
| `waterfall`  | Engenheiro → Arquiteto → Dev → QA (Baseline B)    | ✅ Implementado         |
| `tdd`        | Desenvolvimento Orientado a Testes (Baseline C)   | 🏗️ Arquitetura Pronta |
| `scrum`      | Ciclos de Sprint com Scrum Master (Baseline D)    | 🏗️ Arquitetura Pronta |

### Fluxo dos Agentes (Waterfall)
1. **Engenheiro de Requisitos**: Analisa e extrai requisitos do problema.
2. **Arquiteto**: Projeta a solução e os algoritmos.
3. **Desenvolvedor**: Implementa o código Python.
4. **Tester (QA)**: Gera os casos de teste correspondentes.
5. **Auto-Refinamento**: Caso ativado e haja falhas nos testes em sandbox, o código volta ao Desenvolvedor para correção iterativa.

## 🦙 Configurando o Ollama

O FlowGen utiliza modelos locais via Ollama:
```bash
# Instalar Ollama (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# Baixar um modelo de programação (SLM)
ollama pull qwen2.5-coder:7b

# Iniciar o servidor (caso já não esteja rodando)
ollama serve
```


## 📚 Referências
- Lin, F., Kim, D.J., & Chen, T.H. (2025). *SOEN-101: Code Generation by Emulating Software Process Models Using Large Language Model Agents*. ICSE 2025.
