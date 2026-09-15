"""
Utilitário para extração de código de respostas de LLMs.

SLMs frequentemente misturam código com texto de conversação, esquecem
de usar blocos Markdown, ou adicionam comentários não-Python ao redor
do código. Este módulo implementa heurísticas para lidar com esses
cenários de forma centralizada.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Frases comuns de "conversação" que SLMs adicionam antes/depois do código.
# Estas são removidas quando o modelo não usa blocos Markdown.
_CONVERSATIONAL_PATTERNS = [
    re.compile(r"^(here'?s?\s+(is\s+)?(the|my|a|your)\s+.{0,60}:)\s*$", re.IGNORECASE),
    re.compile(r"^(sure[!,.]?\s*.{0,80}:)\s*$", re.IGNORECASE),
    re.compile(r"^(certainly[!,.]?\s*.{0,80}:)\s*$", re.IGNORECASE),
    re.compile(r"^(below\s+is\s+.{0,80}:)\s*$", re.IGNORECASE),
    re.compile(r"^(i'?ll?\s+.{0,80}:)\s*$", re.IGNORECASE),
    re.compile(r"^(let me\s+.{0,80}:)\s*$", re.IGNORECASE),
    re.compile(r"^(hope\s+this\s+helps.*)", re.IGNORECASE),
    re.compile(r"^(this\s+(should|will|would)\s+.{0,80}\.)\s*$", re.IGNORECASE),
    re.compile(r"^(explanation|note|output|example)[:\s]", re.IGNORECASE),
    re.compile(r"^(feel\s+free\s+.{0,80}\.)\s*$", re.IGNORECASE),
]


def extract_code(text: str) -> str:
    """
    Extrai código Python de uma resposta de LLM/SLM.

    A lógica segue a seguinte ordem de prioridade:
    1. Blocos ```python ... ``` (mais confiável).
    2. Blocos ``` ... ``` genéricos.
    3. Heurísticas para identificar código sem marcação Markdown:
       - Remove linhas de "conversa" do início e fim.
       - Identifica regiões contínuas que parecem código Python.

    Args:
        text: Resposta crua do LLM.

    Returns:
        String contendo apenas o código Python extraído.
    """
    if not text or not text.strip():
        return ""

    # ── 1. Blocos ```python ... ``` ──────────────────────────────────
    pattern_python = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern_python, text, re.DOTALL)
    if matches:
        return "\n".join(matches).strip()

    # ── 2. Blocos ``` ... ``` genéricos ──────────────────────────────
    pattern_generic = r"```\s*\n(.*?)```"
    matches = re.findall(pattern_generic, text, re.DOTALL)
    if matches:
        return "\n".join(matches).strip()

    # ── 3. Heurísticas (SLM não usou Markdown) ──────────────────────
    cleaned = _strip_conversational_lines(text)

    # Se após a limpeza sobrou algo que parece código, retorna
    if _looks_like_python(cleaned):
        return cleaned

    # Último recurso: retorna o texto limpo como estava
    logger.debug(
        "Extração de código: nenhum bloco Markdown encontrado, "
        "retornando texto limpo (%d caracteres).",
        len(cleaned),
    )
    return cleaned


def _strip_conversational_lines(text: str) -> str:
    """
    Remove linhas de 'conversa' do início e fim do texto.

    Mantém apenas o núcleo que aparenta ser código Python.
    """
    lines = text.strip().splitlines()

    # Remove linhas de conversa do início
    start = 0
    for i, line in enumerate(lines):
        if _is_conversational(line):
            start = i + 1
        else:
            break

    # Remove linhas de conversa do final
    end = len(lines)
    for i in range(len(lines) - 1, start - 1, -1):
        if _is_conversational(lines[i]):
            end = i
        else:
            break

    result = "\n".join(lines[start:end]).strip()
    return result if result else text.strip()


def _is_conversational(line: str) -> bool:
    """Verifica se uma linha parece ser texto de conversa (não código)."""
    stripped = line.strip()
    if not stripped:
        return False
    for pattern in _CONVERSATIONAL_PATTERNS:
        if pattern.match(stripped):
            return True
    return False


def _looks_like_python(text: str) -> bool:
    """
    Heurística simples para verificar se o texto parece código Python.

    Retorna True se encontrar indicadores de código Python como:
    - Definições de função/classe
    - Imports
    - Indentação consistente
    """
    indicators = [
        r"^\s*def\s+\w+",
        r"^\s*class\s+\w+",
        r"^\s*(from|import)\s+\w+",
        r"^\s*return\s+",
        r"^\s*if\s+.*:",
        r"^\s*for\s+.*:",
        r"^\s*while\s+.*:",
    ]
    lines = text.strip().splitlines()
    if not lines:
        return False

    matches = 0
    for line in lines:
        for indicator in indicators:
            if re.match(indicator, line):
                matches += 1
                break

    # Se pelo menos 20% das linhas não-vazias parecem código, considere válido
    non_empty = sum(1 for l in lines if l.strip())
    return non_empty > 0 and (matches / non_empty) >= 0.15
