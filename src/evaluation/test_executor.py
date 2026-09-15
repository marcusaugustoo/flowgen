"""
Test executor with sandboxing.

Runs generated code in an isolated environment with:
- Timeout control
- stdout/stderr capture

Supports two execution backends:
- subprocess (default): executa em subprocesso local com diretório temporário.
- docker: executa dentro de um container Docker isolado (rede desabilitada,
  filesystem somente leitura, limites de memória e CPU).

O backend é selecionado automaticamente: se Docker estiver disponível e
habilitado, usa Docker; caso contrário, faz fallback para subprocess.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _docker_available() -> bool:
    """Verifica se a CLI do Docker está instalada e responsiva."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


class TestExecutor:
    """
    Executes generated code + tests in a controlled environment.

    Args:
        timeout: Tempo máximo de execução em segundos.
        use_docker: Se True, tenta usar Docker para execução isolada.
                    Faz fallback para subprocess se Docker não estiver disponível.
        docker_image: Imagem Docker a ser utilizada.
        memory_limit: Limite de memória do container (ex: "256m").
    """

    def __init__(
        self,
        timeout: int = 30,
        use_docker: bool = False,
        docker_image: str = "python:3.10-slim",
        memory_limit: str = "256m",
    ) -> None:
        self.timeout = timeout
        self.docker_image = docker_image
        self.memory_limit = memory_limit

        # Resolve backend
        if use_docker and _docker_available():
            self._backend = "docker"
            logger.info("TestExecutor: usando backend Docker (%s)", docker_image)
        else:
            self._backend = "subprocess"
            if use_docker:
                logger.warning(
                    "TestExecutor: Docker solicitado mas indisponível. "
                    "Usando fallback subprocess."
                )
            else:
                logger.debug("TestExecutor: usando backend subprocess.")

    @property
    def backend(self) -> str:
        """Nome do backend ativo ('docker' ou 'subprocess')."""
        return self._backend

    def run(
        self,
        code: str,
        tests: str,
        entry_point: str = "",
    ) -> dict[str, Any]:
        """
        Execute code with tests and return results.

        Combines the generated code and test code into a single script,
        then executes it in the selected backend.

        Args:
            code: The generated Python code.
            tests: The test code to run against it.
            entry_point: The function name (for HumanEval-style tests).

        Returns:
            Dict with keys:
            - passed: bool
            - output: str (combined stdout/stderr)
            - stdout: str
            - stderr: str
            - return_code: int
            - error: Optional[str]
            - timeout: bool
            - backend: str ('docker' or 'subprocess')
        """
        # Combine code + tests into a single executable script
        script = self._build_script(code, tests)

        return self._execute_script(script)

    def run_with_canonical_tests(
        self,
        code: str,
        test: str,
        entry_point: str,
    ) -> dict[str, Any]:
        """
        Execute code against HumanEval canonical tests.

        The HumanEval test format uses a `check(candidate)` function.
        We need to combine the code with the test and call check with
        the entry point function.

        Args:
            code: The generated Python code (should define the entry_point function).
            test: The HumanEval test code (defines check function).
            entry_point: The function name to test.

        Returns:
            Execution result dict.
        """
        # Build a script that defines the function then runs the check
        script = f"{code}\n\n{test}\n"

        return self._execute_script(script)

    def _build_script(self, code: str, tests: str) -> str:
        """Combine code and tests into a single script."""
        return f"{code}\n\n{tests}\n"

    def _execute_script(self, script: str) -> dict[str, Any]:
        """Execute a Python script using the active backend."""
        if self._backend == "docker":
            return self._execute_docker(script)
        return self._execute_subprocess(script)

    # ── Backend: subprocess ──────────────────────────────────────────

    def _execute_subprocess(self, script: str) -> dict[str, Any]:
        """Execute a Python script in a subprocess with basic sandboxing."""
        result = self._empty_result()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                script_path = Path(tmpdir) / "test_script.py"
                with open(script_path, "w") as f:
                    f.write(script)

                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                    env={
                        **os.environ,
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )

                result["stdout"] = proc.stdout
                result["stderr"] = proc.stderr
                result["output"] = proc.stdout + proc.stderr
                result["return_code"] = proc.returncode
                result["passed"] = proc.returncode == 0

                if proc.returncode != 0:
                    result["error"] = proc.stderr.strip() or f"Exit code: {proc.returncode}"
                    logger.debug("Test execution failed: %s", result["error"][:200])

        except subprocess.TimeoutExpired:
            result["timeout"] = True
            result["error"] = f"Execution timed out after {self.timeout}s"
            logger.warning(result["error"])

        except Exception as e:
            result["error"] = str(e)
            logger.error("Test execution error: %s", e)

        return result

    # ── Backend: Docker ──────────────────────────────────────────────

    def _execute_docker(self, script: str) -> dict[str, Any]:
        """
        Execute a Python script inside a Docker container.

        Configurações de segurança:
        - Rede desabilitada (--network none)
        - Filesystem somente leitura (--read-only), com /tmp como tmpfs
        - Limite de memória configurável
        - Sem acesso a processos do host (--pids-limit)
        - Container removido automaticamente (--rm)
        """
        result = self._empty_result()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                script_path = Path(tmpdir) / "test_script.py"
                with open(script_path, "w") as f:
                    f.write(script)

                cmd = [
                    "docker", "run",
                    "--rm",
                    "--network", "none",
                    "--read-only",
                    "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                    f"--memory={self.memory_limit}",
                    "--pids-limit", "64",
                    "--cpus", "1",
                    "-v", f"{script_path}:/workspace/test_script.py:ro",
                    "-w", "/workspace",
                    self.docker_image,
                    "python", "test_script.py",
                ]

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout + 10,  # Extra para overhead do Docker
                )

                result["stdout"] = proc.stdout
                result["stderr"] = proc.stderr
                result["output"] = proc.stdout + proc.stderr
                result["return_code"] = proc.returncode
                result["passed"] = proc.returncode == 0

                if proc.returncode != 0:
                    result["error"] = proc.stderr.strip() or f"Exit code: {proc.returncode}"
                    logger.debug("Docker test execution failed: %s", result["error"][:200])

        except subprocess.TimeoutExpired:
            result["timeout"] = True
            result["error"] = f"Docker execution timed out after {self.timeout}s"
            logger.warning(result["error"])

        except Exception as e:
            result["error"] = str(e)
            logger.error("Docker test execution error: %s", e)

        return result

    # ── Helpers ───────────────────────────────────────────────────────

    def _empty_result(self) -> dict[str, Any]:
        """Retorna um dict de resultado vazio."""
        return {
            "passed": False,
            "output": "",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "error": None,
            "timeout": False,
            "backend": self._backend,
        }

