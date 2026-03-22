import subprocess
import concurrent.futures
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging
import os

@dataclass
class CommandResult:
    """A simple data structure to hold the output of a command."""
    command: str
    success: bool
    exit_code: int
    log_file: str

class ParallelExecutor:
    def __init__(self, 
                 max_workers: int = 4, 
                 timeout: int = 300,
                 logger: Optional[logging.Logger] = None,
                 gpus: Optional[List[int]] = None,
                 max_processes_per_gpu: int = 1):
        """
        Initialize the executor.
        :param max_workers: Number of parallel processes.
        :param timeout: Max seconds to wait for a single command.
        :param logger: An optional logger object for HIGH-LEVEL status.
        :param gpus: Optional list of GPU indices to pin each process to.
        :param max_processes_per_gpu: Max concurrent processes allowed per GPU.
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.commands = []
        self.logger = logger
        self.gpus = gpus
        self.max_processes_per_gpu = max(1, max_processes_per_gpu)

    def add_command(self, command: str):
        self.commands.append(command)

    def add_commands(self, commands: List[str]):
        self.commands.extend(commands)

    def _log(self, level: str, message: str):
        """Helper to log to the main logger if it exists, or print."""
        if self.logger:
            getattr(self.logger, level.lower())(message)
        else:
            print(f"[{level.upper()}] {message}")

    def log_commands(self):
        """Logs all commands currently in the queue to the main logger."""
        self._log("info", "--- Commands in Queue ---")
        if not self.commands:
            self._log("info", "  No commands in queue.")
            self._log("info", "---------------------------")
            return

        for i, cmd in enumerate(self.commands):
            # Using :04d to pad the number, e.g., 0001, 0002
            self._log("info", f"  [job_{i:04d}]: {cmd}")
        self._log("info", "---------------------------")

    @staticmethod
    def _worker(cmd_info: Tuple[str, int, str, Optional[Dict[str, str]]]) -> CommandResult:
        """
        Internal static worker method.
        Runs a command AND writes its output to a dedicated log file.
        """
        command, timeout, log_path, env = cmd_info
        
        exit_code = -1
        status_msg = ""
        success = False

        try:
            with open(log_path, "w") as log_file:
                log_file.write(f"COMMAND: {command}\n")
                log_file.write(f"LOG_FILE: {log_path}\n")
                log_file.write("-" * 40 + "\n\n")
                log_file.write("--- STREAMING OUTPUT ---\n")
                log_file.flush()

                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env if env is not None else None,
                )
                wait_kwargs = {}
                if timeout != -1:
                    wait_kwargs["timeout"] = timeout
                process.wait(**wait_kwargs)
                exit_code = process.returncode
            success = (exit_code == 0)
            status_msg = "SUCCESS" if success else "FAILURE"

        except subprocess.TimeoutExpired as e:
            status_msg = "TIMEOUT"
            exit_code = -1
            with open(log_path, "a") as log_file:
                log_file.write(f"\nCommand timed out after {timeout} seconds\n")
        except Exception as e:
            status_msg = "WORKER_ERROR"
            exit_code = -1
            with open(log_path, "a") as log_file:
                log_file.write(f"\nWorker failed with Python exception: {str(e)}\n")

        # Append status summary to the log file
        try:
            with open(log_path, "a") as log_file:
                log_file.write("\n" + "-" * 40 + "\n")
                log_file.write(f"STATUS: {status_msg}\n")
                log_file.write(f"EXIT_CODE: {exit_code}\n")
        except Exception as e:
            with open(log_path, "a") as log_file:
                log_file.write(f"\nCRITICAL: Failed to write log summary with error: {e}\n")

        return CommandResult(
            command=command,
            success=success,
            exit_code=exit_code,
            log_file=log_path
        )

    def _detect_available_gpus(self) -> List[int]:
        """Best-effort detection of available GPU indices."""
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            try:
                indices = [int(idx.strip()) for idx in cuda_visible.split(",") if idx.strip() != ""]
                if indices:
                    return indices
            except ValueError:
                pass

        try:
            import torch  # type: ignore
            count = torch.cuda.device_count()
            if count:
                return list(range(count))
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
                if lines:
                    return list(range(len(lines)))
        except FileNotFoundError:
            pass

        return []

    def _prepare_gpu_envs(self, job_count: int) -> List[Optional[Dict[str, str]]]:
        """Create per-job environment mappings with CUDA_VISIBLE_DEVICES set."""
        if not self.gpus:
            return [None] * job_count

        available = self._detect_available_gpus()
        if not available:
            raise RuntimeError("GPU affinity requested but no GPUs were detected.")

        if len(self.gpus) > len(available):
            raise RuntimeError(
                f"Requested {len(self.gpus)} GPUs but only {len(available)} detected."
            )

        invalid = [gpu for gpu in self.gpus if gpu not in available]
        if invalid:
            raise RuntimeError(f"Requested GPU(s) {invalid} not available on this host.")

        capacity = len(self.gpus) * self.max_processes_per_gpu
        if self.max_workers > capacity:
            raise RuntimeError(
                f"max_workers ({self.max_workers}) cannot exceed GPUs * max_processes_per_gpu ({capacity})."
            )

        gpu_slots = []
        for gpu in self.gpus:
            gpu_slots.extend([gpu] * self.max_processes_per_gpu)

        envs: List[Optional[Dict[str, str]]] = []
        for i in range(job_count):
            gpu = gpu_slots[i % len(gpu_slots)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            envs.append(env)
        return envs

    def _log_result(self, res: CommandResult):
        """Logs a high-level summary of the result to the *main* logger."""
        if res.success:
            self._log("info", f"SUCCESS (Code {res.exit_code}): {res.command[:70]}... -> {res.log_file}")
        else:
            self._log("error", f"FAILURE (Code {res.exit_code}): {res.command[:70]}... -> {res.log_file}")

    def run(self, command_log_dir: str) -> List[CommandResult]:
        """
        Execute all queued commands in parallel.
        :param command_log_dir: The directory to store individual command logs.
        """
        # Create the dedicated log directory
        try:
            os.makedirs(command_log_dir, exist_ok=True)
        except Exception as e:
            self._log("critical", f"Could not create command log directory: {e}")
            return []

        gpu_envs = self._prepare_gpu_envs(len(self.commands))

        # Prepare all work items, including the unique log path for each
        work_items = []
        for i, cmd in enumerate(self.commands):
            # Create a unique, padded log filename, e.g., 'job_0001.log'
            log_filename = f"job_{i:04d}.log"
            log_path = os.path.join(command_log_dir, log_filename)
            work_items.append((cmd, self.timeout, log_path, gpu_envs[i]))

        results = []
        
        # --- MODIFICATION: Call log_commands() before starting ---
        self.log_commands()
        # --- END MODIFICATION ---
        
        self._log("info",  "--- ParallelExecutor Starting ---")
        self._log("info", f"Total commands: {len(self.commands)}")
        self._log("info", f"Max processes: {self.max_workers}")
        self._log("info", f"Writing individual command logs to: {command_log_dir}")
        self._log("info", "---------------------------------")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for result in executor.map(self._worker, work_items):
                results.append(result)
                self._log_result(result)

        self._log("info", "--- ParallelExecutor Finished ---")
        return results
