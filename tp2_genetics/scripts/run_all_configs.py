#!/usr/bin/env python3
"""
Script to run genetic algorithm with all configuration files in parallel.
Accepts a parameter to control the maximum number of parallel tasks.
"""

import argparse
import atexit
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Global list to track running processes for cleanup
running_processes = []
shutdown_event = threading.Event()

def cleanup_processes():
    """Terminate all running processes."""
    if not running_processes:
        return

    print("\n\n⚠️  Terminating all processes...")

    for process in running_processes[:]:  # Create a copy to iterate safely
        try:
            if process.poll() is None:  # Process is still running
                print(f"Terminating process with PID {process.pid}")
                if sys.platform == "win32":
                    # On Windows, use taskkill for more reliable termination
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                                 capture_output=True, check=False)
                else:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
        except Exception as e:
            print(f"Error terminating process: {e}")

    running_processes.clear()
    print("All processes terminated.")

# Register cleanup function to run on exit
atexit.register(cleanup_processes)

def run_config(config_file, verbose=False):
    """Run the genetic algorithm with a specific config file."""
    if shutdown_event.is_set():
        return config_file, False, "", "Interrupted before start"

    config_path = f"./configs/{config_file}"
    cmd = [sys.executable, "-u", "./main.py", config_path]

    print(f"Starting: {config_file}")

    try:
        # Create process with proper Windows flags
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if verbose else subprocess.PIPE,
            text=True,
            bufsize=1 if verbose else -1,
            universal_newlines=True,
            cwd=Path(__file__).parent.parent,
            creationflags=creation_flags
        )

        # Add to global list for cleanup
        running_processes.append(process)

        if verbose:
            print(f"\n--- Real-time output for {config_file} ---")
            stdout_lines = []

            def read_output():
                try:
                    while True:
                        if shutdown_event.is_set():
                            break
                        line = process.stdout.readline()
                        if not line:
                            break
                        print(f"[{config_file}] {line.rstrip()}", flush=True)
                        stdout_lines.append(line)
                except Exception:
                    pass

            output_thread = threading.Thread(target=read_output)
            output_thread.daemon = True
            output_thread.start()

            # Poll for completion or shutdown
            while process.poll() is None:
                if shutdown_event.is_set():
                    break
                time.sleep(0.1)

            # Clean shutdown if requested
            if shutdown_event.is_set() and process.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                                 capture_output=True, check=False)
                else:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()

            output_thread.join(timeout=1)
            stdout = ''.join(stdout_lines)
        else:
            # Non-verbose mode
            while process.poll() is None:
                if shutdown_event.is_set():
                    break
                time.sleep(0.1)

            if shutdown_event.is_set() and process.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                                 capture_output=True, check=False)
                else:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                stdout, stderr = "", "Interrupted by user"
            else:
                stdout, stderr = process.communicate()

        # Remove from global list when done
        if process in running_processes:
            running_processes.remove(process)

        if shutdown_event.is_set():
            return config_file, False, stdout if verbose else "", "Interrupted by user"
        elif process.returncode == 0:
            if verbose:
                print(f"✓ Completed: {config_file}", flush=True)
                print("-" * 50, flush=True)
                return config_file, True, stdout, ""
            else:
                print(f"✓ Completed: {config_file}")
                return config_file, True, stdout, ""
        else:
            if verbose:
                print(f"✗ Failed: {config_file} (exit code: {process.returncode})", flush=True)
                print("-" * 50, flush=True)
                return config_file, False, stdout, ""
            else:
                print(f"✗ Failed: {config_file} (exit code: {process.returncode})")
                return config_file, False, stdout, stderr if 'stderr' in locals() else ""

    except Exception as e:
        print(f"✗ Error running {config_file}: {str(e)}")
        return config_file, False, "", str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Run genetic algorithm with all config files in parallel"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Number of parallel tasks to run (default: 4)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output from each run"
    )

    args = parser.parse_args()

    # Get the project root directory
    project_root = Path(__file__).parent.parent
    configs_dir = project_root / "configs"

    # Find all .json config files
    if not configs_dir.exists():
        print(f"Error: configs directory not found at {configs_dir}")
        sys.exit(1)

    config_files = [f.name for f in configs_dir.glob("*.json")]

    if not config_files:
        print("No config files found in the configs directory")
        sys.exit(1)

    print(f"Found {len(config_files)} config files:")
    for config in sorted(config_files):
        print(f"  - {config}")

    print(f"\nRunning with {args.jobs} parallel tasks...")
    print("Press Ctrl+C to stop all processes and exit.")
    print("-" * 50)

    # Track results
    completed = 0
    failed = 0
    results = []

    # Run configs in parallel with timeout-based polling to handle Ctrl+C
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        # Submit all tasks
        future_to_config = {
            executor.submit(run_config, config, args.verbose): config
            for config in config_files
        }

        # Process completed tasks with timeout polling
        while future_to_config:
            try:
                # Use timeout to allow KeyboardInterrupt to be caught
                done_futures = []
                for future in list(future_to_config.keys()):
                    try:
                        # Check if future is done with a very short timeout
                        result = future.result(timeout=0.1)
                        done_futures.append(future)

                        config_file, success, stdout, stderr = result
                        results.append((config_file, success, stdout, stderr))

                        if success:
                            completed += 1
                        else:
                            failed += 1
                            if args.verbose and stderr:
                                print(f"Error output for {config_file}:")
                                print(stderr)
                                print("-" * 30)

                    except Exception:
                        # Future not ready yet, continue polling
                        pass

                # Remove completed futures
                for future in done_futures:
                    if future in future_to_config:
                        del future_to_config[future]

                # Small sleep to prevent excessive CPU usage
                if future_to_config:  # Only sleep if there are still pending futures
                    time.sleep(0.1)

            except KeyboardInterrupt:
                print("\nKeyboard interrupt received! Terminating all processes...")
                shutdown_event.set()

                # Cancel all pending futures
                for future in future_to_config:
                    future.cancel()

                # Force cleanup
                cleanup_processes()

                # Shutdown executor immediately
                executor.shutdown(wait=False)

                print("All processes have been terminated.")
                sys.exit(0)

    # Check if we were interrupted
    if shutdown_event.is_set():
        print("\nExecution was interrupted by user.")
        return

    # Print summary
    print("-" * 50)
    print(f"Execution Summary:")
    print(f"  Total configs: {len(config_files)}")
    print(f"  Completed successfully: {completed}")
    print(f"  Failed: {failed}")

    if failed > 0:
        print(f"\nFailed configurations:")
        for config_file, success, stdout, stderr in results:
            if not success:
                print(f"  - {config_file}")
        sys.exit(1)
    else:
        print("\n✓ All configurations completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt in main! Stopping all processes...")
        shutdown_event.set()
        cleanup_processes()
        sys.exit(0)
