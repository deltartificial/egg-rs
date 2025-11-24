#!/usr/bin/env python3
import subprocess
import sys
import os
import re
import shutil
import time

EXPECTED_LOSS = 8.2900
TOLERANCE = 0.0001
BENCH_DURATION = 10

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bench.py <path_to_egg_binary>")
        print("Example: python3 bench.py ./egg")
        sys.exit(1)

    binary = sys.argv[1]
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(bench_dir, "input.txt")

    binary = os.path.abspath(binary)
    if not os.path.isfile(binary):
        print(f"Error: Binary '{binary}' not found")
        sys.exit(1)

    if not os.access(binary, os.X_OK):
        print(f"Error: Binary '{binary}' is not executable")
        sys.exit(1)

    print("=" * 40)
    print("EGG Benchmark Tool")
    print("=" * 40)
    print(f"Binary:   {binary}")
    print(f"Input:    {input_file}")
    print(f"Duration: {BENCH_DURATION}s")
    print("=" * 40)
    print()

    work_dir = os.path.dirname(os.path.abspath(binary)) or "."
    dest_input = os.path.join(work_dir, "input.txt")

    copied = False
    if os.path.abspath(input_file) != os.path.abspath(dest_input):
        shutil.copy(input_file, dest_input)
        copied = True

    try:
        process = subprocess.Popen(
            ["script", "-q", "/dev/null", binary],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=work_dir
        )

        output_lines = []
        start_time = time.time()

        print(f"Running benchmark for {BENCH_DURATION}s...")

        try:
            import select
            while time.time() - start_time < BENCH_DURATION:
                ready, _, _ = select.select([process.stdout], [], [], 0.1)
                if ready:
                    chunk = process.stdout.read(4096)
                    if not chunk:
                        break
                    output_lines.append(chunk.decode('utf-8', errors='ignore'))
                    if b"Training Done" in chunk:
                        break
                if process.poll() is not None:
                    remaining = process.stdout.read()
                    if remaining:
                        output_lines.append(remaining.decode('utf-8', errors='ignore'))
                    break

            process.terminate()
            try:
                process.wait(timeout=2)
            except:
                process.kill()
        except:
            process.kill()

        raw = "".join(output_lines)
        lines = re.findall(r'Step \d+/\d+ \| Loss: [\d.]+(?: \| Tok/s: [\d.]+)?', raw)
        output = "\n".join(lines)

    finally:
        if copied and os.path.exists(dest_input):
            os.remove(dest_input)

    step0_match = re.search(r"Step 0/\d+ \| Loss: ([\d.]+)", output)
    tps_matches = re.findall(r"Step \d+/\d+ \| Loss: [\d.]+ \| Tok/s: ([\d.]+)", output)
    step_matches = re.findall(r"Step (\d+)/", output)

    if not step0_match:
        print("FAIL: Could not find Step 0 output")
        print("Output was:")
        print(output[:500])
        sys.exit(1)

    loss = float(step0_match.group(1))

    last_step = 0
    if step_matches:
        last_step = max(int(s) for s in step_matches)

    avg_tps = None
    if tps_matches:
        tps_values = [float(t) for t in tps_matches if float(t) > 0]
        if tps_values:
            avg_tps = sum(tps_values) / len(tps_values)

    print()
    print("Results:")
    print("-" * 20)
    print(f"Step 0 Loss:  {loss:.4f}")
    print(f"Last Step:    {last_step}")
    if avg_tps:
        print(f"Avg Tok/s:    {avg_tps:.2f}")
    print()

    if abs(loss - EXPECTED_LOSS) < TOLERANCE:
        print(f"✓ PASS: Loss matches expected ({EXPECTED_LOSS})")
        result = 0
    else:
        print(f"✗ FAIL: Loss mismatch!")
        print(f"  Expected: {EXPECTED_LOSS}")
        print(f"  Got:      {loss}")
        result = 1

    print()
    print("=" * 40)
    sys.exit(result)

if __name__ == "__main__":
    main()
