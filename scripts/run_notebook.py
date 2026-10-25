"""Run a notebook top to bottom and save its outputs in place, e.g. to check the walkthrough
still runs after changing it. The outputs stay in your copy: git strips them at commit time
(nbstripout, see .gitattributes).

    python scripts/run_notebook.py notebooks/walkthrough.ipynb [kernel_name]

Uses jupyter_client, which comes with ipykernel, so nbconvert/nbclient are not needed. The
kernel defaults to sharable_mock_obs. It stops at the first failing cell (the outputs so far are
still written).
"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from jupyter_client import KernelManager

nb_path = Path(sys.argv[1]).resolve()
kernel = sys.argv[2] if len(sys.argv) > 2 else "sharable_mock_obs"
nb = json.loads(nb_path.read_text())

# Local IPC sockets rather than unencrypted TCP on a shared login node. Unix socket paths
# must be short (~108 chars), so they go in a private temporary directory.
ipc_dir = tempfile.mkdtemp(prefix="kipc-")
km = KernelManager(kernel_name=kernel, transport="ipc", ip=f"{ipc_dir}/k")
km.start_kernel(cwd=str(nb_path.parent))
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=120)
failed = False
try:
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        outputs = []

        def hook(msg, outputs=outputs, count=count):
            kind, c = msg["msg_type"], msg["content"]
            if kind == "stream":
                if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == c["name"]:
                    outputs[-1]["text"] += c["text"]
                else:
                    outputs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
            elif kind == "display_data":
                outputs.append({"output_type": "display_data", "data": c["data"], "metadata": c.get("metadata", {})})
            elif kind == "execute_result":
                outputs.append({"output_type": "execute_result", "data": c["data"],
                                "metadata": c.get("metadata", {}), "execution_count": count})
            elif kind == "error":
                outputs.append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"],
                                "traceback": c["traceback"]})

        t0 = time.time()
        reply = kc.execute_interactive("".join(cell["source"]), output_hook=hook, timeout=3600)
        for o in outputs:          # nbformat stores text as a list of lines
            if o["output_type"] == "stream":
                o["text"] = o["text"].splitlines(keepends=True)
        cell["outputs"], cell["execution_count"] = outputs, count
        print(f"cell {count}: {reply['content']['status']} ({time.time() - t0:.1f} s)", flush=True)
        if reply["content"]["status"] != "ok":
            failed = True
            print("\n".join(outputs[-1].get("traceback", [])) if outputs else "", flush=True)
            break
finally:
    kc.stop_channels()
    km.shutdown_kernel(now=True)
    shutil.rmtree(ipc_dir, ignore_errors=True)

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(("FAILED: " if failed else "wrote ") + str(nb_path))
sys.exit(1 if failed else 0)
