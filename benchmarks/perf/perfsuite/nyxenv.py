"""Make the nyxus extension importable before `import nyxus`, and perfsuite importable in child processes.

On Windows, Python 3.8+ resolves an extension module's dependent DLLs without consulting PATH. A
GPU build's backend needs the CUDA runtime, so its directory is registered here: the `bin\\x64` and
`bin` folders under CUDA_PATH, plus any directories listed in PERFSUITE_DLL_DIRS
(os.pathsep-separated). Elsewhere this does nothing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_done = False
_PERFSUITE_PARENT = str(Path(__file__).resolve().parent.parent)


def child_env() -> dict:
    """This process's environment with perfsuite's parent directory on PYTHONPATH, once, so a
    `python -m perfsuite.<module>` child finds the package from any working directory."""
    env = dict(os.environ)
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if not any(os.path.normcase(os.path.abspath(p)) == os.path.normcase(_PERFSUITE_PARENT) for p in parts):
        parts.insert(0, _PERFSUITE_PARENT)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def prepare() -> None:
    global _done
    if _done or sys.platform != "win32":
        return
    _done = True
    dirs = [d for d in os.environ.get("PERFSUITE_DLL_DIRS", "").split(os.pathsep) if d]
    cuda = os.environ.get("CUDA_PATH")
    if cuda:
        dirs += [os.path.join(cuda, "bin", "x64"), os.path.join(cuda, "bin")]
    for d in dirs:
        if os.path.isdir(d):
            os.add_dll_directory(d)


def gpu_info() -> dict:
    """{'gpu_available': bool, 'gpu': description} from a throwaway Nyxus instance."""
    prepare()
    import nyxus

    probe = nyxus.Nyxus(["MEAN"])
    ok = bool(nyxus.gpu_is_available(id(probe)))
    out = {"gpu_available": ok}
    if ok:
        props = nyxus.get_gpu_properties(id(probe))
        out["gpu"] = "; ".join(p.get("name", str(p)) if isinstance(p, dict) else str(p) for p in props)
    return out
