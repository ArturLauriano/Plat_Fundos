from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
INIT_PATH = ROOT / "streamlit_searchbox_local" / "__init__.py"
JS_PATH = ROOT / "streamlit_searchbox_local" / "frontend" / "build" / "static" / "js" / "main.540e12aa.js"


def _check_file_contains(path: Path, patterns: list[str]) -> list[str]:
    missing: list[str] = []
    content = path.read_text(encoding="utf-8")
    for pattern in patterns:
        if pattern not in content:
            missing.append(pattern)
    return missing


def main() -> int:
    if not INIT_PATH.exists():
        print(f"[ERRO] Arquivo ausente: {INIT_PATH}")
        return 1
    if not JS_PATH.exists():
        print(f"[ERRO] Arquivo ausente: {JS_PATH}")
        return 1

    init_missing = _check_file_contains(
        INIT_PATH,
        [
            'components.declare_component(',
            '"searchbox_local"',
            "class SearchboxStyle",
            "menu: dict | None",
            "menuPortal: dict | None",
        ],
    )
    js_missing = _check_file_contains(
        JS_PATH,
        [
            "this.select={menu:n=>({...n,backgroundColor:e.backgroundColor,...t.menu||{}}),",
            "menuPortal:n=>({...n,...t.menuPortal||{}}),",
            "menuList:n=>({...n,backgroundColor:e.backgroundColor,...t.menuList||{}}),",
        ],
    )

    if init_missing or js_missing:
        print("[ERRO] Patch do streamlit_searchbox_local incompleto.")
        for pattern in init_missing:
            print(f"  - Ausente em __init__.py: {pattern}")
        for pattern in js_missing:
            print(f"  - Ausente no JS: {pattern}")
        return 1

    print("[OK] Patch do streamlit_searchbox_local está aplicado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
