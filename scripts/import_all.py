import sys, pkgutil, importlib, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]  # 工程根
sys.path.insert(0, str(ROOT))

# 只在工程目录内找模块；排除虚拟环境、隐藏目录、tests（可选）
def iter_modules():
    for m in pkgutil.walk_packages([str(ROOT)]):
        name = m.name
        # 跳过 __main__ 启动、隐藏/测试模块（按需调整）
        if any(seg.startswith("_") for seg in name.split(".")):
            continue
        if name.startswith("tests"):
            continue
        yield name

failed = []
for name in iter_modules():
    try:
        importlib.import_module(name)
        print(f"[OK] import {name}")
    except Exception as e:
        failed.append((name, repr(e)))
        print(f"[FAIL] import {name}: {e}")

if failed:
    print("\n=== Import failures ===")
    for n, err in failed:
        print(n, "->", err)
    raise SystemExit(1)