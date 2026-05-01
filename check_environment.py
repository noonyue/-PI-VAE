"""
环境检查脚本
Environment Check Script

快速检查所有依赖是否正确安装
"""

import sys
import importlib.util

def check_package(package_name, import_name=None):
    """检查包是否安装"""
    if import_name is None:
        import_name = package_name

    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"[OK] {package_name:20s} {version}")
            return True
        else:
            print(f"[NO] {package_name:20s} NOT INSTALLED")
            return False
    except Exception as e:
        print(f"[ER] {package_name:20s} ERROR: {e}")
        return False

def main():
    print("=" * 60)
    print("环境依赖检查 / Environment Check")
    print("=" * 60)
    print()

    # 核心依赖
    print("核心依赖 / Core Dependencies:")
    print("-" * 60)

    packages = [
        ('PyTorch', 'torch'),
        ('NumPy', 'numpy'),
        ('Pandas', 'pandas'),
        ('SciPy', 'scipy'),
        ('Scikit-learn', 'sklearn'),
        ('Matplotlib', 'matplotlib'),
        ('Seaborn', 'seaborn'),
        ('OpenPyXL', 'openpyxl'),
        ('PyYAML', 'yaml'),
        ('tqdm', 'tqdm'),
    ]

    results = []
    for pkg_name, import_name in packages:
        results.append(check_package(pkg_name, import_name))

    # Stage 2依赖
    print()
    print("Stage 2 依赖 / Stage 2 Dependencies:")
    print("-" * 60)

    stage2_packages = [
        ('SHAP', 'shap'),
    ]

    for pkg_name, import_name in stage2_packages:
        results.append(check_package(pkg_name, import_name))

    # PyTorch详细信息
    print()
    print("PyTorch 详细信息 / PyTorch Details:")
    print("-" * 60)

    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch未安装")

    # Python版本
    print()
    print("Python 信息 / Python Info:")
    print("-" * 60)
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")

    # 数据文件检查
    print()
    print("数据文件检查 / Data File Check:")
    print("-" * 60)

    import os
    data_file = "Sampedata0.xlsx"
    if os.path.exists(data_file):
        size_mb = os.path.getsize(data_file) / (1024 * 1024)
        print(f"[OK] {data_file} exists (size: {size_mb:.2f} MB)")
    else:
        print(f"[NO] {data_file} not found")
        results.append(False)

    # 总结
    print()
    print("=" * 60)
    if all(results):
        print("[OK] All dependencies installed! Ready to train.")
        print()
        print("Next steps:")
        print("  1. Run tests: python test_modules.py")
        print("  2. Start training: python train_upgraded_system.py")
        return 0
    else:
        print("[NO] Some dependencies missing!")
        print()
        print("Install missing dependencies:")
        print("  pip install -r requirements.txt")
        print()
        print("See installation guide:")
        print("  INSTALLATION.md")
        print("  QUICK_INSTALL.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
