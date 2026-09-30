"""
Script de Setup e Validação
Verifica se todas as dependências estão instaladas e configuradas

Execução:
    python setup_check.py
"""

import sys
import subprocess
from pathlib import Path
import importlib.util

def check_python_version():
    """Verifica versão do Python"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print("✓ Python 3.10+ detectado")
        return True
    else:
        print("✗ Python 3.10+ é necessário")
        return False


def check_package(package_name, import_name=None):
    """Verifica se um pacote está instalado"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name} instalado")
        return True
    except ImportError:
        print(f"✗ {package_name} NÃO instalado")
        return False


def check_directory(dir_path, create=False):
    """Verifica se um diretório existe"""
    path = Path(dir_path)
    
    if path.exists():
        print(f"✓ Diretório encontrado: {dir_path}")
        return True
    else:
        if create:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Diretório criado: {dir_path}")
            return True
        else:
            print(f"✗ Diretório não encontrado: {dir_path}")
            return False


def check_file(file_path):
    """Verifica se um arquivo existe"""
    path = Path(file_path)
    
    if path.exists():
        print(f"✓ Arquivo encontrado: {file_path}")
        return True
    else:
        print(f"✗ Arquivo não encontrado: {file_path}")
        return False


def main():
    print("=" * 80)
    print("VERIFICAÇÃO DE SETUP - Datathon Grupo 35")
    print("=" * 80)
    
    all_checks_pass = True
    
    # 1. Verificar Python
    print("\n[1] Verificando Python...")
    if not check_python_version():
        all_checks_pass = False
    
    # 2. Verificar dependências
    print("\n[2] Verificando pacotes Python...")
    packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('jupyter', 'jupyter'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('mlflow', 'mlflow'),
        ('requests', 'requests'),
    ]
    
    missing_packages = []
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            missing_packages.append(pkg_name)
            all_checks_pass = False
    
    if missing_packages:
        print(f"\n⚠ Instale os pacotes faltantes com:")
        print(f"  pip install {' '.join(missing_packages)}")
    
    # 3. Verificar estrutura de diretórios
    print("\n[3] Verificando estrutura de diretórios...")
    dirs_to_check = [
        ('notebooks', False),
        ('src', False),
        ('data', True),
        ('models', True),
        ('mlruns', True),
        ('visualizations', True),
    ]
    
    for dir_name, create in dirs_to_check:
        if not check_directory(dir_name, create=create):
            if not create:
                all_checks_pass = False
    
    # 4. Verificar arquivos críticos
    print("\n[4] Verificando arquivos críticos...")
    files_to_check = [
        'README.md',
        'requirements.txt',
        'train_model.py',
        'test_model.py',
        'api_service.py',
        'src/__init__.py',
        'src/config.py',
        'src/utils.py',
        'src/models.py',
    ]
    
    missing_files = []
    for file_name in files_to_check:
        if not check_file(file_name):
            missing_files.append(file_name)
            all_checks_pass = False
    
    # 5. Verificar dados
    print("\n[5] Verificando dados...")
    data_file = 'data/bank-marketing.csv'
    if check_file(data_file):
        print("✓ Base Kaggle já baixada")
    else:
        print("⚠ Base Kaggle não encontrada")
        print("  Instruções: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing")
    
    # 6. Verificar modelos treinados
    print("\n[6] Verificando modelos treinados...")
    model_files = [
        'models/thompson_model.pkl',
        'models/scaler.pkl',
    ]
    
    models_exist = True
    for model_file in model_files:
        if not check_file(model_file):
            models_exist = False
    
    if not models_exist:
        print("⚠ Modelos ainda não foram treinados")
        print("  Próximo passo: python train_model.py")
    
    # Resumo final
    print("\n" + "=" * 80)
    print("RESUMO")
    print("=" * 80)
    
    if all_checks_pass:
        print("✓ Todas as verificações passaram!")
        print("\nVocê pode executar:")
        
        if models_exist:
            print("  1. python test_model.py         # Testar Golden Set")
            print("  2. python api_service.py        # Iniciar API")
            print("  3. mlflow ui                    # Visualizar experimentos")
        else:
            print("  1. python train_model.py        # Treinar modelo")
            print("  2. python test_model.py         # Testar Golden Set")
            print("  3. python api_service.py        # Iniciar API")
    else:
        print("✗ Algumas verificações falharam")
        print("  Consulte as mensagens acima")
        print("  Execute: pip install -r requirements.txt")
    
    print("\nConsulte QUICK_START.md para instruções detalhadas")
    print("=" * 80)
    
    return 0 if all_checks_pass else 1


if __name__ == "__main__":
    sys.exit(main())
