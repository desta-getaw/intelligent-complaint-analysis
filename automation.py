import os

def create_python_project_structure(project_name):
    """
    Creates a standard Python project structure:
    project_name/
        notebooks/
        scripts/
        src/
        tests/
        data/
        .gitignore
        requirements.txt
    """
    # List of subdirectories to create
    subdirs = ['notebooks', 'scripts', 'src', 'tests', 'data']

    try:
        # Create the main project folder
        os.makedirs(project_name, exist_ok=True)
        
        # Create subdirectories
        for subdir in subdirs:
            os.makedirs(os.path.join(project_name, subdir), exist_ok=True)
        
        # Create an empty .gitignore file with some common Python ignores
        gitignore_content = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# Environments
.env
.venv
env/
venv/
ENV/

# VS Code
.vscode/

# Mac
.DS_Store
"""
        with open(os.path.join(project_name, '.gitignore'), 'w') as f:
            f.write(gitignore_content.strip())

        # Create an empty requirements.txt file
        open(os.path.join(project_name, 'requirements.txt'), 'a').close()

        print(f"Project '{project_name}' structure created successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    project_name = input("Enter your project name: ").strip()
    if project_name:
        create_python_project_structure(project_name)
    else:
        print("Project name cannot be empty.")
