import os
import subprocess

def run_command(command, cwd=None):
    """Helper to run shell commands and handle errors nicely"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, cwd=cwd, shell=True, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.cmd}")
        print(f"Exit code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        raise  # re-raise to stop script if needed

def setup_git_and_branches(project_path):
    os.chdir(project_path)

    # Initialize git repo
    run_command("git init")

    # Create initial commit
    with open("README.md", "w") as f:
        f.write("# My Project\n")
    run_command("git add .")
    run_command('git commit -m "Initial commit on main"')

    # Create branches
    run_command("git branch task-1")
    run_command("git branch task-2")

    print("Branches created: main, task-1, task-2")

def create_github_actions_workflow(project_path):
    workflow_dir = os.path.join(project_path, ".github", "workflows")
    os.makedirs(workflow_dir, exist_ok=True)

    ci_content = """
name: CI

on:
  push:
    branches:
      - main
      - task-1
      - task-2
  pull_request:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.x'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

    - name: Run tests
      run: |
        echo "Run your tests here"
        # e.g., pytest tests/
"""
    workflow_file = os.path.join(workflow_dir, "ci.yml")
    with open(workflow_file, "w") as f:
        f.write(ci_content.strip())

    print("✅ GitHub Actions workflow created at .github/workflows/ci.yml")

def main():
    project_name = input("Enter your existing project folder name: ").strip()
    if not project_name:
        print("Project name cannot be empty.")
        return

    project_path = os.path.abspath(project_name)

    if not os.path.isdir(project_path):
        print(f"Folder '{project_path}' does not exist.")
        return

    try:
        setup_git_and_branches(project_path)
        create_github_actions_workflow(project_path)
        print("\n✅ CI/CD setup completed! Don't forget to push your repo to GitHub.")
    except Exception as e:
        print(f"❌ Script failed: {e}")

if __name__ == "__main__":
    main()
