#!/usr/bin/env python3
"""Setup script for MCP server data science environment on Windows."""

import json
import subprocess
import sys
import os
import re
import time
from pathlib import Path

def run_command(cmd, check=True, shell=True):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{cmd}': {e}")
        return None

def ask_permission(question):
    """Ask user for permission."""
    while True:
        response = input(f"{question} (y/n): ").lower()
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Please answer 'y' or 'n'")

def check_uv():
    """Check if uv is installed and install if needed."""
    if not run_command("where uv", check=False):
        if ask_permission("uv is not installed. Would you like to install it?"):
            print("Installing uv...")
            # Windows installation is different - using PowerShell
            ps_command = "powershell -Command \"irm https://astral.sh/uv/install.ps1 | iex\""
            run_command(ps_command)
            # Update PATH for current session
            os.environ["PATH"] = f"{os.environ['USERPROFILE']}\\.uv\\bin;{os.environ['PATH']}"
            print("uv installed successfully")
        else:
            sys.exit("uv is required to continue")

def setup_venv():
    """Create virtual environment if it doesn't exist."""
    if not Path(".venv").exists():
        if ask_permission("Virtual environment not found. Create one?"):
            print("Creating virtual environment...")
            run_command("uv venv")
            print("Virtual environment created successfully")
        else:
            sys.exit("Virtual environment is required to continue")

def sync_dependencies():
    """Sync project dependencies."""
    print("Syncing dependencies...")
    run_command("uv sync")
    print("Dependencies synced successfully")

def check_claude_desktop():
    """Check if Claude desktop app is installed on Windows."""
    # Common installation paths for Windows apps
    possible_paths = [
        os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), "Claude", "Claude.exe"),
        os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'), "Claude", "Claude.exe"),
        os.path.join(os.environ.get('LOCALAPPDATA', 'C:\\Users\\' + os.getenv('USERNAME') + '\\AppData\\Local'), "Programs", "Claude", "Claude.exe"),
        os.path.join(os.environ.get('LOCALAPPDATA', 'C:\\Users\\' + os.getenv('USERNAME') + '\\AppData\\Local'), "AnthropicClaude", "claude.exe")  
    ]
    
    claude_exists = any(os.path.exists(path) for path in possible_paths)
    
    if not claude_exists:
        print("Claude desktop app not found.")
        print("Please download and install from: https://claude.ai/download")
        if not ask_permission("Continue after installing Claude?"):
            sys.exit("Claude desktop app is required to continue")

def setup_claude_config():
    """Setup Claude desktop config file for Windows."""
    # Windows config location is different from macOS
    config_path = Path(os.environ.get('APPDATA', os.path.join(os.environ['USERPROFILE'], 'AppData\\Roaming'))) / "Claude" / "claude_desktop_config.json"
    config_dir = config_path.parent
    
    if not config_dir.exists():
        config_dir.mkdir(parents=True, exist_ok=True)
    
    config = {"mcpServers": {}} if not config_path.exists() else json.loads(config_path.read_text()) if config_path.exists() else {"mcpServers": {}}
    return config_path, config

def build_package():
    """Build package and get wheel path."""
    print("Building package...")
    try:
        # Use Popen for real-time and complete output capture
        process = subprocess.Popen(
            "uv build",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()  # Capture output
        output = stdout + stderr  # Combine both streams
        print(f"Raw output: {output}")  # Debug: check output
    except Exception as e:
        sys.exit(f"Error running build: {str(e)}")
    
    # Check if the command was successful
    if process.returncode != 0:
        sys.exit(f"Build failed with error code {process.returncode}")

    # Extract wheel file path from the combined output
    match = re.findall(r'dist\\[^\s]+\.whl', output.strip())
    if not match:
        # Try forward slashes too as some Windows commands output with forward slashes
        match = re.findall(r'dist/[^\s]+\.whl', output.strip())
    
    whl_file = match[-1] if match else None
    if not whl_file:
        sys.exit("Failed to find wheel file in build output")
    
    # Convert to absolute path
    path = Path(whl_file).absolute()
    return str(path)

def update_config(config_path, config, wheel_path):
    """Update Claude config with MCP server settings."""
    config.setdefault("mcpServers", {})
    config["mcpServers"]["mcp-server-ds"] = {
        "command": "uvx",
        "args": ["--from", wheel_path, "mcp-server-ds"]
    }
    
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Updated config at {config_path}")

def restart_claude():
    """Restart Claude desktop app if running on Windows."""
    # Check if Claude is running using Windows commands
    if run_command("tasklist /FI \"IMAGENAME eq claude.exe\" /NH | find \"claude.exe\"", check=False):
        if ask_permission("Claude is running. Restart it?"):
            print("Restarting Claude...")
            run_command("taskkill /F /IM claude.exe")
            time.sleep(2)
            # Start Claude - find the executable first
            possible_paths = [
                os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), "Claude", "Claude.exe"),
                os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'), "Claude", "Claude.exe"),
                os.path.join(os.environ.get('LOCALAPPDATA', 'C:\\Users\\' + os.getenv('USERNAME') + '\\AppData\\Local'), "Programs", "Claude", "Claude.exe"),
                os.path.join(os.environ.get('LOCALAPPDATA', 'C:\\Users\\' + os.getenv('USERNAME') + '\\AppData\\Local'), "AnthropicClaude", "claude.exe")  
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    run_command(f'start "" "{path}"', shell=True)
                    print("Claude restarted successfully")
                    return
                    
            print("Could not find Claude executable. Please start Claude manually.")
    else:
        print("Starting Claude...")
        # Same logic as above to find and start Claude
        possible_paths = [
            os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), "Claude", "Claude.exe"),
            os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'), "Claude", "Claude.exe"),
            os.path.join(os.environ.get('LOCALAPPDATA', 'C:\\Users\\' + os.getenv('USERNAME') + '\\AppData\\Local'), "Programs", "Claude", "Claude.exe"),
            os.path.join(os.environ.get('LOCALAPPDATA', 'C:\\Users\\' + os.getenv('USERNAME') + '\\AppData\\Local'), "AnthropicClaude", "claude.exe")  
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                run_command(f'start "" "{path}"', shell=True)
                print("Claude started successfully")
                return
                
        print("Could not find Claude executable. Please start Claude manually.")

def main():
    """Main setup function."""
    print("Starting setup...")
    check_uv()
    setup_venv()
    sync_dependencies()
    check_claude_desktop()
    config_path, config = setup_claude_config()
    wheel_path = build_package()
    update_config(config_path, config, wheel_path)
    restart_claude()
    print("Setup completed successfully!")

if __name__ == "__main__":
    main()