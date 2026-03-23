import subprocess
import os
import sys
import threading

def run_backend(root):
    backend_dir = os.path.join(root, "app", "backend")
    print("Starting backend on port 8005...")
    venv_python = os.path.join(backend_dir, "venv", "bin", "python")
    if os.path.exists(venv_python):
        subprocess.run([venv_python, "-m", "uvicorn", "main:app", "--reload", "--port", "8005"], cwd=backend_dir)
    else:
        subprocess.run(["uvicorn", "main:app", "--reload", "--port", "8005"], cwd=backend_dir)

def run_frontend(root):
    import shutil
    import pathlib

    if not shutil.which("npm"):
        print("\n❌ CRITICAL ERROR: Node.js (npm) is missing from your system!")
        print("Required Action: Please install Node.js from https://nodejs.org/ to serve the frontend UI.")
        print("Exiting...\n")
        os._exit(1)

    source_frontend_dir = os.path.join(root, "app", "frontend")
    
    # Establish a user-writable local directory for the frontend cache/build
    user_home = str(pathlib.Path.home())
    local_frontend_dir = os.path.join(user_home, ".attention-encoders-ui")
    
    # Copy source frontend files to the writable home directory
    if not os.path.exists(local_frontend_dir):
        print(f"\n📦 Initializing frontend build directory at {local_frontend_dir}...")
        shutil.copytree(source_frontend_dir, local_frontend_dir, ignore=shutil.ignore_patterns("node_modules", ".next", "__pycache__"))
        
    node_modules_dir = os.path.join(local_frontend_dir, "node_modules")
    if not os.path.exists(node_modules_dir):
        print("\n🔧 First-time setup detected: Installing frontend dependencies (this may take a minute)...")
        try:
            subprocess.run(["npm", "install"], cwd=local_frontend_dir, check=True)
        except subprocess.CalledProcessError:
            print("\n❌ Failed to install frontend dependencies. Please ensure npm has write privileges.")
            os._exit(1)
            
    print("Starting frontend on port 3000...")
    subprocess.run(["npm", "run", "dev"], cwd=local_frontend_dir)

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    
    t1 = threading.Thread(target=run_backend, args=(root,))
    t2 = threading.Thread(target=run_frontend, args=(root,))
    
    t1.daemon = True
    t2.daemon = True
    
    t1.start()
    t2.start()
    
    print("\n✅ Servers are booting up!")
    print("👉 Frontend will be at: http://localhost:3000")
    print("👉 Backend API will be at: http://localhost:8005")
    print("Press Ctrl+C to exit.\n")
    
    try:
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()
