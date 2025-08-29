"""
Setup script for downloading data and creating required directories
"""
import os
import subprocess
import gdown

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def download_data():
    """Download data using gdown"""
    print("Downloading data from Google Drive...")
    try:
        # Download the folder (update this ID with your actual Google Drive folder ID)
        gdown.download_folder('18_pZMpxI2nwBn6kfz60OSFlXDWw5Pz29', output='data/', quiet=False)
        print("Data downloaded successfully!")
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please manually download your data files to the 'data/' directory")

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")

def main():
    print("Setting up the project...")
    create_directories()
    install_requirements()
    download_data()
    print("\nSetup complete! You can now run: python main.py")

if __name__ == "__main__":
    main()
