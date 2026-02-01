# How to Build for Windows

Since you are currently on a Mac, you cannot directly generate a Windows `.exe` file here. However, I have prepared all the necessary configuration files for you to easily build it on a Windows machine.

## Steps:

1.  **Copy Files**: Copy the entire `mp4_converter` folder to a Windows computer.
2.  **Install Python**: Ensure Python (3.8 or newer) is installed on the Windows machine.
3.  **Run Build Script**:
    *   Open the folder in File Explorer.
    *   Double-click `build_windows.bat`.
4.  **Find Executable**:
    *   Once the script finishes, a new folder named `dist` will appear.
    *   Inside `dist`, you will find `ArtNetPixelConsole.exe`.
    *   You can run this `.exe` directly.

## Files Created:

*   `requirements.txt`: Lists all Python libraries needed.
*   `build.spec`: Configuration for PyInstaller (tells it to include the `templates` folder and necessary imports).
*   `build_windows.bat`: A simple script to install dependencies and run the build command automatically.
