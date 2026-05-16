import zipfile
import os

def zip_folders(folders, files, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Zip folders
        for folder in folders:
            for root, _, filenames in os.walk(folder):
                for filename in filenames:
                    abs_path = os.path.join(root, filename)
                    # Use forward slashes for Linux compatibility
                    rel_path = os.path.relpath(abs_path, os.getcwd()).replace('\\', '/')
                    zf.write(abs_path, rel_path)
        
        # Zip individual files
        for file in files:
            if os.path.exists(file):
                zf.write(file, file.replace('\\', '/'))

if __name__ == "__main__":
    folders_to_zip = ['src', 'configs', 'scripts']
    files_to_zip = ['requirements.txt']
    output = 'deepfake_project_kaggle_v2.zip'
    
    print(f"Creating {output}...")
    zip_folders(folders_to_zip, files_to_zip, output)
    print("Done!")
