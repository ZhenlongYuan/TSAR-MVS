import os

def clean_subfolders_and_keep_ply(parent_folder, files_to_keep):
    """
    Keeps specific files in subfolders and .ply files in the main folder.
    
    Args:
    - parent_folder (str): The parent folder where the 'sss' folder is located.
    - files_to_keep (list): Filenames to keep in each subfolder of 'sss'.
    """
    sss_path = os.path.join(parent_folder, 'APD')
    
    if not os.path.exists(sss_path):
        print(f"The folder {sss_path} does not exist.")
        return
    
    # Keeping .ply files in the 'sss' folder but not in its subfolders
    for item in os.listdir(sss_path):
        item_path = os.path.join(sss_path, item)
        if os.path.isdir(item_path):
            # If it's a subfolder, keep only specific files
            for subdir, _, files in os.walk(item_path):
                for file in files:
                    if file not in files_to_keep:
                        os.remove(os.path.join(subdir, file))
                        print(f"Deleted {os.path.join(subdir, file)}")
        elif not item.endswith('.ply'):
            # If it's not a .ply file in the 'sss' folder, remove it
            os.remove(item_path)
            print(f"Deleted {item_path}")

# Adjust this variable to your specific folder structure
parent_folder = '.'  # Assuming the current directory contains the 'sss' folder
files_to_keep = [
    "depth_15.jpg", "depths_geom.dmb", "normal_15.jpg", "weak.png", "weak.bin", 
    "depths.dmb", "TSAR_disp.dmb", "TSAR_normals.dmb", "TSAR_normals.png", "normals.dmb"
]

clean_subfolders_and_keep_ply(parent_folder, files_to_keep)
