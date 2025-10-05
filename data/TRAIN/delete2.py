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
import os
import glob
import sys

def delete_files_in_subfolders(root_folder):
    # 遍历根文件夹下的所有子文件夹
    for subfolder in sorted(glob.glob(os.path.join(root_folder, '*'))):
        if os.path.isdir(subfolder):
            print(f"处理文件夹: {subfolder}")
            
            # 删除 depth_*.jpg 文件 (1-14)
            for i in range(1, 15):
                file_pattern = os.path.join(subfolder, f"depth_{i}.jpg")
                for file in glob.glob(file_pattern):
                    try:
                        os.remove(file)
                        print(f"已删除: {file}")
                    except OSError as e:
                        print(f"无法删除 {file}: {e}")
            
            # 删除 normal_*.jpg 文件 (1-14)
            for i in range(1, 15):
                file_pattern = os.path.join(subfolder, f"normal_{i}.jpg")
                for file in glob.glob(file_pattern):
                    try:
                        os.remove(file)
                        print(f"已删除: {file}")
                    except OSError as e:
                        print(f"无法删除 {file}: {e}")
            
            # 删除 weak_*.jpg 文件 (1-14)
            for i in range(1, 15):
                file_pattern = os.path.join(subfolder, f"weak_{i}.jpg")
                for file in glob.glob(file_pattern):
                    try:
                        os.remove(file)
                        print(f"已删除: {file}")
                    except OSError as e:
                        print(f"无法删除 {file}: {e}")
            
            # 删除 rawedge_*.jpg 文件 (0-3)
            for i in range(0, 4):
                file_pattern = os.path.join(subfolder, f"rawedge_{i}.jpg")
                for file in glob.glob(file_pattern):
                    try:
                        os.remove(file)
                        print(f"已删除: {file}")
                    except OSError as e:
                        print(f"无法删除 {file}: {e}")
            
            # 删除 edges_*.dmb 文件 (0-3)
            for i in range(0, 4):
                file_pattern = os.path.join(subfolder, f"edges_{i}.dmb")
                for file in glob.glob(file_pattern):
                    try:
                        os.remove(file)
                        print(f"已删除: {file}")
                    except OSError as e:
                        print(f"无法删除 {file}: {e}")
            
            # 删除 labels_*.dmb 文件 (0-3)
            for i in range(0, 4):
                file_pattern = os.path.join(subfolder, f"labels_{i}.dmb")
                for file in glob.glob(file_pattern):
                    try:
                        os.remove(file)
                        print(f"已删除: {file}")
                    except OSError as e:
                        print(f"无法删除 {file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请指定 FACADE 文件夹路径作为参数")
        print("用法: python script.py <FACADE文件夹路径>")
        sys.exit(1)
    
    facade_path = sys.argv[1]
    apd_path = os.path.join(facade_path, "APD")
    
    if not os.path.exists(apd_path):
        print(f"错误: APD 文件夹不存在于 {facade_path}")
        sys.exit(1)
    
    print(f"开始在根文件夹 {apd_path} 中删除文件...")
    delete_files_in_subfolders(apd_path)
    print("文件删除操作完成。")