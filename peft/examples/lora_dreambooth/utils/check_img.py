from PIL import Image
import os

def check_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # 尝试打开图片文件
                img = Image.open(file_path)
                img.verify()  # 验证图片文件是否损坏
            except (IOError, SyntaxError) as e:
                # 捕获损坏的图片异常
                print(f"Image file {file_path} is corrupted.")
                # 如果需要，你也可以将损坏的图片文件进行备份或删除等操作
                # os.remove(file_path)  # 删除损坏的图片文件
                # 或者将损坏的图片文件移动到另一个目录
                # os.rename(file_path, os.path.join(corrupted_images_dir, file))  
            except Exception as e:
                print(f"An error occurred while processing image {file_path}: {str(e)}")

# 指定数据集目录
data_directory = "/root/shiym_proj/DiffLook/outputs"

# 检查图片文件
check_images(data_directory)
