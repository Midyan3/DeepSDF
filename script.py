import os
import shutil
folder = "F:\\shapeOfMotorVehicles"

def createFolder(path): 
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder created: {path}")
def main(type, folderPath): 
    path = folderPath
    ext = ".obj"
    index = 0
    files = os.listdir(path)
    for folders in files:
        inIt = os.path.join(path, folders)
        if os.path.isdir(inIt) and os.path.isdir(os.path.join(inIt, "models")):
            model = os.path.join(inIt, "models")
            model_files = os.listdir(model)
            for file in model_files: 
                if file.endswith(ext):
                    file_path = os.path.join(model, file)
                    new_path = os.path.join(folder, type + str(index) + ext) 
                    shutil.copy(file_path, new_path)
                    print(new_path)
                    index += 1
                    
                 
createFolder(folder)
main(type = "vehicle", folderPath= "F:\\data\\ShapeNetCore.v2\\02958343\\02958343")