import shutil
import os
source_dirs = [r"C:\Users\tihan\Desktop\APIzza\MeshLab\Андрей_Неделя_1",
r"C:\Users\tihan\Desktop\APIzza\MeshLab\Андрей_Неделя_2",
r"C:\Users\tihan\Desktop\APIzza\MeshLab\Андрей_Неделя_3",
r"C:\Users\tihan\Desktop\APIzza\MeshLab\Андрей_Неделя_1",
r"C:\Users\tihan\Desktop\APIzza\MeshLab\Юния_Неделя_1",
r"C:\Users\tihan\Desktop\APIzza\MeshLab\Юния_Неделя_2",
r"C:\Users\tihan\Desktop\APIzza\MeshLab\Юния_Неделя_3",
r"C:\Users\tihan\Desktop\APIzza\MeshLab\Юния_Неделя_4",
r"C:\Users\tihan\Desktop\APIzza\MeshLab\Юния_Неделя_5",
r"C:\Users\tihan\Desktop\APIzza\MeshLab\Юния_Неделя_6",
r"C:\Users\tihan\Desktop\APIzza\MeshLab\Юрий_неделя_6",
r"C:\Users\tihan\Desktop\APIzza\MeshLab\Юрий_Неделя1-3"]


dest_dir = 'models'


for source_dir in source_dirs:
    # get a list of all files in the source directory
    files = os.listdir(source_dir)

    # iterate over the list of files
    for file_name in files:
        # construct a full file path
        source_file = os.path.join(source_dir, file_name)
        dest_file = os.path.join(dest_dir, file_name)

        shutil.copy2(source_file, dest_file)