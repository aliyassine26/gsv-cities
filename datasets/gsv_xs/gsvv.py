"""A temporary script that has been used for generating dataframe from the GSV XS dataset cities."""
import os
import pandas as pd

os.chdir('datasets/gsv_xs')
root_dir = 'train/'
for class_name in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_name)
    if class_name == 'londonn':
        if os.path.isdir(class_path):
            csv_data = []
            for img_name in os.listdir(class_path):
                if img_name.endswith('.jpg'):
                    UTMx = img_name.split('@')[1]
                    UTMy = img_name.split('@')[2]
                    place_id = img_name.split('@')[-2].split('_')[0]
                    city = img_name.split('@')[-2].split('_')[1]
                csv_data.append(
                    [place_id, class_name, float(UTMx), float(UTMy),  img_name])
        df = pd.DataFrame(csv_data, columns=[
            'place_id', 'class_name', 'UTMx', 'UTMy', 'filename'])
        df.to_csv(os.path.join(
            root_dir, f'{class_name}.csv'), index=False)
