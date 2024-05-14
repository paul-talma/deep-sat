import os

folder_paths = [
    '/Users/paultalma/Programming/python_deep_sat/data/uf100-430',
    '/Users/paultalma/Programming/python_deep_sat/data/uuf100-430'
]

def clean_form(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line for line in lines if not line.startswith(('0', '%'))]
    with open(file_path, 'w') as file:
        file.writelines(lines)

for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            clean_form(file_path)