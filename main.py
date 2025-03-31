#%%
wd = "/media/sheshank/Work_Code/AI/datasets/"
import zipfile

zip_path = wd + "arxiv_scientific_research_papers_dataset.zip"  # Replace with the actual path to your zip file

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(wd)  # Extracts all the contents into the specified directory
    print("Contents of the zip file:")
    for name in zip_ref.namelist():
        if name.endswith(".csv"):
            temp_csv_file = wd + name  # Create a temporary CSV file path
            with open(temp_csv_file, 'w') as temp_file:
                temp_file.write(zip_ref.read(name).decode('utf-8'))  # Write the contents of the CSV file to the temporary file

#%%