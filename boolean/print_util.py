import os

def create_text_file(output_file, header, delete=False):
    if delete:
        if os.path.exists(output_file):
            os.remove(output_file)
        file = open(output_file, 'a') 
        file.writelines(header)
        file.flush()
        return file
        
    else:
        file = open(output_file, 'a') 
        if os.stat(output_file).st_size < 10:
            file.writelines(header)
            file.flush()
        return file

def print_list_file(list_out, file):
    str_output = ""
    for elem in list_out:
        str_output += str(elem) + ","
    str_output+= "\n"

    file.writelines(str_output)