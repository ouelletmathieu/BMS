

def print_list_file(list_out, file):
    str_output = ""
    for elem in list_out:
        str_output += str(elem) + ","
    str_output+= "\n"

    file.writelines(str_output)