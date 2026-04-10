                            
                                                            
                                                                                      
                        
 
 
 
                                    
                                  
                 
 
                                       
                     
                       
                                                                   
                                                                                               
                              
                 


def is_letter(char):
    if len(char) == 1:
        char_code = ord(char)
        return (65 <= char_code <= 90) or (97 <= char_code <= 122)
    return False

def get_person_name(file_name):
    name1=file_name.split('.')[0]
    name=''
    for c in name1:
        if is_letter(c):
            name=name+c
    return name

def find_value_from_xlsx(feat_name:str,excel_data,name:str):
    for idx in range(excel_data.shape[0]):
        if excel_data['name'][idx]==name:
            return excel_data[feat_name][idx]

    print('not found')
    print(name)
    exit(1)