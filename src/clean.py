import re


def complex_clean(text):
    try:
        clean = re.sub("\\'",'',text).strip()
        clean = re.sub("[^\x20-\x7E]", "",clean).strip()
        clean = re.sub("[0-9/-]+ to [0-9/-]+ percent","percenttarget ",clean)
        clean = re.sub("[0-9/-]+ percent","percenttarget ",clean)
        clean = re.sub("[0-9]+.[0-9]+ percent","dpercent",clean)
        clean = re.sub(r"[0-9]+","dd",clean)
        clean = re.sub("U.S.","US",clean).strip()
        clean = re.sub("p.m.","pm",clean).strip()
        clean = re.sub("a.m.","am",clean).strip()
        clean = re.sub("S&P","SP",clean).strip()
        clean = re.sub(r'(?<!\d)\.(?!\d)'," ",clean).strip()
        clean = re.sub(r"""
                   [,;@#?!&$"]+  # Accept one or more copies of punctuation
                   \ *           # plus zero or more copies of a space
                   """,
                   " ",          # and replace it with a single space
                    clean, flags=re.VERBOSE)
        clean = re.sub('--', ' ', clean).strip()  
        clean = re.sub("'",' ',clean).strip()
        clean = re.sub("- ","-",clean).strip()
        clean = re.sub('\(A\)', ' ', clean).strip()
        clean = re.sub('\(B\)', ' ', clean).strip()
        clean = re.sub('\(C\)', ' ', clean).strip()
        clean = re.sub('\(D\)', ' ', clean).strip()
        clean = re.sub('\(E\)', ' ', clean).strip()
        clean = re.sub('\(i\)', ' ', clean).strip()
        clean = re.sub('\(ii\)', ' ', clean).strip()
        clean = re.sub('\(iii\)', ' ', clean).strip()
        clean = re.sub('\(iv\)', ' ', clean).strip()
        clean = re.sub('/^\\:/',' ',clean).strip()
        clean = re.sub(r"FRB: .*Minutes of", "Minutes of", clean)

        clean=re.sub('\s+', ' ',clean).strip()
    except:
        print("Unable to clean file %s" % files)
        return " "
    return clean


def simple_clean(text):
    # *** see attribution below *****
    # this code is taken from the following web-site
    # https://medium.com/data-from-the-trenches/text-classification-the-first-step-toward-nlp-mastery-f5f95d525d73

    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """
    
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)    
    
    # convert text to lowercase
    text = text.strip().lower()

    
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text
