from typing import Set

NORMALIZE_DICT = {"/.": ".", "/?": "?",
                  "-LRB-": "(", "-RRB-": ")",
                  "-LCB-": "{", "-RCB-": "}",
                  "-LSB-": "[", "-RSB-": "]"}
REMOVED_CHAR = ["/", "%", "*"]

def conll_clean_token(token: str) -> str:
    """ Substitute in /?(){}[] for equivalent CoNLL-2012 representations,
    remove /%* 
    Args:
        token (str): token to clean
    Returns:
        str: cleaned token
    """
    cleaned_token = token
    if cleaned_token in NORMALIZE_DICT:
        cleaned_token = NORMALIZE_DICT[cleaned_token]

    if cleaned_token not in REMOVED_CHAR:
        for char in REMOVED_CHAR:
            cleaned_token = cleaned_token.replace(char, u'')

    if len(cleaned_token) == 0:
        cleaned_token = ","
    return cleaned_token

def read_file(filename:str) -> Set[str]:
    """Read file and return  string object
        Args:
            filename (str): path to the file
        Returns:
            Set[str]: set of tokens in the file
    """
    with open(filename, 'r') as f:
        return set(f.read().split('\n'))