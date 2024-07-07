
from models.utils.model_func import process_color_codes

CONFIG_PATH = 'models/utils/config/config.json'
config = process_color_codes(CONFIG_PATH)
ERROR_COLOR = config.get('ERROR_COLOR')
SUCCESS_COLOR = config.get('SUCCESS_COLOR')
INFO_COLOR = config.get('INFO_COLOR')
WARNING_COLOR = config.get('WARNING_COLOR')
END_COLOR = config.get('END_COLOR')
DEFAULT_COLOR = config.get('DEFAULT_COLOR')


def get_msg(msg: str, hint: str = None):
    """Prepares logging message"""
    if hint == 'ERROR':
        return f"{ERROR_COLOR}{msg}{DEFAULT_COLOR}"
    elif hint == 'SUCCESS':
        return f"{SUCCESS_COLOR}{msg}{DEFAULT_COLOR}"
    elif hint == 'INFO':
        return f"{INFO_COLOR}{msg}{DEFAULT_COLOR}"
    elif hint == 'WARNING':
        return f"{WARNING_COLOR}{msg}{DEFAULT_COLOR}"
    elif hint == 'END':
        return f"{END_COLOR}{msg}{DEFAULT_COLOR}"
    elif hint == 'DEFAULT':
        return f"{DEFAULT_COLOR}{msg}{DEFAULT_COLOR}"
    else:
        return f"{msg}"
