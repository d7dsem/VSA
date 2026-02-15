
from typing import List, NamedTuple, Tuple


class Colors(NamedTuple):
    green: str = "\033[92m"
    bright_green: str = "\033[1;92m"
    red: str = "\033[91m"
    bright_red: str = "\033[1;91m"
    yellow: str = "\033[93m"
    bright_yellow: str = "\033[1;93m"
    gray: str = "\033[90m"
    bright_gray: str = "\033[1;90m"
    cyan: str = "\033[96m"
    bright_cyan: str = "\033[1;96m"
    blue: str = "\033[94m"
    bright_blue: str = "\033[1;94m"
    magenta: str = "\033[95m"
    bright_magenta: str = "\033[1;95m"
    white: str = "\033[97m"
    bright_white: str = "\033[1;97m"
    black: str = "\033[30m"
    bright_black: str = "\033[1;30m"
    reset: str = "\033[0m"

def color_to_rgb(color_name: str) -> Tuple[int, int, int]:
    color_map = {
        "black": (0, 0, 0),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "blue": (0, 0, 255),
        "magenta": (255, 0, 255),
        "cyan": (0, 255, 255),
        "white": (255, 255, 255)
    }
    return color_map[color_name.lower()]

def rgb_to_ansi(r: int, g: int, b: int, use_8bit: bool = False) -> str:
    if use_8bit:
        color_code = 16 + 36 * (r // 51) + 6 * (g // 51) + (b // 51)
        return f"\033[38;5;{color_code}m"
    else:
        # True Color (24-біт)
        return f"\033[38;2;{r};{g};{b}m"

def get_gradient_colors(gradient_len: int, use_8bit: bool = False, start_color: str = "yellow", end_color: str =  "magenta") -> List[str]:
    start_rgb = color_to_rgb(start_color)
    end_rgb = color_to_rgb(end_color)

    gradient = []
    for i in range(gradient_len):
        r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * i // (gradient_len - 1)
        g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * i // (gradient_len - 1)
        b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * i // (gradient_len - 1)
        
        # Приведення кольору до підтримуваного діапазону
        r = max(1, min(255, r))  # 1-255, щоб уникнути некоректних кольорів
        g = max(1, min(255, g))
        b = max(1, min(255, b))
        
        gradient.append(rgb_to_ansi(r, g, b, use_8bit))
    
    return gradient

COLORS = Colors()

def colorize(value):
    if isinstance(value, int):
        return f"{COLORS.green}{value}{COLORS.reset}"
    elif isinstance(value, float):
        return f"{COLORS.cyan}{value:.2f}{COLORS.reset}"
    elif isinstance(value, str):
        return f"{COLORS.yellow}{value}{COLORS.reset}"
    return str(value)

from enum import Enum

class _COLOR(str, Enum):
    GREEN = "\033[92m"
    BRIGHT_GREEN = "\033[1;92m"
    RED = "\033[91m"
    BRIGHT_RED = "\033[1;91m"
    YELLOW = "\033[93m"
    BRIGHT_YELLOW = "\033[1;93m"
    GRAY = "\033[90m"
    BRIGHT_GRAY = "\033[1;90m"
    CYAN = "\033[96m"
    BRIGHT_CYAN = "\033[1;96m"
    BLUE = "\033[94m"
    BRIGHT_BLUE = "\033[1;94m"
    MAGENTA = "\033[95m"
    BRIGHT_MAGENTA = "\033[1;95m"
    WHITE = "\033[97m"
    BRIGHT_WHITE = "\033[1;97m"
    BLACK = "\033[30m"
    BRIGHT_BLACK = "\033[1;30m"
    RESET = "\033[0m"

    WARN = BLUE + "[WARN]" + RESET
    ERR = RED + "[CRIT]" + RESET
    INFO = YELLOW + "[INFO]" + RESET
    DBG = GRAY + "[DeBG]" + RESET
    OK = GREEN + "OK" + RESET

def inject_colors_into(module_globals: dict) -> None:
    '''
    usage:
        # --- color names for IDE/static analysis suppress warnings ---
        GREEN: str; BRIGHT_GREEN: str
        RED: str; BRIGHT_RED: str
        YELLOW: str; BRIGHT_YELLOW: str
        GRAY: str; BRIGHT_GRAY: str
        CYAN: str; BRIGHT_CYAN: str
        BLUE: str; BRIGHT_BLUE: str
        MAGENTA: str; BRIGHT_MAGENTA: str
        WHITE: str; BRIGHT_WHITE: str
        BLACK: str; BRIGHT_BLACK: str
        RESET: str
        WARN: str
        ERR: str
        INFO: str
        DBG: str
        inject_colors_into(globals())
    '''
    # module_globals.update({k: v for k, v in _COLOR.__members__.items()})

    module_globals.update({k: v.value for k, v in _COLOR.__members__.items()})

    
