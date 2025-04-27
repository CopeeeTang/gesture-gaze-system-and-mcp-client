import math
import shutil


def log_title(message: str) -> None:
    """
    日志输出
    """
    
    terminal_width = shutil.get_terminal_size((80, 20)).columns # Default to 80 if size cant be determined
    message_length = len(message)
    padding = max(0, terminal_width - message_length - 2)
    left_padding = "=" * (padding // 2)
    right_padding = "=" * (padding - (padding // 2))
    print(f"{left_padding} {message} {right_padding}")