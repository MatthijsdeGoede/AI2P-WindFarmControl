import time

last_call = time.time()

def start_timer():
    global last_call
    last_call = time.time()

def print_timer(print_str):
    global last_call
    dt = time.time() - last_call
    print(f"{print_str}: {dt}")
    last_call = time.time()