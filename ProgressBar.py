import sys

progress_step = 0   
progress_counter = 0

def initializeProgressBar(size):
    global progress_step
    progress_step = 100/size
    sys.stdout.write("\n")
    for i in range(100):
        sys.stdout.write("-")
    sys.stdout.write("\n")
    sys.stdout.flush()


def increaseProgressBar():
    global progress_counter
    progress_counter += progress_step
    if progress_counter >= 1:
        num_steps = int(progress_counter)
        for i in range(num_steps):
            sys.stdout.write('|')
        sys.stdout.flush()
        progress_counter -= num_steps


def completeProgressBar():
    global progress_counter
    global progress_step
    sys.stdout.write("\n")
    for i in range(100):
        sys.stdout.write("-")
    sys.stdout.write("\n\n")
    sys.stdout.flush()
    progress_counter = 0
    progress_step = 0