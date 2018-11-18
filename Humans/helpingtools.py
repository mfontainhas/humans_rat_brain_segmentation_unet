import sys


def update_progress(job_title, progress):  # Create a Loading Bar
    length = 30  # modify this to change the length
    block = int(round(length * progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, ">" * block + "-" * (length - block), round(progress * 100, 2))
    sys.stdout.write(msg)
    sys.stdout.flush()
    if progress >= 1:
        msg += " DONE\r\n"
        sys.stdout.write(msg)
        sys.stdout.flush()
