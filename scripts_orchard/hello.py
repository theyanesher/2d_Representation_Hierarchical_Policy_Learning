import time

for i in range(24):
    time.sleep(3600)
print("hello world")

#SBATCH --exclude=orchard-flame-15,orchard-flame-25