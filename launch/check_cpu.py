import psutil

# Get CPU usage
cpu_usage = psutil.cpu_percent(interval=1)  # interval=1 means it will calculate CPU usage over a period of 1 second
print(f"{cpu_usage}")

# Get memory usage
memory_info = psutil.virtual_memory()
memory_usage = memory_info.percent
print(f"{memory_usage}")

# # Detailed memory information
# total_memory = memory_info.total / (1024 ** 3)  # Convert from bytes to GB
# used_memory = memory_info.used / (1024 ** 3)  # Convert from bytes to GB
# available_memory = memory_info.available / (1024 ** 3)  # Convert from bytes to GB

# print(f"Total Memory: {total_memory:.2f} GB")
# print(f"Used Memory: {used_memory:.2f} GB")
# print(f"Available Memory: {available_memory:.2f} GB")
