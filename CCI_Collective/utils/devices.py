import cupy as cp

def get_available_devices():
    num_devices = cp.cuda.runtime.getDeviceCount()
    devices = []
    for device_id in range(num_devices):
        device_properties = cp.cuda.runtime.getDeviceProperties(device_id)
        devices.append({
            "Device ID": device_id,
            "Name": device_properties["name"],
            "Total Memory (MB)": device_properties["totalGlobalMem"] / (1024 ** 2),
            "Multiprocessors": device_properties["multiProcessorCount"],
            "Compute Capability": f"{device_properties['major']}.{device_properties['minor']}"
        })
    return devices