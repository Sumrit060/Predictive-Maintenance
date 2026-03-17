import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import detrend, windows

def calculate_vibration_final(filepath):
    """คำนวณ Velocity RMS ด้วยวิธี FFT + Windowing"""
    try:
        try:
            with open(filepath, 'r', encoding='utf-16') as f: content = f.read()
        except:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
        
        lines = content.splitlines()
        equipment = "Unknown"
        for line in lines:
            if "Equipment:" in line:
                equipment = re.sub(r'\(.*?\)', '', line.split("Equipment:")[1]).strip()
                break
        
        data_pairs = []
        for line in lines[9:]:
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(matches) >= 2:
                for i in range(0, len(matches), 2):
                    try: data_pairs.append((float(matches[i]), float(matches[i+1])))
                    except: continue

        if len(data_pairs) < 500: return equipment, 0.0

        data_pairs.sort(key=lambda x: x[0])
        times = np.array([p[0] for p in data_pairs])
        accel_g = np.array([p[1] for p in data_pairs])
        
        dt = np.mean(np.diff(times)) / 1000.0
        fs = 1.0 / dt
        n = len(accel_g)
        
        accel_mm_s2 = detrend((accel_g - np.mean(accel_g)) * 9806.65)
        window = windows.hann(n)
        accel_windowed = accel_mm_s2 * window
        
        accel_fft = fft(accel_windowed)
        freqs = fftfreq(n, d=dt)
        
        velocity_fft = np.zeros_like(accel_fft, dtype=complex)
        mask = (np.abs(freqs) >= 10.0) & (np.abs(freqs) <= 1000.0)
        
        omega = 2.0 * np.pi * freqs[mask]
        velocity_fft[mask] = accel_fft[mask] / (1j * omega)
        
        velocity_time = np.real(ifft(velocity_fft))
        velocity_final = velocity_time / (window.mean())
        
        rms_velocity = np.sqrt(np.mean(velocity_final**2))
        return equipment, rms_velocity

    except Exception:
        return "Unknown", 0.0

# --- ส่วนจัดการไฟล์และจัดกลุ่มตาม RPM ---

data_path = "./Data"
files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
results = []

print(f"\n{'เครื่องจักร':<25} | {'RPM':<6} | {'Group':<7} | {'RMS':<6} | {'สถานะ'}")
print("-" * 95)

for file in sorted(files):
    full_path = os.path.join(data_path, file)
    equip, rms = calculate_vibration_final(full_path)
    
    # 1. ดึงค่า RPM จากชื่อไฟล์ (ตัวเลข 4 หลัก เช่น 1480, 2925)
    rpm_match = re.search(r'_(\d{4})_+', file)
    rpm = int(rpm_match.group(1)) if rpm_match else 0
    
    # 2. จัดกลุ่มตาม RPM (เกณฑ์ Rigid Foundation ตามตาราง ISO)
    if rpm > 2000:
        group = "2&4"
        limits = [1.4, 2.8, 4.5] # Group 2 & 4 เกณฑ์เข้มงวดกว่า
    else:
        group = "1&3"
        limits = [2.3, 4.5, 7.1] # Group 1 & 3 เกณฑ์มาตรฐานเครื่องใหญ่
        
    # 3. ระบุเดือน
    month_match = re.search(r'[_]+(Jun|Sep|Oct)24', file, re.IGNORECASE)
    month = month_match.group(1).capitalize() if month_match else "N/A"
    
    # 4. ตัดสินสถานะตาม Group ของตัวเอง
    if rms > limits[2]: status = "🔴 Zone D (Damage)"
    elif rms > limits[1]: status = "🟡 Zone C (Restricted)"
    elif rms > limits[0]: status = "🟢 Zone B (Unrestricted)"
    else: status = "🔵 Zone A (New)"
        
    print(f"{equip[:25]:<25} | {rpm:<6} | {group:<7} | {rms:<6.2f} | {status}")
    results.append({'Equipment': equip, 'Month': month, 'RMS': rms, 'Group': group})

# --- พล็อตกราฟแยกตามกลุ่ม ---
df_plot = pd.DataFrame(results)
df_plot = df_plot[df_plot['Month'] != 'N/A']
month_map = {'Jun': 1, 'Sep': 2, 'Oct': 3}
df_plot['Month_Idx'] = df_plot['Month'].map(month_map)
df_plot = df_plot.sort_values(['Equipment', 'Month_Idx'])

plt.figure(figsize=(12, 7))
for name in df_plot['Equipment'].unique():
    subset = df_plot[df_plot['Equipment'] == name]
    # แสดงชื่อ Group ใน Label ด้วย
    grp = subset['Group'].iloc[0]
    plt.plot(subset['Month'], subset['RMS'], marker='o', linewidth=2.5, label=f"{name} (Grp {grp})")

plt.title("Vibration Trend Analysis with ISO Group Classification", fontsize=14, fontweight='bold')
plt.ylabel("Velocity RMS (mm/s)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()