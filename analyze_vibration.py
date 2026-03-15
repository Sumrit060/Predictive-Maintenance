import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import detrend, windows

def calculate_vibration_final(filepath):
    """คำนวณ Velocity RMS ด้วยวิธี FFT + Windowing (มาตรฐานสูงสุด)"""
    try:
        # 1. อ่านไฟล์ด้วยการรองรับทุก Encoding
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
        
        # 2. ดึงข้อมูลและจัดเรียงตามเวลา
        data_pairs = []
        for line in lines[9:]:
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(matches) >= 2:
                for i in range(0, len(matches), 2):
                    try: data_pairs.append((float(matches[i]), float(matches[i+1])))
                    except: continue

        if len(data_pairs) < 500: return equipment, 0.0 # ข้อมูลน้อยไปคำนวณ FFT ไม่ได้

        data_pairs.sort(key=lambda x: x[0])
        times = np.array([p[0] for p in data_pairs])
        accel_g = np.array([p[1] for p in data_pairs])
        
        # 3. คำนวณความถี่พื้นฐาน
        dt = np.mean(np.diff(times)) / 1000.0
        fs = 1.0 / dt
        n = len(accel_g)
        
        # --- Advanced Signal Processing ---
        # A. ลบค่าเฉลี่ยและ Detrend เพื่อขจัด DC Offset
        accel_mm_s2 = detrend((accel_g - np.mean(accel_g)) * 9806.65)
        
        # B. ใส่ Window (Hanning) เพื่อลดการรั่วไหลของสัญญาณที่ขอบ
        window = windows.hann(n)
        accel_windowed = accel_mm_s2 * window
        
        # C. ทำ FFT
        accel_fft = fft(accel_windowed)
        freqs = fftfreq(n, d=dt)
        
        # D. FFT Integration (V = A / jw) 
        # บังคับ Bandpass 10Hz - 1000Hz ตาม ISO 10816-3
        velocity_fft = np.zeros_like(accel_fft, dtype=complex)
        mask = (np.abs(freqs) >= 10.0) & (np.abs(freqs) <= 1000.0)
        
        omega = 2.0 * np.pi * freqs[mask]
        velocity_fft[mask] = accel_fft[mask] / (1j * omega)
        
        # E. แปลงกลับและชดเชยค่าความแรงสัญญาณจาก Window (Window Gain Compensation)
        velocity_time = np.real(ifft(velocity_fft))
        velocity_final = velocity_time / (window.mean()) # แก้ไขค่าที่ลดลงจาก Hanning
        
        # 4. คำนวณ RMS
        rms_velocity = np.sqrt(np.mean(velocity_final**2))
        return equipment, rms_velocity

    except Exception as e:
        return "Unknown", 0.0

#สรุปผล
data_path = "./Data"
files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
results = []

print(f"\n{'เครื่องจักร':<30} | {'เดือน':<8} | {'RMS (mm/s)':<12} | {'สถานะ'}")
print("-" * 85)

for file in sorted(files):
    full_path = os.path.join(data_path, file)
    equip, rms = calculate_vibration_final(full_path)
    month_match = re.search(r'[_]+(Jun|Sep|Oct)24', file, re.IGNORECASE)
    month = month_match.group(1).capitalize() if month_match else "N/A"
    
    if rms > 7.1: status = "🔴 Zone D (Damage)"
    elif rms > 4.5: status = "🟡 Zone C (Restricted)"
    elif rms > 2.3: status = "🟢 Zone B (Unrestricted)"
    else: status = "🔵 Zone A (New)"
        
    print(f"{equip[:30]:<30} | {month:<8} | {rms:<12.2f} | {status}")
    results.append({'Equipment': equip, 'Month': month, 'RMS': rms})

#Plotกราฟ
df_plot = pd.DataFrame(results)
df_plot = df_plot[df_plot['Month'] != 'N/A']
month_map = {'Jun': 1, 'Sep': 2, 'Oct': 3}
df_plot['Month_Idx'] = df_plot['Month'].map(month_map)
df_plot = df_plot.sort_values(['Equipment', 'Month_Idx'])

plt.figure(figsize=(12, 6))
for name in df_plot['Equipment'].unique():
    subset = df_plot[df_plot['Equipment'] == name]
    plt.plot(subset['Month'], subset['RMS'], marker='o', linewidth=2.5, label=name)

plt.axhline(y=7.1, color='#e74c3c', linestyle='--', label='Limit: Damage (7.1)')
plt.axhline(y=4.5, color='#f39c12', linestyle='--', label='Limit: Restricted (4.5)')
plt.axhline(y=2.3, color='#27ae60', linestyle='--', label='Limit: Normal (2.3)')

plt.title("Predictive Maintenance: Vibration Health Trend (FFT Method)", fontsize=14, fontweight='bold')
plt.ylabel("Velocity RMS (mm/s)", fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()