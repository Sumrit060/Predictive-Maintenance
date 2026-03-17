import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import detrend, windows

def calculate_vibration_math(filepath):
    """ฟังก์ชันคำนวณทางวิศวกรรมเพื่อหาค่า RMS (FFT Method)"""
    try:
        try:
            with open(filepath, 'r', encoding='utf-16') as f: content = f.read()
        except:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
        
        lines = content.splitlines()
        data_pairs = []
        for line in lines[9:]:
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(matches) >= 2:
                for i in range(0, len(matches), 2):
                    try: data_pairs.append((float(matches[i]), float(matches[i+1])))
                    except: continue

        if len(data_pairs) < 500: return 0.0

        data_pairs.sort(key=lambda x: x[0])
        times = np.array([p[0] for p in data_pairs])
        accel_g = np.array([p[1] for p in data_pairs])
        
        dt = np.mean(np.diff(times)) / 1000.0
        n = len(accel_g)
        
        accel_mm_s2 = detrend((accel_g - np.mean(accel_g)) * 9806.65)
        window = windows.hann(n)
        accel_fft = fft(accel_mm_s2 * window)
        freqs = fftfreq(n, d=dt)
        
        velocity_fft = np.zeros_like(accel_fft, dtype=complex)
        mask = (np.abs(freqs) >= 10.0) & (np.abs(freqs) <= 1000.0)
        omega = 2.0 * np.pi * freqs[mask]
        velocity_fft[mask] = accel_fft[mask] / (1j * omega)
        
        velocity_time = np.real(ifft(velocity_fft))
        velocity_final = velocity_time / (window.mean())
        return np.sqrt(np.mean(velocity_final**2))

    except Exception:
        return 0.0

# --- ส่วนจัดการไฟล์และสรุปผลตาม Group และ Filename ---

data_path = "./Data"
files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
results = []

print(f"\n{'เครื่องจักร':<25} | {'เดือน':<8} | {'RPM':<6} | {'Group':<7} | {'RMS':<6} | {'สถานะ'}")
print("-" * 105)

for file in sorted(files):
    full_path = os.path.join(data_path, file)
    rms = calculate_vibration_math(full_path)
    
    # 1. กำหนดชื่อเครื่องจากไฟล์
    if "CH-06" in file:
        equip_display = "Motor Compressor (CH-06)"
    elif "Cooling Pump" in file:
        equip_display = "Cooling Pump (OAH-02)"
    elif "Jockey" in file:
        equip_display = "Jockey Pump"
    else:
        equip_display = file.split('_')[1]

    # 2. ดึง RPM จากชื่อไฟล์
    rpm_match = re.search(r'_(\d{4})_', file)
    rpm = int(rpm_match.group(1)) if rpm_match else 0
    
    # 3. จัดกลุ่มเครื่องจักร (ISO 10816-3) อ้างอิงจาก RPM
    # ปั๊มรอบสูง (2925) มักเป็น Group 2&4, เครื่องจักรหลัก (1480-1490) มักเป็น Group 1&3
    if rpm > 2000:
        group_label = "2&4"
        limits = [1.4, 2.8, 4.5] # เกณฑ์ Rigid
    else:
        group_label = "1&3"
        limits = [2.3, 4.5, 7.1] # เกณฑ์ Rigid
    
    # 4. ดึงชื่อเดือนจากท้ายชื่อไฟล์
    month_match = re.search(r'_(Jun|Sep|Oct)24', file, re.IGNORECASE)
    month_label = month_match.group(1) if month_match else "N/A"
    
    # 5. ตัดสินสถานะ
    if rms > limits[2]: status = "🔴 Zone D (Damage)"
    elif rms > limits[1]: status = "🟡 Zone C (Restricted)"
    elif rms > limits[0]: status = "🟢 Zone B (Unrestricted)"
    else: status = "🔵 Zone A (New)"
        
    print(f"{equip_display[:25]:<25} | {month_label:<8} | {rpm:<6} | {group_label:<7} | {rms:<6.2f} | {status}")
    
    results.append({
        'Equipment': equip_display, 
        'Month': month_label, 
        'RMS': rms,
        'Group': group_label,
        'Order': {'Jun': 1, 'Sep': 2, 'Oct': 3}.get(month_label, 0)
    })

# --- พล็อตกราฟแนวโน้มแยกตามเครื่องและระบุ Group ---
df_plot = pd.DataFrame(results).sort_values(['Equipment', 'Order'])

plt.figure(figsize=(12, 6))
for name in df_plot['Equipment'].unique():
    subset = df_plot[df_plot['Equipment'] == name]
    grp = subset['Group'].iloc[0]
    plt.plot(subset['Month'], subset['RMS'], marker='o', linewidth=2.5, label=f"{name} (Grp {grp})")

plt.title("Vibration Trend Report by ISO Machinery Group", fontsize=14, fontweight='bold')
plt.ylabel("Velocity RMS (mm/s)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()