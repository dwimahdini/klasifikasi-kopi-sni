import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# 1. PENGATURAN HALAMAN & SIDEBAR
st.set_page_config(page_title="Klasifikasi Mutu Kopi SNI", layout="wide")
st.title("☕ Sistem Klasifikasi Mutu Biji Kopi (SNI 01-2907-2008)")

st.sidebar.header("⚙️ Pengaturan Kalibrasi Ukuran")
st.sidebar.write("Sesuaikan rasio ini agar ukuran Milimeter (mm) akurat sesuai jarak kamera Anda.")
# Slider untuk mengkonversi piksel ke mm
px_per_mm = st.sidebar.slider("Piksel per Milimeter (px/mm):", min_value=1.0, max_value=50.0, value=15.0, step=0.5)

# 2. MEMUAT MODEL AI
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 3. KAMUS BOBOT SNI LENGKAP
info_sni = {
    'black': {'nama': 'Biji Hitam', 'pembagi': 1, 'bobot': 1.0},        # 1 biji hitam = 1 nilai
    'broken': {'nama': 'Biji Pecah', 'pembagi': 5, 'bobot': 0.2},       # 5 biji pecah = 1 nilai
    'foreign': {'nama': 'Benda Asing', 'pembagi': 1, 'bobot': 1.0},     # 1 benda asing = 1 nilai
    'fraghusk': {'nama': 'Pecahan Kulit', 'pembagi': 1, 'bobot': 1.0},  
    'husk': {'nama': 'Kulit Kopi Utuh', 'pembagi': 1, 'bobot': 1.0},    # 1 kulit = 1 nilai
    'immature': {'nama': 'Biji Muda', 'pembagi': 5, 'bobot': 0.2},      # 5 biji muda = 1 nilai
    'infested': {'nama': 'Biji Berlubang', 'pembagi': 10, 'bobot': 0.1},# 10 biji berlubang = 1 nilai
    'sour': {'nama': 'Biji Asam / Busuk', 'pembagi': 2, 'bobot': 0.5},  # 2 biji asam = 1 nilai
    'green': {'nama': 'Biji Sehat (Normal)', 'pembagi': 0, 'bobot': 0.0}
}

# Fungsi Pembaca Warna
def tebak_warna(r, g, b):
    if r > 150 and g > 100 and b < 100: return "Kuning Kecokelatan"
    elif r < 80 and g < 80 and b < 80: return "Hitam / Gelap"
    elif r > 100 and g > 80 and b > 50: return "Cokelat Terang"
    elif g > r and g > b: return "Kehijauan"
    else: return "Cokelat Gelap"

# 4. INPUT GAMBAR
pilihan = st.radio("Pilih sumber gambar:", ("Unggah File", "Gunakan Kamera"), horizontal=True)
gambar_masuk = st.file_uploader("Pilih gambar biji kopi...", type=["jpg", "jpeg", "png"]) if pilihan == "Unggah File" else st.camera_input("Ambil foto biji kopi")

# 5. PROSES UTAMA
if gambar_masuk is not None:
    image_pil = Image.open(gambar_masuk).convert("RGB")
    image_np = np.array(image_pil)
    
    with st.spinner('AI sedang menganalisis secara mendetail...'):
        hasil = model(image_pil)
        gambar_hasil = hasil[0].plot()
        
        # Variabel Penyimpanan Data
        rekap_cacat = {k: 0 for k in info_sni.keys()}
        data_biji = []
        
        # Loop Analisis Per Biji Kopi
        for i, box in enumerate(hasil[0].boxes):
            nama_kelas = model.names[int(box.cls[0])]
            akurasi = float(box.conf[0]) * 100
            
            # Hitung jumlah jenis cacat yang sama
            if nama_kelas in rekap_cacat:
                rekap_cacat[nama_kelas] += 1
            
            # Ekstraksi Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            potongan_biji = image_np[y1:y2, x1:x2]
            
            # Perhitungan Diameter (px & mm)
            tinggi_px, lebar_px, _ = potongan_biji.shape
            diameter_px = max(tinggi_px, lebar_px)
            diameter_mm = diameter_px / px_per_mm # Dihitung dengan slider kalibrasi
            
            # Perhitungan Warna RGB
            if potongan_biji.size > 0:
                rata_warna = potongan_biji.mean(axis=0).mean(axis=0)
                r, g, b = int(rata_warna[0]), int(rata_warna[1]), int(rata_warna[2])
                nama_warna = tebak_warna(r, g, b)
                rgb_text = f"RGB({r},{g},{b})"
            else:
                nama_warna, rgb_text = "N/A", "N/A"
                
            data_biji.append({
                "ID": f"Biji-{i+1}",
                "Status": info_sni.get(nama_kelas, {}).get('nama', nama_kelas).upper(),
                "Warna Rata-rata": nama_warna,
                "Nilai RGB": rgb_text,
                "Diameter (Piksel)": diameter_px,
                "Estimasi Ukuran (mm)": f"{diameter_mm:.1f} mm",
                "Kepercayaan AI": f"{akurasi:.1f}%"
            })

        # Menghitung Rumus Mutu SNI
        tabel_rekap = []
        total_poin_cacat = 0.0
        total_biji = len(hasil[0].boxes)
        total_sehat = rekap_cacat['green']
        total_rusak = total_biji - total_sehat

        for kode, jumlah in rekap_cacat.items():
            if jumlah > 0 and kode != 'green':
                aturan = info_sni[kode]
                poin = jumlah * aturan['bobot']
                total_poin_cacat += poin
                tabel_rekap.append({
                    "Jenis Cacat": aturan['nama'],
                    "Jumlah Ditemukan": jumlah,
                    "Aturan SNI": f"{aturan['pembagi']} biji = 1 poin",
                    "Total Poin": round(poin, 2)
                })

        # Logika Keputusan Grade
        grade, warna_grade = "TIDAK MASUK MUTU", "red"
        if total_poin_cacat <= 11: grade, warna_grade = "MUTU 1 (GRADE 1)", "green"
        elif total_poin_cacat <= 25: grade, warna_grade = "MUTU 2 (GRADE 2)", "blue"
        elif total_poin_cacat <= 44: grade, warna_grade = "MUTU 3 (GRADE 3)", "orange"
        elif total_poin_cacat <= 80: grade, warna_grade = "MUTU 4 (GRADE 4)", "red"

        # 6. TAMPILAN DASHBOARD HASIL
        st.markdown("---")
        col_img, col_res = st.columns([1.2, 1])
        
        with col_img:
            st.image(gambar_hasil, caption='Hasil Pemindaian AI (YOLOv11)', use_container_width=True)
            
        with col_res:
            st.subheader("🎯 Ringkasan Analisis Mutu")
            
            # Indikator Metrik Angka
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Biji", total_biji)
            m2.metric("Biji Normal", total_sehat)
            m3.metric("Biji Cacat", total_rusak)
            
            # Status Grade
            st.markdown(f"### Grade Final SNI: :{warna_grade}[{grade}]")
            st.info(f"**Total Nilai Cacat: {total_poin_cacat:.2f} Poin**")
            
            st.write("**Rincian Perhitungan Poin SNI:**")
            if len(tabel_rekap) > 0:
                df_rekap = pd.DataFrame(tabel_rekap)
                # Menyembunyikan index agar tabel lebih rapi
                st.dataframe(df_rekap, use_container_width=True, hide_index=True)
            else:
                st.success("Tidak ada cacat fisik yang ditemukan. Kualitas sempurna!")

        # Tabel Data Keseluruhan
        st.markdown("---")
        st.subheader("📋 Data Spesifikasi Ekstraksi Tiap Biji")
        if len(data_biji) > 0:
            df_biji = pd.DataFrame(data_biji)
            st.dataframe(df_biji, use_container_width=True, hide_index=True)