def read_annotation_csv(csv_path):
    """
    Membaca file anotasi EEG dan mengabaikan baris metadata.
    """
    try:
        df = pd.read_csv(csv_path, sep=',', comment='#')
        expected_cols = ['channel', 'start_time', 'stop_time', 'label', 'confidence']
        if list(df.columns[:5]) != expected_cols:
            df.columns = expected_cols  # kalau header tidak dikenali otomatis
        print("✅ File CSV berhasil dibaca dengan benar.")
        return df
    except Exception as e:
        raise ValueError(f"❌ Gagal membaca file CSV: {e}")


# -----------------------------------
# Sekarang panggil fungsi-nya
csv_file = '/content/aaaaaaac_s002_t000.csv'
df = read_annotation_csv(csv_file)

# Tampilkan hasilnya
print("\n📋 Isi DataFrame:")
print(df.head())
