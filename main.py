from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pedalboard import Pedalboard, Compressor, HighpassFilter, Limiter, Gain, PeakFilter, HighShelfFilter, LowShelfFilter
from pedalboard.io import AudioFile
import tempfile
import os

# 🛡️ IMPORTA A NOSSA CALCULADORA DE LUFS
from lufs_radar import enforce_lufs_and_peak

app = FastAPI(title="RQS DSP API v4.0", description="Motor de Masterização e Homologação Comercial")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/masterize/parametric/")
async def masterize_parametric(
    file: UploadFile = File(...),
    estilo: str = Form("blue_team"),
    controle_sibilancia: bool = Form(False),
    largura_estereo: bool = Form(False),
    centro_foco: bool = Form(False)
):
    print(f"📡 [RECEBIDO] Faixa: {file.filename} | Estilo: {estilo.upper()}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
        tmp_in.write(await file.read())
        input_path = tmp_in.name

    effected_path = input_path.replace(".wav", "_effected.wav")
    final_output_path = input_path.replace(".wav", "_master_rqs.wav")

    try:
        # 🎛️ FASE 1: APLICAR EQ E COMPRESSÃO (Pedalboard)
        with AudioFile(input_path) as f:
            audio = f.read(f.frames)
            samplerate = f.samplerate

        if largura_estereo or centro_foco:
            mid = (audio[0] + audio[1]) / 2.0
            side = (audio[0] - audio[1]) / 2.0
            if largura_estereo: side = side * 1.3  
            if centro_foco: mid = mid * 1.15
            audio[0] = mid + side
            audio[1] = mid - side

        plugins = [HighpassFilter(cutoff_frequency_hz=25.0)]

        if controle_sibilancia:
            plugins.append(PeakFilter(cutoff_frequency_hz=7500.0, gain_db=-3.5, q=1.5))

        if estilo == "suno_style":
            plugins.extend([
                HighpassFilter(cutoff_frequency_hz=30.0),
                LowShelfFilter(cutoff_frequency_hz=100.0, gain_db=-1.0),
                PeakFilter(cutoff_frequency_hz=4000.0, gain_db=-1.5, q=1.0)
            ])
            largura_estereo = False 
        elif estilo == "blue_team":
            plugins.extend([
                LowShelfFilter(cutoff_frequency_hz=100.0, gain_db=1.5),
                HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=1.5),
                Compressor(threshold_db=-18.0, ratio=2.0, attack_ms=15.0, release_ms=150.0)
            ])
        elif estilo == "red_team":
            plugins.extend([
                LowShelfFilter(cutoff_frequency_hz=80.0, gain_db=3.0),
                PeakFilter(cutoff_frequency_hz=500.0, gain_db=-2.0, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=2.5),
                Compressor(threshold_db=-22.0, ratio=4.0, attack_ms=5.0, release_ms=80.0)
            ])

        board = Pedalboard(plugins)
        effected = board(audio, samplerate)

        with AudioFile(effected_path, 'w', samplerate, effected.shape[0]) as f:
            f.write(effected)

        print(f"🎛️ [FASE 1] Estilo {estilo.upper()} aplicado.")

        # 📊 FASE 2: O RADAR DE LUFS (Normalização e Limiter de Segurança)
        lufs_final = enforce_lufs_and_peak(effected_path, final_output_path)
        print(f"✅ [FASE 2] Homologação concluída. Volume estabilizado em: {lufs_final} LUFS.")

        return FileResponse(final_output_path, media_type="audio/wav", filename=f"RQS_PRO_{estilo}_{file.filename}")

    finally:
        # Varredura do Lixo de Processamento
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(effected_path): os.remove(effected_path)