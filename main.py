import io
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pedalboard import Pedalboard, Compressor, HighpassFilter, HighShelfFilter, LowShelfFilter, Limiter, Gain

app = FastAPI(title="RQS DSP CORE", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/masterize/parametric/")
async def masterize_audio(file: UploadFile = File(...), estilo: str = Form(...)):
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(
            status_code=400, 
            detail="⚠️ A RQS exige arquivos .WAV puros."
        )

    print(f"📡 [RECEBIDO] Faixa: {file.filename} | Payload: {estilo}")

    try:
        # 1. Extração do Áudio e Alinhamento de Matriz (O Patch de Matriz)
        audio_data, sample_rate = sf.read(file.file)
        audio_data_board = audio_data.T  # Transpõe de (Frames, Canais) para (Canais, Frames)

        # 2. Desacoplamento do Payload
        partes = estilo.split('_')
        estilo_solicitado = partes[0] if len(partes) > 0 else "equilibrado"
        intensidade_solicitada = partes[1] if len(partes) > 1 else "media"

        # 3. Metas de Volume (LUFS)
        target_lufs = -14.0 # Padrão Spotify/SoundCloud
        if intensidade_solicitada == "baixa":
            target_lufs = -16.0
        elif intensidade_solicitada == "alta":
            target_lufs = -9.0 # Volume brutal (EDM/Nu-Metal)

        # 4. Roteamento da Pedaleira Analógica (A Mágica DSP)
        board = Pedalboard()
        
        # Reduzimos um pouco o ganho inicial para dar espaço ("Headroom") para os plugins trabalharem
        board.append(Gain(gain_db=-3.0))

        if estilo_solicitado == "quente":
            board.append(LowShelfFilter(cutoff_frequency_hz=120, gain_db=2.5))
            board.append(HighShelfFilter(cutoff_frequency_hz=8000, gain_db=-1.5))
            board.append(Compressor(threshold_db=-15, ratio=3.0, attack_ms=5, release_ms=100))
            
        elif estilo_solicitado == "aberto":
            board.append(HighpassFilter(cutoff_frequency_hz=30))
            board.append(HighShelfFilter(cutoff_frequency_hz=6000, gain_db=3.5))
            board.append(Compressor(threshold_db=-14, ratio=2.5, attack_ms=10, release_ms=150))
            
        else: # Equilibrado
            board.append(HighpassFilter(cutoff_frequency_hz=20))
            board.append(Compressor(threshold_db=-16, ratio=2.0, attack_ms=15, release_ms=200))

        # 🛡️ BRICKWALL LIMITER: O Escudo contra Distorção Digital
        board.append(Limiter(threshold_db=-0.3, release_ms=50))

        # 5. Processamento dos Filtros e Re-alinhamento
        effected_audio_board = board(audio_data_board, sample_rate)
        effected_audio = effected_audio_board.T if effected_audio_board.ndim > 1 else effected_audio_board

        # 6. Elevação de Intensidade (LUFS)
        meter = pyln.Meter(sample_rate)
        current_lufs = meter.integrated_loudness(effected_audio)
        final_audio = pyln.normalize.loudness(effected_audio, current_lufs, target_lufs)

        # 🛡️ HARD CLIP: Garantia absoluta de que a onda não passa do limite do alto-falante
        final_audio = np.clip(final_audio, -1.0, 1.0)

        # 7. Empacotamento de Saída
        buffer = io.BytesIO()
        sf.write(buffer, final_audio, sample_rate, format='WAV')
        buffer.seek(0)
        
        print("🚀 [SUCESSO] Engenharia acústica finalizada. Retornando carga blindada.")
        return Response(content=buffer.getvalue(), media_type="audio/wav")

    except Exception as e:
        print(f"💥 [CRASH] Erro interno: {e}")
        raise HTTPException(status_code=500, detail="Erro no DSP.")