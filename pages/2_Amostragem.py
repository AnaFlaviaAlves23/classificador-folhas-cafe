import math

import streamlit as st
import torch
from PIL import Image

from estilo import (CEREJA, CLASSES, FOLHA, NOMES, OURO, barras,
                    carregar_modelo, cor_classe, iniciar_pagina, tf)

iniciar_pagina("Amostragem", "🌱", "Amostragem do talhão",
               "Envie várias folhas do mesmo talhão e veja a porcentagem de folhas doentes, com margem de erro.")


def intervalo_wilson(k, n, z=1.96):
    p = k / n
    centro = (p + z**2 / (2*n)) / (1 + z**2 / n)
    margem = z * math.sqrt(p*(1-p)/n + z**2 / (4*n**2)) / (1 + z**2 / n)
    return max(0, centro - margem), min(1, centro + margem)


def nivel(incidencia):
    # Limiares ilustrativos: ajuste com a recomendação agronômica da sua cultura.
    if incidencia < 0.05:
        return "Baixa", FOLHA
    if incidencia < 0.15:
        return "Atenção", OURO
    return "Alta", CEREJA


modelo = carregar_modelo()
arquivos = st.file_uploader("Fotos das folhas", type=['jpg', 'jpeg', 'png'],
                            accept_multiple_files=True)

if arquivos:
    previstos, miniaturas = [], []
    barra = st.progress(0.0, text="Analisando folhas...")
    for i, a in enumerate(arquivos):
        img = Image.open(a).convert('RGB')
        with torch.no_grad():
            previstos.append(CLASSES[modelo(tf(img).unsqueeze(0)).argmax(1).item()])
        mini = img.copy()
        mini.thumbnail((240, 240))
        miniaturas.append(mini)
        barra.progress((i + 1) / len(arquivos), text=f"Analisando folhas... {i + 1}/{len(arquivos)}")
    barra.empty()

    n = len(previstos)
    k = sum(1 for p in previstos if p != "Healthy")  # folhas previstas como diferentes de "Healthy"
    baixo, alto = intervalo_wilson(k, n)
    texto_nivel, cor_nivel = nivel(k / n)

    st.markdown(
        f'<div class="card"><div class="rotulo">Incidência estimada ({n} folhas)</div>'
        f'<div class="grande">{k/n:.0%}<span class="pill" style="background:{cor_nivel}">{texto_nivel}</span></div>'
        f'<div class="track"><div class="faixa" style="left:{baixo*100:.1f}%;width:{(alto-baixo)*100:.1f}%"></div>'
        f'<div class="ponto" style="left:{k/n*100:.1f}%"></div></div>'
        f'<div class="escala"><span>0%</span><span>Intervalo de 95%: {baixo:.0%} a {alto:.0%}</span><span>100%</span></div></div>',
        unsafe_allow_html=True,
    )
    if n < 20:
        st.info("Amostra pequena: colete mais folhas para uma estimativa mais confiável.")

    itens = [(NOMES[c], previstos.count(c) / n, cor_classe(c), str(previstos.count(c))) for c in CLASSES]
    st.markdown(f'<div class="card"><div class="rotulo">Folhas por classe</div>{barras(itens)}</div>',
                unsafe_allow_html=True)

    with st.expander("Ver as folhas analisadas", expanded=True):
        st.image(miniaturas, caption=[NOMES[p] for p in previstos], width=130)