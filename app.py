import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from estilo import (CLASSES, NOMES, barras, carregar_modelo, cor_classe,
                    iniciar_pagina, tf)

iniciar_pagina("Diagnóstico", "☕", "Diagnóstico de folhas de café",
               "Envie a foto de uma folha e veja a doença provável e onde o modelo olhou.")


def cor_jet(v):
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def gradcam(modelo, x):
    guarda = {}

    def hook(m, i, o):
        guarda['a'] = o
        o.register_hook(lambda g: guarda.update(g=g))

    h = modelo.features[-1].register_forward_hook(hook)
    modelo.zero_grad()
    saida = modelo(x)
    saida[0, saida.argmax(1).item()].backward()
    h.remove()
    w = guarda['g'].mean((2, 3), keepdim=True)
    cam = F.relu((w * guarda['a']).sum(1)).squeeze().detach().numpy()
    return cam / (cam.max() + 1e-8)


st.warning("Versão de desenvolvimento: o modelo reconhece 5 classes e foi validado com poucas fotos "
           "de campo. Serve como apoio e não substitui o agrônomo.")

modelo = carregar_modelo()
arquivo = st.file_uploader("Foto da folha", type=['jpg', 'jpeg', 'png'])

if arquivo is not None:
    img = Image.open(arquivo).convert('RGB')
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.softmax(modelo(x), dim=1)[0]
    melhor = prob.argmax().item()
    classe = CLASSES[melhor]
    cor = cor_classe(classe)

    st.markdown(
        f'<div class="card" style="border-left:8px solid {cor}"><div class="rotulo">Resultado</div>'
        f'<div class="grande" style="color:{cor}">{NOMES[classe]}</div>'
        f'<div class="rotulo" style="margin-top:.3rem">{prob[melhor].item():.0%} de confiança</div></div>',
        unsafe_allow_html=True,
    )

    cam = gradcam(modelo, x)
    mapa = Image.fromarray((cam * 255).astype('uint8')).resize(img.size)
    calor = (cor_jet(np.array(mapa) / 255.0) * 255).astype('uint8')
    sobre = Image.blend(img, Image.fromarray(calor), 0.45)

    c1, c2 = st.columns(2)
    c1.image(img, caption="Sua foto", width="stretch")
    c2.image(sobre, caption="Onde o modelo olhou (Grad-CAM)", width="stretch")
    st.caption("O mapa mostra onde o modelo olhou, mas não prova que o diagnóstico está certo.")

    itens = [(NOMES[c], p, cor_classe(c), f"{p:.0%}") for c, p in zip(CLASSES, prob.tolist())]
    st.markdown(f'<div class="card"><div class="rotulo">Probabilidade de cada classe</div>{barras(itens)}</div>',
                unsafe_allow_html=True)