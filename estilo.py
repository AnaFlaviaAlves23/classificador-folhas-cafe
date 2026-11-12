import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASSES = ['Cerscospora', 'Healthy', 'Leaf_rust', 'Miner', 'Phoma']
NOMES = {'Cerscospora': 'Cercosporiose', 'Healthy': 'Folha saudável',
         'Leaf_rust': 'Ferrugem', 'Miner': 'Bicho-mineiro', 'Phoma': 'Mancha de Phoma'}

CEREJA, FOLHA, OURO = "#B3261E", "#3E7A4A", "#C98A1B"

tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------- Ilustração: ramo de café com cerejas, folhas e grãos ----------
def _grao(x, y, r, s=1):
    return (f'<g transform="translate({x} {y}) rotate({r}) scale({s})">'
            '<ellipse rx="14" ry="20" fill="#6B4530"/>'
            '<path d="M0 -19 C-7 -7 7 7 0 19" stroke="#2B1810" stroke-width="2.6" fill="none" stroke-linecap="round"/></g>')


def _folha(x, y, r, s=1, cor="#3E7A4A"):
    return (f'<g transform="translate({x} {y}) rotate({r}) scale({s})">'
            f'<path d="M0 0 C18 -24 62 -24 84 0 C62 24 18 24 0 0Z" fill="{cor}"/>'
            '<path d="M4 0 L78 0" stroke="#2C5A36" stroke-width="2" stroke-linecap="round"/></g>')


def _cereja(x, y):
    return (f'<g transform="translate({x} {y})">'
            '<line x1="0" y1="-16" x2="0" y2="-6" stroke="#4B2E1E" stroke-width="2.5"/>'
            '<circle r="13" fill="#B3261E"/><circle cx="-4" cy="-4" r="3.2" fill="#fff" opacity=".35"/>'
            '<circle cy="10" r="2" fill="#7A1812"/></g>')


ARTE = ('<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">'
        + _grao(45, 192, -30) + _grao(100, 205, 25, .8) + _grao(335, 205, -15, 1.1) + _grao(372, 160, 40, .8)
        + '<path d="M10 200 C120 180 230 130 390 30" stroke="#6B4530" stroke-width="6" fill="none" stroke-linecap="round"/>'
        + _folha(60, 189, -35) + _folha(145, 160, 38, 1, "#4E8B57") + _folha(219, 127, -48, 1.1)
        + _folha(291, 89, 32, 1, "#4E8B57") + _folha(352, 53, -45, .9)
        + _cereja(112, 190) + _cereja(180, 162) + _cereja(260, 124) + _cereja(320, 90)
        + '</svg>')

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Figtree:wght@400;500;600&display=swap');
.stApp { background:#F5EBDD; font-family:'Figtree',sans-serif; color:#2B1810; }
.block-container { max-width:880px; padding-top:1.5rem; }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#EADBC6; }
h1,h2,h3 { font-family:'Fraunces',serif !important; color:#2B1810; }
.hero { position:relative; overflow:hidden; background:linear-gradient(135deg,#2B1810 0%,#4B2E1E 100%);
        border-radius:20px; padding:2.4rem 2.2rem; margin-bottom:1.3rem; min-height:200px; }
.hero .texto { position:relative; z-index:1; max-width:52%; }
.hero h1 { color:#F5EBDD !important; font-size:2.3rem; line-height:1.15; margin:0 0 .6rem; padding:0; }
.hero p { color:#D9C3A5; margin:0; font-size:1rem; }
.hero .arte { position:absolute; right:-8px; bottom:-8px; width:54%; }
.hero .arte svg { width:100%; height:auto; display:block; }
@media (max-width:700px) { .hero .texto { max-width:100%; } .hero .arte { width:85%; opacity:.25; } }
.card { background:#FFFBF5; border:1px solid #E4D3BC; border-radius:16px; padding:1.2rem 1.4rem; margin:.9rem 0; }
.card .rotulo { color:#7A5C45; font-size:.9rem; margin-bottom:.2rem; }
.card .grande { font-family:'Fraunces',serif; font-size:2.5rem; font-weight:700; color:#2B1810; line-height:1.1; }
.pill { display:inline-block; padding:.15rem .75rem; border-radius:999px; color:#fff; font-size:.85rem; font-weight:600; margin-left:.7rem; vertical-align:middle; }
.linha { display:flex; align-items:center; gap:.8rem; margin:.5rem 0; font-size:.95rem; }
.linha .nome { width:9.5rem; flex-shrink:0; }
.linha .fundo { flex:1; height:11px; background:#EADBC6; border-radius:99px; overflow:hidden; }
.linha .cheio { display:block; height:100%; border-radius:99px; }
.linha .valor { width:3rem; text-align:right; font-weight:600; }
.track { position:relative; height:12px; background:#EADBC6; border-radius:99px; margin:1.2rem 0 .3rem; }
.track .faixa { position:absolute; top:0; height:100%; background:#C9A57F; border-radius:99px; }
.track .ponto { position:absolute; top:-4px; width:20px; height:20px; margin-left:-10px; background:#4B2E1E;
                border:3px solid #FFFBF5; border-radius:50%; box-shadow:0 0 0 1px #4B2E1E; }
.escala { display:flex; justify-content:space-between; color:#7A5C45; font-size:.8rem; }
[data-testid="stFileUploaderDropzone"] { background:#FFFBF5; border:2px dashed #B08968; border-radius:16px; }
[data-testid="stImage"] img { border-radius:12px; }
</style>
"""


def iniciar_pagina(titulo_aba, icone, titulo, subtitulo):
    st.set_page_config(page_title=titulo_aba, page_icon=icone, layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="hero"><div class="arte">{ARTE}</div>'
                f'<div class="texto"><h1>{titulo}</h1><p>{subtitulo}</p></div></div>',
                unsafe_allow_html=True)


@st.cache_resource
def carregar_modelo():
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, len(CLASSES))
    m.load_state_dict(torch.load('melhor_v3b.pt', map_location='cpu'))
    m.eval()
    return m


def cor_classe(c):
    return FOLHA if c == 'Healthy' else CEREJA


def barras(itens):
    """itens: lista de (nome, fração 0-1, cor, texto)."""
    html = ""
    for nome, frac, cor, texto in itens:
        html += (f'<div class="linha"><span class="nome">{nome}</span>'
                 f'<span class="fundo"><span class="cheio" style="width:{frac*100:.1f}%;background:{cor}"></span></span>'
                 f'<span class="valor">{texto}</span></div>')
    return html