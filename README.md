# Classificador de doenças em folhas de café

Projeto da disciplina SIN 323 (IA), UFV. Classifica folhas de café em cinco classes (folha saudável, ferrugem, cercosporiose, bicho-mineiro e mancha de phoma) com MobileNetV2, e inclui uma página de amostragem de talhão.

## Como rodar

Requer Python 3.12 (foi o testado). No Windows:

```
git clone https://github.com/AnaFlaviaAlves23/classificador-folhas-cafe.git
cd classificador-folhas-cafe
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8502
```

Depois abra http://localhost:8502 no navegador.