# Meme Face Detector

Projeto local inspirado no video: a camera detecta o rosto da pessoa, extrai sinais simples de expressao/pose e mostra o meme mais parecido.

## O que ele faz

- Abre a webcam em tempo real.
- Detecta o rosto com MediaPipe Face Landmarker.
- Mede sorriso, boca aberta, sobrancelha, inclinacao da cabeca, centralizacao, iluminacao e proporcao do rosto.
- Detecta maos e dedos com MediaPipe Hand Landmarker.
- Reconhece gestos basicos como hang loose, joinha, apontando e mao aberta.
- Compara esses sinais com perfis de memes em `meme_profiles.json`.
- Mostra o meme vencedor e a porcentagem de match.

## Memes cadastrados

Os memes atuais ficam em `assets/memes/`:

- `cachorrodesconfiado.jpg`
- `cristianodedinho.jpg`
- `cristianonaogrita.jpg`
- `gatohangloose.jpg`
- `gatojoinha.jpg`
- `gatolingua.jpg`
- `macacobenca.jpg`
- `macacodecepção.jpg`
- `tiozinhocomoculos.jpg`

## Como rodar

```powershell
python -m pip install -r requirements.txt
python scripts/download_hand_model.py
python main.py
```

Se quiser usar uma webcam externa, primeiro veja quais cameras o OpenCV encontra:

```powershell
python main.py --list-cameras
```

Depois rode escolhendo o indice. Normalmente a camera do notebook e `0`, e a webcam externa fica em `1` ou `2`:

```powershell
python main.py --camera 1
```

Para poses com a mao, deixe os dedos visiveis perto do rosto. O detector usa MediaPipe Hand Landmarker, entao ele deve mostrar os pontos da mao e uma leitura como `hang loose dedos: 2 [1, 0, 0, 0, 1]`.

Quando a camera pisca a deteccao por poucos frames, o app segura a ultima mao/rosto detectado por um instante. Se aparecer `rosto previsto`, significa que ele esta usando essa memoria curta para evitar que o meme suma.

## Onde mexer

- `main.py`: app principal com camera, deteccao e comparacao.
- `meme_profiles.json`: pesos/traits de cada meme.
- `assets/memes/`: imagens dos memes.

## Proximos upgrades bons

- Criar uma tela para cadastrar meme novo sem editar JSON.
- Usar MediaPipe Face Mesh para medir olhos, boca e sobrancelhas com mais precisao.
- Salvar historico dos matches e ranking dos memes mais frequentes.
