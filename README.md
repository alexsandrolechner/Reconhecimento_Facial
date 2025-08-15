# Reconhecimento Facial

Este projeto demonstra um sistema simples de reconhecimento facial utilizando a biblioteca [face_recognition](https://github.com/ageitgey/face_recognition) e o OpenCV.

## Estrutura

- `data/known_images`: fotos das pessoas que você deseja reconhecer.
- `data/known_videos`: vídeos contendo rostos conhecidos. O primeiro rosto detectado em cada vídeo é usado.
- `reconhecimento_facial.py`: script principal que realiza o reconhecimento em tempo real com a webcam.

## Requisitos

Instale as dependências com:

```bash
pip install -r requirements.txt
```

## Uso

1. Adicione as imagens e/ou vídeos das pessoas a serem reconhecidas nas pastas `data/known_images` e `data/known_videos`.
2. Conecte uma webcam ao computador.
3. Execute o script:

```bash
python reconhecimento_facial.py
```

Pressione `q` para encerrar a execução.

## Licença

Este projeto está disponível sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
