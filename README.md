![1rvg95](https://github.com/user-attachments/assets/0c6c7e0a-4d56-4077-8c7b-597a5d00c78f)


https://github.com/user-attachments/assets/2f788bb8-1797-4c7a-9e88-d47531059cda


# Projeto de Detector de Emoções

### Visão Geral do Projeto

Este projeto utiliza a biblioteca OpenCV e DeepFace para detectar emoções em tempo real a partir do vídeo capturado pela câmera do computador. As emoções detectadas são exibidas diretamente no vídeo, juntamente com um retângulo ao redor do rosto da pessoa.

### Funcionalidades

- Detecção de emoções em tempo real usando a câmera do computador.
- Exibição da emoção dominante em português diretamente no vídeo.

### Tecnologias Utilizadas

- **Python:** Linguagem de programação principal do projeto.
- **OpenCV:** Biblioteca para processamento de imagens e vídeo.
- **DeepFace:** Biblioteca para análise facial que suporta a detecção de emoções.

### Configuração e Instalação

#### Requisitos

- Python 3.7 ou superior
- OpenCV
- DeepFace

#### Clone o Repositório

```bash
git clone https://github.com/FlazO0/EmotionDetect.git
cd EmotionDetect
```

#### Instale as Dependências

Certifique-se de ter o Python instalado. Em seguida, instale os pacotes necessários:

```bash
pip install opencv-python deepface
```

### Uso

1. Certifique-se de que a câmera do seu computador está funcionando.
2. Execute o script Python:

    ```bash
    python app.py
    ```

3. A aplicação abrirá uma janela mostrando o vídeo da sua câmera com as emoções detectadas exibidas no rosto das pessoas.
4. Pressione `q` para sair da aplicação.
