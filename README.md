# 🧠 **Classificação Binária de Cães e Gatos com Logic Tensor Networks (LTN) e PyTorch**

---

## 📌 **1. Introdução**

Este projeto apresenta um estudo sobre o uso de **Logic Tensor Networks (LTN)** aplicadas a uma tarefa de **classificação binária**, utilizando **PyTorch** como backend de aprendizado profundo.

O trabalho parte de um:
- Um **notebook original** traduzido para o português pelo professor Edjard Mota, cujo objetivo era demonstrar o funcionamento de LTN em um cenário simplificado com **dados sintéticos**, ou seja, gerados no próprio código.
- Um trabalho de Tommaso Carraro, Luciano Serafini e Fabio Aiolli, disponível em https://github.com/tommasocarraro/LTNtorch, de título "LTNtorch: PyTorch Implementation of Logic Tensor Networks".

A partir desse notebook, realizamos uma **adaptação** (guiados pelo artigo) para aplicar a mesma estrutura lógica e de treinamento em um contexto real: a classificação de imagens de **gatos** e **cachorros**.

---

## 🎯 **2. Objetivo**

### **Notebook original**

* Demonstrar como LTN pode supervisionar um modelo neural simples.
* Trabalhar com dados gerados no próprio código, sem imagens reais.
* Servir como introdução aos blocos fundamentais: Predicados, Fórmulas, Variáveis e Funções de agregação lógica.

### **Adaptação desenvolvida**

* Substituir os dados sintéticos por um **dataset real de gatos e cachorros**.
* Preparar arquivos usando `ImageFolder` do PyTorch.
* Atualizar o predicado lógico para lidar com imagens RGB.
* Estruturar DataLoaders, pré-processamento, CNN e avaliação real.
* Demonstrar como LTN se comporta em um caso prático de visão computacional.

---

## **3. O que são Logic Tensor Networks (LTN)?**

LTN combina:

* **Deep Learning**: aprendizado a partir de dados.
* **Lógica Fuzzy**: graus de verdade entre `[0, 1]`.
* **Lógica de primeira ordem (FOL)**: relacionamentos simbólicos.

Isso permite impor **restrições lógicas** durante o treinamento.
Na prática, em vez de treinar apenas com uma função de perda padrão, definimos:

* Predicados (`P(x)`).
* Axiomas (`axiom_1 = Forall(x_dog, Dog(x_dog))`).
* Agregadores de perda baseados em lógica fuzzy.

Com isso, o modelo não aprende só pelos dados, mas também pelas **relações lógicas especificadas**.

---

## 🛠️ **4. Metodologia**

### 4.1 Notebook original (dados sintéticos)

O notebook fornecido pelo professor Edjard Mota continha:

* Um pequeno perceptron neural implementado em PyTorch.
* Dados numéricos 2D gerados dentro do código.
* Uma função lógica `P(x)` representando a probabilidade do ponto pertencer à classe positiva.

Esse modelo era válido para estudo, ótimo para demonstração, mas **sem relação com imagem**.

---

### 4.2 Adaptação para classificação de **gatos e cachorros**

A partir da base original, realizamos:

#### 1. Preparação do Dataset

Utilizando um dataset real organizado como:

```
data/
 ├── train/
 │     ├── cats/
 │     └── dogs/
```

Carregamento com:

```python
datasets.ImageFolder(..., transform=transforms.Compose([...]))
```

#### 2. Transformações

* `Resize`
* `ToTensor`

#### 3. DataLoaders

Criados com batch size adequado.

#### 4. Modelo CNN

Implementação de uma rede simples:

* Conv → ReLU → Pool
* Conv → ReLU → Pool
* Conv → ReLU → Pool
* Flatten + Dense
* Saída sigmoidal para probabilidade da classe "dog"

#### 5. Definição do Predicado LTN

```python
Dog = ltn.Predicate(cnn_model)
```

#### 6. Variáveis lógicas

```python
# x_dog: Variável representando o conceito "imagens de cachorro"
x_dog = ltn.Variable("x_dog", imgs_dogs)
# x_cat: Variável representando o conceito "imagens de gato"
x_cat = ltn.Variable("x_cat", imgs_cats)
```

#### 7. Base de Conhecimento

Usamos axiomas universais para cada subconjunto:

* Dog(x_dog) é verdadeiro
* Dog(x_cat) é falso

```python
# Axioma 1: Para todo x_dog, Dog(x_dog) deve ser verdade
axiom_1 = Forall(x_dog, Dog(x_dog))

# Axioma 2: Para todo x_cat, NÃO Dog(x_cat) deve ser verdade
axiom_2 = Forall(x_cat, Not(Dog(x_cat)))
```

---

## 📈 **5. Resultados**

A classificação binária usando LTN + CNN mostrou ser viável e coerente:

* O modelo aprende bem mesmo com poucas épocas.
* A equivalência lógica se mostrou eficiente para guiar o modelo.
* Resultados dependem muito da qualidade e quantidade do dataset.

---

## ▶️ **6. Como executar**

### 1️. Instale dependências

* Serão necessários: 
- torch
- torchvision
- ltn
- matplotlib

### 2️. Organize o dataset

Estruture como indicado anteriormente.

Para este projeto, **diminuímos a quantidade de imagens treino** para termos resultados mais rapidamente.

* Você pode baixar o dataset completo (~24988 imagens) aqui: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset 

* Você pode baixar o dataset modificado, **usado neste projeto**, aqui (~5045 imagens) aqui: https://drive.google.com/drive/folders/1eEiiv0X2xHn4bdeF0NqayNTCWcstEe63?usp=sharing

### 3️. Execute o notebook adaptado

Basta rodar célula por célula.

### 4️. Opcional: treinar em GPU no Google Colab

* Fazer upload do notebook
* Ativar GPU em:
  **Runtime → Change runtime type → GPU**

---

## **7. Discussão**

A adaptação mostra como LTN pode:

* Ser usada em conjunto com modelos neurais tradicionais.
* Expressar supervisão lógica diretamente.
* Generalizar bem mesmo em tarefas reais de visão computacional.

Enquanto o notebook original era apenas educacional e "ilustrativo", com dados gerados aleatoriamente, a adaptação permite ver LTN em ação com dados complexos, imagens ruidosas e classificadores convolucionais.


---

## **8. Conclusão**

Este trabalho mostra de forma clara:

1. A **estrutura e objetivo** do notebook original, voltado para entendimento básico de Logic Tensor Networks.
2. A **adaptação completa** para um problema real de visão computacional: classificar imagens de gatos e cachorros.
3. A flexibilidade do framework LTN em integrar lógica simbólica com aprendizado profundo.

A experiência demonstra que LTN é uma alternativa interessante para supervisionar redes neurais com restrições lógicas explícitas, especialmente em cenários onde **conhecimento humano** pode complementar ou guiar o processo de aprendizado.

---

## **9. Integrantes**

<div align="center">

<table>
  <tr>
    <td align="center" style="padding: 10px;">
      <a href="https://github.com/giovani-artil" target="_blank">
        <img src="https://github.com/giovani-artil.png" width="150px" style="border-radius: 50%;" alt="Giovani Carvalho"/><br />
        <span style="font-weight: bold; font-size: 16px; color: #333;">Giovani Carvalho</span>
      </a><br />
      <span style="font-size: 14px; color: #777;">Developer</span>
    </td>
    <td align="center" style="padding: 10px;">
      <a href="https://github.com/samuelcoelhoam" target="_blank">
        <img src="https://github.com/samuelcoelhoam.png" width="150px" style="border-radius: 50%;" alt="Jorge Coelho"/><br />
        <span style="font-weight: bold; font-size: 16px; color: #333;">Jorge Coelho</span>
      </a><br />
      <span style="font-size: 14px; color: #777;">Developer</span>
    </td>
    <td align="center" style="padding: 10px;">
      <a href="https://github.com/rehOtsedom12" target="_blank">
        <img src="https://github.com/rehOtsedom12.png" width="150px" style="border-radius: 50%;" alt="Renata Fernandes"/><br />
        <span style="font-weight: bold; font-size: 16px; color: #333;">Renata Fernandes</span>
      </a><br />
      <span style="font-size: 14px; color: #777;">Developer</span>
    </td>
    <td align="center" style="padding: 10px;">
      <a href="https://github.com/sofiaIcavino" target="_blank">
        <img src="https://github.com/sofiaIcavino.png" width="150px" style="border-radius: 50%;" alt="Sofia Moura"/><br />
        <span style="font-weight: bold; font-size: 16px; color: #333;">Sofia Moura</span>
      </a><br />
      <span style="font-size: 14px; color: #777;">Developer</span>
    </td>
   <td align="center" style="padding: 10px;">
      <a href="https://github.com/ValtXD" target="_blank">
        <img src="https://github.com/ValtXD.png" width="150px" style="border-radius: 50%;" alt="Francisco Felipe"/><br />
        <span style="font-weight: bold; font-size: 16px; color: #333;">Francisco Felipe</span>
      </a><br />
      <span style="font-size: 14px; color: #777;">Developer</span>
    </td>
  </tr>
</table>
</div>

* Giovani Artil Oliveira de Carvalho (giovaniartil@icomp.ufam.edu.br)
* Jorge Samuel Silva Coelho (samcoelho@icomp.ufam.edu.br)
* Renata Modesto Fernandes (renata.modesto@icomp.ufam.edu.br)
* Sofia Pinho Icavino Moura (sofiaicavino@icomp.ufam.edu.br)
* Francisco Felipe Barros dos Santos (francisco.santos@icomp.ufam.edu.br)

---

