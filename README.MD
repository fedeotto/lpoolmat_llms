## Liverpool Materials repository for "LLMs for Materials and Chemistry Hackaton 2024"

This repository hosts projects developed by the Liverpool Materials team for the **LLMs for Materials and Chemistry hackathon**.

### Introduction
**Deep learning** models have demonstrated remarkable capabilities in predicting material properties, owing to their specific inductive biases.  Nevertheless, the limited size of chemical datasets presents  substantial challenges for leveraging these architectures effectively. <br> Here, we experiment with a hybrid approach that aims at integrating the architectural biases of deep learning alongside the abstract knowledge provided by LLMs. Specifically, we utilize Roost (https://www.nature.com/articles/s41467-020-19964-7), an attentional graph neural network for material property prediction that only leverages the stoichiometry of the underlying materials. We aggregate the material representations created by Roost with *context information* provided by the last layer of **MatBert/MatSciBert** (https://www.sciencedirect.com/science/article/pii/S2666389922000733 , https://www.nature.com/articles/s41524-022-00784-w). <br> <br>

![Example Image](example.png)

<a href="https://www.youtube.com/watch?v=fQVjRmM-Hf4">Here</a>, you can find a concise overview of our project. 