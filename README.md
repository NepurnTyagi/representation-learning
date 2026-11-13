This project was completed as **Assignment 2, Part 1 (A2, Part 1)** of **COL775: Deep Learning**, offered at **IIT Delhi** during **Semester 2, 2025–26**, under **Prof. Parag Singla**. The original [assignment PDF](https://www.cse.iitd.ac.in/~parags/teaching/col775/assignments/ass2/col775_a2.pdf) is available here.

## Dataset

This project uses the **COL775 A2 Extended CLEVR Dataset**, provided for the COL775 Deep Learning assignment at IIT Delhi.

The dataset is used for visual representation learning and downstream evaluation tasks, including object counting and color prediction.

### Dataset Sources

The dataset is available through two distribution platforms:

- **Hugging Face:** [aggr8/COL775-A2-Clevr-Extended-100k](https://huggingface.co/datasets/aggr8/COL775-A2-Clevr-Extended-100k)
- **Kaggle:** [COL775-A2-Clevr-Extended-100k-Tars](https://www.kaggle.com/datasets/ashish2001/col775-a2-clevr-extended-100k-tars/data)

The dataset is hosted externally and is not included in this repository.

Refer to the respective dataset pages for download instructions, organization, and licensing information.

### Model Checkpoints

The trained CLIP and DINO model checkpoints, including the linear probing models, are hosted on Hugging Face:

[Download Model Checkpoints](https://huggingface.co/NepurnTyagi/representation-learning)

### Dataset Observations

During the assignment, some duplicated samples were identified in the dataset. These samples contained identical images but different captions, with one or both captions differing between the duplicates.

The identified duplicates are documented in [`duplicated.csv`](./duplicated.csv), included in this repository for reference.
