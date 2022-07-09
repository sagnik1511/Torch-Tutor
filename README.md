<div align="center">
<h1>Torch-Trainer</h1>
<img width="600px" src="https://github.com/sagnik1511/Torch-Tutor/blob/main/extras/banner.png"><br>
<img src="https://forthebadge.com/images/badges/built-with-love.svg">
<img src="https://forthebadge.com/images/badges/made-with-python.svg">
<img src="https://forthebadge.com/images/badges/built-with-science.svg">
<h2>GOALS of the Project</h2>
<i>1. Reduces implementation time upto 50%.</i><br>
<i>2. Presents Eye-catching Training Job Monitor</i><br>
<i>3. Stores training data efficiently.</i><br>
<b>Visit at <a href="https://pypi.org/project/torch-tutor/0.0.1/#description">PyPI</a></b><br>
<b>Primary Release</b>
</div>

## Installation

1. Primary Requirements :

    a) Python >= 3.9

    b) Torch >= 1.11.0 + cu113
    
    Visit [Custom Installation PyTorch](https://pytorch.org/) to install the latest version(Date : 07-09-2022)
2. Install the `torch_tutor` package.

Procedures :

a) From PyPi :
```shell
pip install torch_tutor
```
b) From Repository :

Install the reporsitory :
```shell
git clone https://github.com/sagnik1511/Torch-Tutor
```
Go to the directory.
```shell
cd Torch_Tutor
python -m install torch_tutor
```

## Usage

```python3
from torch_tutor.core.trainer import Trainer
from torch_tutor.core.callbacks import CallBack

callback = CallBack(tracker="accuracy",
                    stop_epoch=5,
                    save_weights=True,
                    on="training",
                    save_directory="../weight_directory"
                    )

trainer = Trainer(train_dataset="<add your train_dataset_here>",
                  model="<add your model here>",
                  device="cpu")

optimizer_hyperparameter_dict = {"lr" : 0.0001}

trainer.compile(optimizer="<add your optimizer here>",
                loss_fn="<add your los function here>",
                metrics=["accuracy", "precision",...],
                optimizer_hparams=optimizer_hyperparameter_dict)

trainer.train(batch_size=32,
              num_epochs=50,
              training_steps=100,
              validation_set="<add your validation dataset here>",
              validation_steps=50,
              logging_index=10,
              shuffle=True,
              drop_last_batches=True,
              callback=callback)

```

Check in [Google Colab](https://colab.research.google.com/drive/1ce_sMuLcsHz-YCNLFYsQtM25qpYIZXIn?usp=sharing)

## Feature Description

### CallBack

```python
from torch_tutor.core.callbacks import CallBack
```

`tracker` [type: list] : On which metric the model will be tested. Currently, supporting *"accuracy"*, *"precision"*, *"recall"*, *"f1_score"* and *"mean_squared_error"*.

`stop_epoch` [type: int] : Number of epochs of continuous degradation after training stops.

`save_weights` [type: bool] : Flag to save best model.

`on` [type: str] : On which dataset the best results will be tracked. Takes either  *"training"* or *"validation"*.

`save_directory` [type: Path] : On which directory the best model will be saved.

---

### Trainer
```python
from torch_tutor.core.trainer import Trainer
```

`train_dataset` [type: torch.utils.data.Dataset] : The dataset used for training.

`model` [type: torch.nn.Module] : The model used for training.

`device` [type: str] : The device used for training . Currently supporting `cpu` and `cuda`.

#### compile

---

`optimizer` : The optimizer which will be used for training, e.g. `Adam`, `Adagrad`

`loss_fn` : The loss function that is used for backpropagation, e.g. `CrossEntropyLoss`, `MSELoss`

`metrics` [type: str] :  On which metric the model will be tested. Currently, supporting *"accuracy"*, *"precision"*, *"recall"*, *"f1_score"* and *"mean_squared_error"*.

`optimizer_hparams` [type: dict] : The parameters that are used inside Optimizer e.e *learning_rate*,*weight_decay*.


#### train

---

`batch_size` [type: int] : Batch size of the training and validation set.

`num_epochs` [type: int] : Number of epoch on which the model will be trained.

`training_steps` [type: int] : Number of batches per epoch on which model will be trained.

`validation_set` [type: None or torch.utils.data.Dataset] : On which the model will be validated.

`validation_steps` [type : int] : Number of batches per epoch on which model will be validated.

`logging_index` [type: int] : Number of indexes after which results will be shown.

`shuffle` [type: bool] : Flag to shuffle data indexes.

`drop_last_batches` [type: bool] : Flag on which the last small batches will be cut.

`callback` [type: torch_tutor.core.callback.callBack] : CallBack function.

---


<div align = "center">
<h3>If you get any errors while running the code, please make a PR.</h3>
<h1>Thanks for Visiting!!!</h1>
<h1>If you like the project, do ⭐</h1>
</div>

<div align = "center"><h1>Also follow me on <a href="https://github.com/sagnik1511">GitHub</a> , <a href="https://kaggle.com/sagnik1511">Kaggle</a> , <a href="https://in.linkedin.com/in/sagnik1511">LinkedIn</a></h1></div>



