"""
src/train.py

Shared training loop for the BiLSTM model.

Will contain:
- Training function with epoch loop
- Validation evaluation per epoch
- Loss computation (CrossEntropyLoss)
- Optimizer (Adam) and learning rate scheduler
- Early stopping based on validation loss/metric
- Model checkpoint saving (best model)
- TensorBoard / logging integration
"""