import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model : torch.nn.Module,
                loader : DataLoader,
                optimizer : torch.optim,
                loss_fn: torch.Callable,
                n_epochs : int = 5)->torch.tensor:
    """Train a neural network model

    Args:
        model (torch.nn.Module): neural network
        loader (DataLoader): data 
        optimizer (torch.optim): optimizer (e.g. torch.optim.Adam)
        loss_fn (torch.Callable): loss function
        n_epochs (int, optional): number of epochs. Defaults to 5.

    Returns:
        torch.tensor: train errors
    """    
    
    errors = torch.tensor([])
    
    for epoch in tqdm(range(n_epochs)):
        batch_error = torch.tensor([])
        for X_batch, y_batch in loader:

            y_pred = model(X_batch)

            loss = loss_fn(y_pred, y_batch)
            batch_error = torch.cat([batch_error, torch.tensor([loss.item()])])
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

        errors = torch.cat([errors, torch.tensor([batch_error.mean()])])


    return errors

def model_accuracy(loader : DataLoader,
                   model : torch.nn.Module)->float:
    """Calculates the accuracy for a classifier

    Args:
        loader (DataLoader): data
        model (torch.nn.Module): trained model

    Returns:
        float: accuracy (number of correct predicts / total number of points)
    """    
    success_rate = torch.tensor([])

    with torch.no_grad():
        for X_batch, y_batch in loader:

            y_pred = model(X_batch)

            success_rate = torch\
                .cat([
                    success_rate, 
                    y_pred.argmax(dim = 1) == y_batch
                ])
            
    return success_rate.sum()/success_rate.shape[0]