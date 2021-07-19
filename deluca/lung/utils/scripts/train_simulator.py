import tqdm
import torch
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_simulator(
    sim,
    munger,
    u_window=5,
    p_window=3,
    device='cpu',
    exp_name='baseline',
    losscurve=False,
    train_key="train",
    test_key="test",
    batch_size=512,
    epochs=500,
    optimizer=torch.optim.Adam,
    optimizer_params={"lr": 1e-3},
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_params={"factor": 0.9, "patience": 15},
    loss_fn=torch.nn.L1Loss,
    loss_fn_params={},
    print_loss=1,
):
    print("Getting data loader")
    loader = munger.get_data_loader(
        key=train_key, u_window=u_window, p_window=p_window, batch_size=batch_size
    )

    print("Scaling train data")
    X_tensor, y_tensor = munger.scale_and_window(
        key=train_key, u_window=u_window, p_window=p_window
    )

    print("Scaling test data")
    X_test_tensor, y_test_tensor = munger.scale_and_window(
        key=test_key, u_window=u_window, p_window=p_window
    )

    sim = sim.to(device)
    optim = optimizer(sim.parameters(), **optimizer_params)
    schedule = scheduler(optim, **scheduler_params)
    loss_fn = loss_fn(**loss_fn_params)

    

    # start training
    for epoch in tqdm.tqdm(range(epochs)):
        # training
        sim.reset()
        sim.train()
        train_objs = AverageMeter()
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            sim.zero_grad()
            preds = sim.train_step(X_batch).squeeze()
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optim.step()
            train_objs.update(loss.item(), batch_size)
        schedule.step(loss)

        # log the loss to tensorboard
        if losscurve:
            writer.add_scalar('train_loss', train_objs.avg, epoch)

        # testing
        if epoch % print_loss == 0:
            sim.reset()
            sim.eval()
            with torch.no_grad():
                X_tensor = X_tensor.to(device)
                y_tensor = y_tensor.to(device)
                X_test_tensor = X_test_tensor.to(device)
                y_test_tensor = y_test_tensor.to(device)
                preds = sim(X_tensor).squeeze()
                train_loss = loss_fn(preds, y_tensor)
                preds = sim(X_test_tensor).squeeze()
                test_loss = loss_fn(preds, y_test_tensor)
                print(
                    f"Epoch {epoch:2d}: train={train_loss.item():.5f}, test={test_loss.item():.5f}"
                )
            # log the loss to tensorboard
            if losscurve:
                writer.add_scalar('test_loss', test_loss, epoch)

    if losscurve:
        writer.flush()
