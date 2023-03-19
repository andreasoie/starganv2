import numpy as np


def lambda_rule_by_epoch(epoch, epoch_init, epoch_decay_start, epoch_decay_end): return 1.0 - max(0, epoch + epoch_init - epoch_decay_start) / float(epoch_decay_end + 1)                                                              

# def lambda_rule_by_iter(iteration, iter_init, iter_decay_start, iter_decay_end):
#     total_iter = 356  # total number of iterations in one epoch
#     epoch_init = iter_init // total_iter  # initial epoch
#     epoch_decay_start = iter_decay_start // total_iter  # epoch at which decay starts
#     epoch_decay_end = iter_decay_end // total_iter  # epoch at which decay ends
#     iter_per_epoch = total_iter  # number of iterations per epoch
#     iter_adjusted = iteration + iter_init
#     iter_decay_start = epoch_decay_start * iter_per_epoch
#     iter_decay_end = epoch_decay_end * iter_per_epoch
#     return 1.0 - max(0, iter_adjusted - iter_decay_start) / float(iter_decay_end + 1)

# test_iters = [0, 35_000, 72_000, 100_000, 141_999]
# test_epochs = [0, 100, 200, 300, 400]

# lr = 0.0002
# start_decay_at_iteration = 72_000
# total_decay_iterations = 72_000

# start_decay_at_epoch = 200
# total_decay_epochs = 200

# for iter, epoch, in zip(test_iters, test_epochs):
#     adlr = lr * lambda_rule_by_epoch(epoch, 0, start_decay_at_epoch, total_decay_epochs)
#     adep = lr * lambda_rule_by_iter(iter, 0, start_decay_at_iteration, total_decay_iterations)
    
#     is_close = np.isclose(adlr, adep)
#     print(adlr, adep, is_close)
    
# initial_lr = 0.0002
# start_epoch = 0
# start_decay_epoch = 200
# decay_epochs = 200

# for epoch in range(401):
#     lr = initial_lr * lambda_rule_by_epoch(epoch, start_epoch, start_decay_epoch, decay_epochs)
#     print(f"Epoch {epoch}: Learning rate = {lr:.6f}")
    
def lambda_rule_by_iter(iteration, iter_init, iter_decay_start, decay_iterations, iter_per_epoch):
    epoch_decay_start = iter_decay_start // iter_per_epoch
    epoch_num = (iteration + iter_init) // iter_per_epoch
    decay_epochs = decay_iterations // iter_per_epoch
    return 1.0 - max(0, epoch_num + 1 - epoch_decay_start) / float(decay_epochs + 1)

initial_lr = 0.0002
start_iter = 0
start_decay_iter = 200 * 355
decay_iterations = 200 * 355
batch_size = 8
total_images = 2840

iterations_per_epoch = total_images // batch_size

test_iterations = [0, 50, 70_000, 71_000, 100_000, 142_000]
for iteration in test_iterations:
    lr = initial_lr * lambda_rule_by_iter(iteration, start_iter, start_decay_iter, decay_iterations, iterations_per_epoch)
    print(f"Iteration {iteration}: Learning rate = {lr:.6f}")

    
# OUTPUT: 
# 0.0002 0.0002 True
# 0.0002 0.0002 True
# 0.0002 0.00019975525982784755 False
# 0.00010049751243781096 0.00012188338687024599 False
# 9.95024875621886e-07 5.078358572163589e-06 False