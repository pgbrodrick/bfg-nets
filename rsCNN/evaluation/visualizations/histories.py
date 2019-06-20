import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def plot_history(history: dict) -> [plt.Figure]:
    fig = plt.figure(figsize=(12, 10))
    gs1 = gridspec.GridSpec(2, 2)

    # Epoch times and delays
    ax = plt.subplot(gs1[0, 0])
    epoch_time = [(finish - start).seconds for start, finish in zip(history['epoch_start'], history['epoch_finish'])]
    epoch_delay = [(start - finish).seconds for start, finish
                   in zip(history['epoch_start'][1:], history['epoch_finish'][:-1])]
    ax.plot(epoch_time, c='black', label='Epoch time')
    ax.plot(epoch_delay, '--', c='blue', label='Epoch delay')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Seconds')
    ax.legend()

    # Epoch times different view
    ax = plt.subplot(gs1[0, 1])
    ax.hist([(dt - history['train_start']).seconds / 60 for dt in history['epoch_finish']])
    ax.set_xlabel('Minutes since start')
    ax.set_ylabel('Epochs completed')

    # Loss
    ax = plt.subplot(gs1[1, 0])
    ax.plot(history['loss'][-160:], c='black', label='Training loss')
    if 'val_loss' in history:
        ax.plot(history['val_loss'][-160:], '--', c='blue', label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    # Learning rate
    ax = plt.subplot(gs1[1, 1])
    ax.plot(history['lr'][-160:], c='black')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning rate')

    # Add figure title
    plt.suptitle('Model Training History')
    return [fig]
