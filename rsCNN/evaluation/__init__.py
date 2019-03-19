from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os


def plot_history(history):
    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)

    # Epoch times and delays
    ax = axes.ravel()[0]
    epoch_time = [(finish - start).seconds for start, finish in zip(history['epoch_start'], history['epoch_finish'])]
    epoch_delay = [(start - finish).seconds for start, finish
                   in zip(history['epoch_start'][1:], history['epoch_finish'][:-1])]
    ax.plot(epoch_time, c='black', label='Epoch time')
    ax.plot(epoch_delay, '--', c='blue', label='Epoch delay')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Seconds')
    ax.legend()

    # Epoch times different view
    ax = axes.ravel()[1]
    dts = [epoch.strftime('%d %H:%M') for epoch in history['epoch_finish']]
    ax.hist(dts)
    ax.xaxis.set_tick_params(rotation=45)
    ax.set_ylabel('Epochs completed')

    # Loss
    ax = axes.ravel()[2]
    ax.plot(history['loss'][-160:], c='black', label='Training loss')
    if 'val_loss' in history:
        ax.plot(history['val_loss'][-160:], '--', c='blue', label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    # Learning rate
    ax = axes.ravel()[3]
    ax.plot(history['lr'][-160:], c='black')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning rate')

    # Add figure title
    plt.suptitle('Training History')
    return fig


def plot_model_summary_as_fig(model):
     stringlist = []
     model.summary(print_fn=lambda x: stringlist.append(x))
     model_summary_string = "\n".join(stringlist)

     fig, axes = plt.subplots(figsize=(8.5, 11), nrows=1, ncols=1)
     plt.text(0,0,model_summary_string,**{'fontsize':8,'fontfamily':'monospace'})
     plt.axis('off')
     plt.suptitle('CNN Summary')




def generate_eval_report(cnn, report_name):

    model_architecture = True
    training_history = True

    assert os.path.isdir(os.path.dirname(report_name)), 'Invalid report location'
   
    if (report_name.split('.')[-1] != 'pdf'):
      report_name = report_name + '.pdf'

    with PdfPages(report_name) as pdf: 

        #TODO: find max allowable number of lines, and carry over to the next page if violated
        if (model_architecture):
            plot_model_summary_as_fig(cnn.model)
            pdf.savefig()
        
        # TODO: handle subfigure spacing better
        if (training_history):
            fig = plot_history(cnn.history)
            pdf.savefig(fig)








   






