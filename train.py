"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
import torch
import numpy as np
from tqdm import tqdm
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer

print("**managed imports**\n")

# parse options
opt = TrainOptions().parse()
print("**parsed options**\n")

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)
print("**loaded dataset**\n")

# create trainer for our model
trainer = Pix2PixTrainer(opt)
print("**created trainer**\n")

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))
print("**created iterations counter**\n")

# create tool for visualization
visualizer = Visualizer(opt)
print("**created visualization tool**\n")
print(" image, image shape- min, max, label shape- min, max, instance shape- min, max" )

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # list_img = data_i['image'].numpy()
        # list_label = data_i['label'].numpy()
        # list_inst = data_i['instance'].numpy()
        # print( (data_i['path'][0]).split("/")[-1],
        #          list_img.shape, np.amin(list_img), np.amax(list_img),"\t",
        #          list_label.shape, np.amin(list_label), np.amax(list_label),"\t",
        #          list_inst.shape, np.amin(list_inst), np.amax(list_inst))

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
