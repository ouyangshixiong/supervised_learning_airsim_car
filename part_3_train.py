#############################################################################################
#    MIT License
#
#    Copyright (c) AlexOuyang. All rights reserved.
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE
############################################################################################

############################################################################################
#    训练程序
#    输出 target_model/model.pdparams  target_model/adam.pdopt
#    By AlexOuyang
############################################################################################

import datetime
import numpy as np
import paddle
import random
from part_2_dataset_loader import AirSimCarDataSet
from part_3_model import CarBaselineModel
from visualdl import LogWriter

train_dataset = AirSimCarDataSet('train' )
test_dataset = AirSimCarDataSet('test' )

train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=10, drop_last=True)
test_loader = paddle.io.DataLoader(train_dataset, batch_size=32)
model = CarBaselineModel()
model.train()

epochs = 50
optim = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
loss_fn = paddle.nn.MSELoss()
val_loss = 1
total_batch = 581
for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader()):
        img = data[0]
        states = paddle.unsqueeze(data[1].astype('float32'), axis=1)
        label = data[2].astype('float32')
        predicts = model(img, states)
        loss = loss_fn(predicts, label)
        loss.backward()
        # if batch_id % 100 == 0:
        #     ct = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #     print("current_time:{}, epoch: {}, batch_id: {}, loss is: {}".format(ct, epoch, batch_id, loss.numpy()))
        optim.step()
        optim.clear_grad()
        if batch_id % 10 == 0:
            with LogWriter(logdir="./log") as writer:
                writer.add_scalar(tag="loss", step=(batch_id + total_batch*epoch), value=loss*100)

    #每个epoch做一下evaluate
    model.eval()
    for batch_id, data in enumerate(test_loader()):
        img = data[0]
        states = paddle.unsqueeze(data[1], axis=1)
        label = data[2]
        predicts = model(img, states)
        loss = loss_fn(predicts, label)
        break
    print('[eval] epoch: {} average loss is {}'.format(epoch, loss.numpy()))
    if val_loss > loss:
        val_loss = loss
        paddle.save(model.state_dict(), "target_model/model.pdparams")
        paddle.save(optim.state_dict(), "target_model/adam.pdopt")
        print('model saved')
    model.train()







