import paddle
from part_2_dataset_loader import AirSimCarDataSet
from part_3_model import CarBaselineModel
from part_2_dataset_loader import draw_image_with_label

test_dataset = AirSimCarDataSet('test' )

test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, shuffle=True)
model = CarBaselineModel()

saved = paddle.load("target_model/model.pdparams")
model.set_state_dict(saved)
model.eval()

for batch_id, data in enumerate(test_loader()):
    img = data[0]
    states = data[1]
    label = data[2]
    predicts = model(img, states)
    print('predicts[63]:{}'.format(predicts[63]))
    draw_image_with_label(img[63],label[63],predicts[63])
    break
    
