import time
import numpy as np
import paddle

import airsim

from part_3_model import CarBaselineModel

def get_image():
    image_request = airsim.ImageRequest(
            "0", airsim.ImageType.Scene, False, False
        )
    image_response = carClient.simGetImages([image_request])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 3)
    
    img = image_rgba[76:135,0:255,0:3].astype(float)
    return img.transpose((2, 0, 1))

ip_address="127.0.0.1"
carClient = airsim.CarClient(ip=ip_address)
carClient.reset()
carClient.enableApiControl(True)
# carClient.armDisarm(True)
time.sleep(0.01)
car_controls = airsim.CarControls()
car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0

model = CarBaselineModel()

saved = paddle.load("target_model/model.pdparams")
model.set_state_dict(saved)
model.eval()

while (True):
    car_state = carClient.getCarState()
    
    if (car_state.speed < 5):
        car_controls.throttle = 1.0
    else:
        car_controls.throttle = 0.0
    
    image = paddle.to_tensor(get_image()).astype('float32')
    image = paddle.unsqueeze(image, axis=0)
    states = paddle.to_tensor(np.array([car_controls.steering]).astype('float32'))
    states = paddle.unsqueeze(states, axis=0)
    predicts = model(image, states)
    predicts = round(predicts.numpy()[0][0], 2)
    car_controls.steering = predicts.item()
    
    print('Sending steering = {0}, throttle = {1}'.format(car_controls.steering, car_controls.throttle))
    
    carClient.setCarControls(car_controls)

    time.sleep(1)



