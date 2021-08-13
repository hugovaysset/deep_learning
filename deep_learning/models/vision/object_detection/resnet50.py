import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, ReLU, GlobalAveragePooling2D, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU, Softmax
# from tensorflow.python.ops.gen_array_ops import Reshape

class ResBlock(Model):
    """
    INPUT : Conv -> BN -> LeakyReLu -> Conv -> Add -> BN -> LeakyReLU = OUTPUT
      |__________________________________________^
    """
    def __init__(self, channels, stride=1, index="0"):
        super(ResBlock, self).__init__(name=f"ResBlock_{index}")
        self.flag = (stride != 1)
        self.conv1 = Conv2D(channels, 3, stride, padding='same', name=f"conv_{index}_1")
        self.bn1 = BatchNormalization()
        self.leaky_relu = LeakyReLU()
        self.conv2 = Conv2D(channels, 3, padding='same', name=f"conv_{index}_2")
        self.bn2 = BatchNormalization()
        self.relu = ReLU()
        if self.flag:
            self.bn3 = BatchNormalization()
            self.conv3 = Conv2D(channels, 1, stride, name=f"conv_{index}_3")

    def call(self, x):
        x1 = self.conv1(x)  # if stride != 1, x1.shape < x.shape
        x1 = self.bn1(x1)
        x1 = self.leaky_relu(x1)
        x1 = self.conv2(x1)
        # x1 = self.bn2(x1)
        if self.flag:
            x = self.conv3(x)  # if stride != 1, we also reduce x.shape to perform x1 += 1
            # x = self.bn3(x)
        x1 = layers.add([x, x1])
        x1 = self.bn2(x1)
        x1 = self.leaky_relu(x1)
        return x1


def ResNet34(input_shape, nb_boxes, cell_shape, nb_classes):
    """
    ResNet34 for YOLO implementation
    """
    input = layers.Input(shape=input_shape, dtype="float32")

    # first treatments
    result = Conv2D(64, 7, strides=2, padding="same")(input)
    result = BatchNormalization()(result)
    result = MaxPooling2D((3, 3), strides=2)(result)

    # residual blocks
    result = ResBlock(64, stride=1, index="10")(result)
    result = ResBlock(64, stride=1, index="11")(result)
    result = ResBlock(64, stride=1, index="12")(result)

    result = ResBlock(128, index="21", stride=2)(result)  # reducing input shape
    result = ResBlock(128, stride=1, index="22")(result)
    result = ResBlock(128, stride=1, index="23")(result)
    result = ResBlock(128, stride=1, index="24")(result)

    result = ResBlock(256, index="31", stride=2)(result)  # reducing input shape
    result = ResBlock(256, stride=1, index="32")(result)
    result = ResBlock(256, stride=1, index="33")(result)
    result = ResBlock(256, stride=1, index="34")(result)
    result = ResBlock(256, stride=1, index="35")(result)
    result = ResBlock(256, stride=1, index="36")(result)

    result = ResBlock(512, index="41", stride=2)(result)  # reducing input shape
    result = ResBlock(512, stride=1, index="42")(result)
    result = ResBlock(512, stride=1, index="43")(result)

    # top layers
    result = Conv2D(nb_boxes * (5 + nb_classes), 1, padding="same")(result)  # predict nb_boxes * (5 + nb_classes) values per cell
    output = Reshape((cell_shape[1], cell_shape[0], nb_boxes, 5 + nb_classes))(result)  # reshape the predictions into (nb_boxes, 5) per cell

    return Model(inputs=input, outputs=output)


if __name__ == '__main__':
    model = ResNet34((512, 512, 1), 3, (16, 16), 1)

    model.summary()