import argparse
import numpy as np
import tensorflow as tf
from load import load_solar_data, load_wind_data
from model import GAN



def main(args):
    """
    Main function
    """
    #Check available device
    if tf.config.list_physical_devices('GPU'):
        print(tf.config.list_physical_devices('GPU'))
        tf.config.set_soft_device_placement(True)

    #Random seeds
    np.random.seed(42)

    #Load data and labels
    trX = None
    trY = None
    m = None
    if args.data.endswith('solar.csv'):
        trX, trY, m = load_solar_data(args.data, args.label)
    elif args.data.endswith('wind.csv'):
        trX, trY, m = load_wind_data(args.data, args.label)
    
    #Determine number of unique labels
    events_num = len(np.unique(trY))

    #Instantiate model
    GAN_model = GAN(epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    dim_y=events_num)
    
    #Start training
    GAN_model.fit(trX, trY)

    #Generate samples
    data_gen, labels_sampled = GAN_model.predict()

    #Rescaling
    data_gen = data_gen*m

    input('\nPress Enter to finish execution...')

    return data_gen
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        help='Select data set for training',
        type=str
    )
    parser.add_argument(
        '--label',
        help='Select labels corresponding to data set',
        type=str
    )
    parser.add_argument(
        '--epochs',
        help='Training iterations',
        default=5000,
        type=int
    )
    parser.add_argument(
        '--batch_size',
        help='Number of samples for one optimization',
        default=32,
        type=int
    )
    parser.add_argument(
        '--learning_rate',
        help='Learning rate for optimizer',
        default=1e-4,
        type=float
    )

    #Exclude the following structural parameters
    # parser.add_argument(
    #     '--image_shape',
    #     help='Define image shape (channels, height, width)',
    #     default=[1, 24, 24],
    #     tpye=list
    # )
    # parser.add_argument(
    #     '--dim_y',
    #     help='Sets number of channels (corresponds to the number of unique labels)',
    #     default=6,
    #     type=int
    # )
    # parser.add_argument(
    #     '--dim_z',
    #     help='Sets number of channels for sampled noise images',
    #     default=100,
    #     type=int
    # )
    # parser.add_argument(
    #     '--dim_W1',
    #     help='Layer dimension parameter',
    #     default=1024,
    #     type=int
    # )
    # parser.add_argument(
    #     '--dim_W2',
    #     help='Layer dimension parameter',
    #     default=128,
    #     type=int
    # )
    # parser.add_argument(
    #     '--dim_W3',
    #     help='Layer dimension parameter',
    #     default=64,
    #     type=int
    # )
    # parser.add_argument(
    #     '--dim_channel',
    #     help='Output dimension channels',
    #     default=1,
    #     type=int
    # )

    args = parser.parse_args()

    data_gen = main(args)