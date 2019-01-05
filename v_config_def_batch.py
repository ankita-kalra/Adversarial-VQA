# paths
qa_path = '/home/akalra1/projects/adversarial-attacks/data/vqa_v1'  # directory containing the question and annotation jsons
train_path = '/home/akalra1/projects/adversarial-attacks/data/vqa_v1/train_im/train2014'  # directory of training images
val_path = '/home/akalra1/projects/adversarial-attacks/data/vqa_v1/val_im/val2014'  # directory of validation images
test_path = '/home/akalra1/projects/adversarial-attacks/data/vqa_v1/test_im/test2015'  # directory of test images
preprocessed_path = './resnet-14x14.h5'  # path where preprocessed features are saved to and loaded from
vocabulary_path = './vocab.json'  # path where the used vocabularies for question and answers are saved to

fdict_path = '/home/akalra1/projects/adversarial-attacks/show_ask_answer/pytorch-vqa/id_to_filename_'

task = 'OpenEnded'
dataset = 'mscoco'
vqa_model_path = '/home/akalra1/projects/adversarial-attacks/show_ask_answer/pytorch-vqa/logs/2017-08-04_00:55:19.pth'
# preprocess config
preprocess_batch_size = 8
image_size = 448  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

# training config
epochs = 50
batch_size = 1  #Need to modify code to support batch_sizes greater than 1
init_lr = 5e-3  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 8
max_answers = 3000
initial_lr = 1e-3  # default Adam lr
#loss multipliers
lambda_v = 1000
lambda_q = 0.01
