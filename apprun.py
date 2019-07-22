#sys.path.append('../input/jigsaw-net')
from absl import app
import run_jigsaw
# from tensorflow.python.platform import flags

argv = [
    "run_jigsaw.py",
    "--train_steps=1000000",
    "--num_core_per_host=1",
    "--predict_batch_size=4",
    "--do_train=False",
    "--do_eval=False",
    "--do_predict=True",
    "--eval_all_ckpt=False",
    "--eval_split='dev'",
    "--data_dir='../input/jigsaw-unintended-bias-in-toxicity-classification'",
    "--predict_dir='jigsaw/predict'",
    "--output_dir='jigsaw/output'",
    "--model_dir='jigsaw/model'",
    "--spiece_model_file='../input/jigsaw-net/spiece.model'",
    "--model_config_path='../input/jigsaw-net/xlnet_config.json'"
]

run_jigsaw.tf.app.run(run_jigsaw.main, argv)
